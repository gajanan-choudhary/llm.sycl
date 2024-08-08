/*
Kernels for crossentropy forward pass.

GPU Compile example for Intel PVC GPU with ahead-of-time compilation:
icpx -O3 -fsycl -fno-sycl-id-queries-fit-in-int -std=c++17 -xhost -DLLMSYCL -fp-model=precise \
     -qopenmp -qopenmp-simd -DOMP  -DTIMEPROFILE \
     -fsycl-targets=spir64_gen -Xs "-device pvc" -D_DISABLE_SG_SIZE_8 \
     crossentropy_forward.sycl.cpp -lsycl -lOpenCL -liomp5 -o crossentropy_forward

If encountering "error: identifier "M_PI" is undefined", add the following lines to the top of the file:

#define _USE_MATH_DEFINES
#include <math.h>  OR  #include <cmath>

version 1 is naive CPU port
./crossentropy_forward 1

version 2 uses sycl::nd_range/workgroups
./crossentropy_forward 2

Observations as of 08/08/2024: version 2 is faster on CPU device with SG_SIZE={8}, SG_PER_WG={64, 32, 8}, with
                               SG_SIZE=16, SG_PER_WG=8 also being close, which we will use.
                               version 2 is faster on GPU device with {SG_SIZE, SG_PER_WG}={{2, 32}, {1, 32}, {2, 16}}

*/

#include <stdio.h>
#include <stdlib.h>
#ifdef OMP
#include <omp.h>
#endif
#include "common.h"
#include <sycl/sycl.hpp>

#define LOG logf /* sycl::log  */

// ----------------------------------------------------------------------------
// TFLOP/s
double get_crossentropy_fwd_tflops(double elapsed_ms, size_t B, size_t T) {
    // Time is in milliseconds
    return (double) (B * T * 4) / elapsed_ms * 1e-9;
}

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 crossentropy forward pass
void crossentropy_forward_cpu(float* losses, const float* probs, const int* targets,
                              const int B, const int T, const size_t Vp) {
    timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,V) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // loss = -log(probs[target])
            const float* probs_bt = probs + b * T * Vp + t * Vp;
            const int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = get_elapsed_ms(start, end);
    double tflops = get_crossentropy_fwd_tflops(elapsed_ms, B, T);
    printf("kernel ref OMP | time %.4f ms | tflops %.2f\n", elapsed_ms, tflops);
}

// ----------------------------------------------------------------------------
// GPU kernels

sycl::event crossentropy_forward1(sycl::queue &queue, float* losses,
                                  const float* probs, const int* targets,
                                  const int B, const int T, const int Vp,
                                  const std::vector<sycl::event> &dependencies = {}) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<1> item) {
            const size_t bt = item.get_id(0);
            // loss = -log(probs[target])
            const float* probs_bt = probs + bt * Vp;
            const int ix = targets[bt];
            losses[bt] = -LOG(probs_bt[ix]);
        };
        cgh.parallel_for<class ker_crossentropy_forward_1>(sycl::range<1>(B*T), kernel);
    });
    return last;
}

template <int SG_SIZE>
class ker_crossentropy_forward_2;

template<int SG_SIZE>
sycl::event crossentropy_forward2(sycl::queue &queue, float* losses,
                                  const float* probs, const int* targets,
                                  const int B, const int T, const int Vp, const int sg_per_wg,
                                  const std::vector<sycl::event> &dependencies = {}) {
    const int adjusted_sg_per_wg = std::min((B*T + SG_SIZE - 1) / SG_SIZE, sg_per_wg);
    const int wg_size = adjusted_sg_per_wg * SG_SIZE;
    const size_t ceilBT = ((B * T + wg_size - 1) / wg_size) * wg_size;

    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
            const size_t bt = item.get_global_id(0);
            if (bt >= B * T) return;
            // loss = -log(probs[target])
            const float* probs_bt = probs + bt * Vp;
            const int ix = targets[bt];
            losses[bt] = -LOG(probs_bt[ix]);
        };
        cgh.parallel_for<class ker_crossentropy_forward_2<SG_SIZE>>(
                sycl::nd_range<1>(sycl::range<1>(ceilBT), sycl::range<1>(wg_size)), kernel);
    });
    return last;
}

// kernel version dispatch
sycl::event crossentropy_forward(int kernel_num,
                                 sycl::queue &queue, float* losses,
                                 const float* probs, const int* targets,
                                 const int B, const int T, const int Vp,
                                 const int sg_per_wg, const int sg_size) {
    switch (kernel_num) {
        case 1: {
            return crossentropy_forward1(queue, losses, probs, targets, B, T, Vp);
        } break;
        case 2: {
            if (sg_size == 32) {
                return crossentropy_forward2<32>(queue, losses, probs, targets, B, T, Vp, sg_per_wg);
            }
            else if (sg_size == 16) {
                return crossentropy_forward2<16>(queue, losses, probs, targets, B, T, Vp, sg_per_wg);
            }
#ifndef _DISABLE_SG_SIZE_8
            else if (sg_size == 8) {
                return crossentropy_forward2< 8>(queue, losses, probs, targets, B, T, Vp, sg_per_wg);
            }
#endif
            else {
                printf("Invalid sg_size\n");
                exit(2);
            }
        } break;
        default: {
            printf("Invalid kernel number\n");
            exit(1);
        } break;
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    size_t B  = 32;
    size_t T  = 1024;
    size_t V  = 50257;
    size_t Vp = 50304; // padded vocabulary size

    // Catch asynchronous exceptions
    auto exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const &e) {
                std::cout << "Caught an asynchronous SYCL exception:\n"
                          << e.what() << std::endl;
            }
        }
    };

    // set up the device and queue
    sycl::device dev;
    sycl::queue queue(dev, exception_handler);
    const size_t max_workgroup_size = dev.get_info<sycl::info::device::max_work_group_size>();
    const std::vector<size_t> supported_sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    printf("Device maximum workgroup size = %zu\n", max_workgroup_size);
    printf("Device sub_group sizes = [");
    for (int sg_size : supported_sg_sizes) { printf("%i, ", sg_size); }
    printf("]\n");

    // setup oneMKL BLAS

    // create host memory of random numbers
    float* losses = (float*)hostMallocCheck(B * T * sizeof(float), queue);
    float* probs  = (float*)hostMallocCheck(B * T * Vp * sizeof(float), queue);
    int* targets  = (int*)hostMallocCheck(B * T * sizeof(int), queue);

    queue.fill<float>(losses, float(0), B * T);
    make_random_float_01(probs, B * T * Vp);
    make_random_int(targets, B * T, V);
    // Reset the padding to 0 as expected
    #pragma omp parallel for collapse(3)
    for(size_t b = 0; b < B; ++b) {
        for(size_t t = 0; t < T; ++t) {
            for (size_t v = V; v < Vp; ++v) {
                probs[b * T * Vp + t * Vp + v] = float(0);
            }
        }
    }

    // move to GPU
    float* d_losses = (float*)deviceMallocCheck(B * T * sizeof(float), queue);
    float* d_probs  = (float*)deviceMallocCheck(B * T * Vp * sizeof(float), queue);
    int* d_targets  = (int*)deviceMallocCheck(B * T * sizeof(int), queue);

    queue.fill<float>(d_losses, float(0), B * T);
    queue.memcpy(d_probs, probs, B * T * Vp * sizeof(float));
    queue.memcpy(d_targets, targets, B * T * sizeof(int));
    queue.wait();

    // read kernel_num from command line
    int kernel_num = 1;
    bool run_all = true;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
        printf("Using kernel %d\n", kernel_num);
        run_all = false;
    }
    const int repeat_times = 10;

    // first check the correctness of the kernel
    printf("Running reference CPU kernel.\n");
    crossentropy_forward_cpu(losses, probs, targets, B, T, Vp);

    // Test kernel 1
    if (kernel_num == 1) {
        printf("************************************************.\n");
        printf("Checking kernel set #%i.\n", kernel_num);
        crossentropy_forward(kernel_num, queue, d_losses, d_probs, d_targets, B, T, Vp, 0, 0);
        queue.wait();
        validate_result(queue, d_losses, losses, "losses", B * T, 1e-5f);
        printf("All results match. Benchmarking kernel %i.\n", kernel_num);
        double elapsed_ms = benchmark_kernel(repeat_times, crossentropy_forward,
                                             kernel_num, queue, d_losses, d_probs,
                                             d_targets, B, T, Vp, 0, 0);
        double tflops = get_crossentropy_fwd_tflops(elapsed_ms, B, T);
        printf("kernel %2i | time %.4f ms | tflops %.2f\n", kernel_num, elapsed_ms, tflops);

        queue.fill<float>(d_losses, float(0), B * T);
        queue.wait();
    }
    if (run_all) kernel_num++;
    printf("\n");

    // Test kernel 3, 4
    if (kernel_num == 2) {
        printf("************************************************.\n");
        printf("Checking kernel set #%i.\n", kernel_num);
        for (int sg_size : supported_sg_sizes) {
            if (sg_size < 8 || sg_size > 32) {printf("Skipping sg_size = %i\n", sg_size); continue;}
            printf("Testing sg_size = %i\n", sg_size);
            for (int sg_per_wg = 1; sg_per_wg <= max_workgroup_size/sg_size; sg_per_wg <<= 1) {
                printf("Checking kernel %i<%i, %i>.\n", kernel_num, sg_per_wg, sg_size);
                crossentropy_forward(kernel_num, queue, d_losses, d_probs, d_targets, B, T, Vp, sg_per_wg, sg_size);
                queue.wait();
                validate_result(queue, d_losses, losses, "losses", B * T, 1e-5f);
                printf("All results match. Benchmarking kernel %i<%i, %i>.\n", kernel_num, sg_per_wg, sg_size);
                double elapsed_ms = benchmark_kernel(repeat_times, crossentropy_forward,
                                                     kernel_num, queue, d_losses, d_probs,
                                                     d_targets, B, T, Vp, sg_per_wg, sg_size);
                double tflops = get_crossentropy_fwd_tflops(elapsed_ms, B, T);
                printf("kernel %2i<%2i, %2i> | time %.4f ms | tflops %.2f\n", kernel_num, sg_per_wg, sg_size, elapsed_ms, tflops);

                queue.fill<float>(d_losses, float(0), B * T);
                queue.wait();
            }
            printf("\n");
        }
    }
    if (run_all) kernel_num++;
    printf("\n");

    // free memory
    queue.wait_and_throw();
    sycl::free(losses, queue);
    sycl::free(probs, queue);
    sycl::free(targets, queue);
    sycl::free(d_losses, queue);
    sycl::free(d_probs, queue);
    sycl::free(d_targets, queue);
    return 0;
}
