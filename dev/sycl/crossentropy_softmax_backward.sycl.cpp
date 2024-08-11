/*
Kernels for crossentropy_softmax backward pass.

GPU Compile example for Intel PVC GPU with ahead-of-time compilation:
icpx -O3 -fsycl -fno-sycl-id-queries-fit-in-int -std=c++17 -xhost -DLLMSYCL -fp-model=precise \
     -qopenmp -qopenmp-simd -DOMP  -DTIMEPROFILE \
     -fsycl-targets=spir64_gen -Xs "-device pvc" -D_DISABLE_SG_SIZE_8 \
     crossentropy_softmax_backward.sycl.cpp -lsycl -lOpenCL -liomp5 -o crossentropy_softmax_backward

If encountering "error: identifier "M_PI" is undefined", add the following lines to the top of the file:

#define _USE_MATH_DEFINES
#include <math.h>  OR  #include <cmath>

version 1 is naive CPU port
./crossentropy_softmax_backward 1

version 2 uses sycl::nd_range/workgroups
./crossentropy_softmax_backward 2

Observations as of 08/08/2024: version 4 is marginally faster on CPU device with SG_SIZE={32}, SG_PER_WG={8}. All V1/V2/V3/V4
                               kernels have mostly similar perf on CPU.
                               version 4 is faster on GPU device with SG_SIZE={32}, SG_PER_WG>=2 or SG_SIZE={16}, SG_PER_WG>=4
                               version 3/4 have similar perf for larger sg_per_wg on GPU. V3 has better perf for smaller SG_PER_WG

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
double get_crossentropy_softmax_bwd_tflops(double elapsed_ms, size_t B, size_t T, size_t V) {
    // Time is in milliseconds
    return (double) (B * T * V * 3) / elapsed_ms * 1e-9;
}

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 crossentropy_softmax backward pass
void crossentropy_softmax_backward_cpu(float* dlogits,
                                       const float* dlosses, const float* probs, const int* targets,
                                       const size_t B, const size_t T, const size_t V, const size_t Vp) {
    timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // backwards through both softmax and crossentropy
    #pragma parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const size_t bt = b * T + t;
            float* dlogits_bt     = dlogits + bt * Vp;
            const float* probs_bt = probs + bt * Vp;
            const float dloss     = dlosses[bt];
            const int ix          = targets[bt];
            for (int i = 0; i < V; i++) {
                const float p         = probs_bt[i];
                const float indicator = (i == ix ? 1.0f : 0.0f);
                dlogits_bt[i]        += (p - indicator) * dloss;
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = get_elapsed_ms(start, end);
    double tflops = get_crossentropy_softmax_bwd_tflops(elapsed_ms, B, T, V);
    printf("kernel ref OMP | time %.4f ms | tflops %.2f\n", elapsed_ms, tflops);
}

// ----------------------------------------------------------------------------
// GPU kernels

sycl::event crossentropy_softmax_backward1(sycl::queue &queue, float* dlogits,
                                           const float* dlosses, const float* probs, const int* targets,
                                           const size_t B, const size_t T, const size_t V, const size_t Vp,
                                           const std::vector<sycl::event> &dependencies = {}) {
    // backwards through both softmax and crossentropy
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<1> item) {
            const size_t bt       = item.get_id(0);
            float* dlogits_bt     = dlogits + bt * Vp;
            const float* probs_bt = probs + bt * Vp;
            const float dloss     = dlosses[bt];
            const int ix          = targets[bt];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            for (int i = 0; i < V; i++) {
                float p = probs_bt[i];
                float indicator = (i == ix ? float(1.0f) : float(0));
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        };
        cgh.parallel_for<class ker_softmax_crossentropy_backward_1>(sycl::range<1>(B * T), kernel);
    });
    return last;
}

sycl::event crossentropy_softmax_backward2(sycl::queue &queue, float* dlogits,
                                           const float* dlosses, const float* probs, const int* targets,
                                           const size_t B, const size_t T, const size_t V, const size_t Vp,
                                           const std::vector<sycl::event> &dependencies = {}) {
    // backwards through both softmax and crossentropy
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<2> item) {
            const size_t bt       = item.get_id(0);
            const size_t v        = item.get_id(1);
            float* dlogits_bt     = dlogits + bt * Vp;
            const float* probs_bt = probs + bt * Vp;
            const float dloss     = dlosses[bt];
            const int ix          = targets[bt];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            const float p = probs_bt[v];
            const float indicator = (v == ix ? float(1.0f) : float(0));
            dlogits_bt[v] += (p - indicator) * dloss;
        };
        cgh.parallel_for<class ker_softmax_crossentropy_backward_2>(sycl::range<2>(B * T, V), kernel);
    });
    return last;
}

template <int SG_SIZE>
class ker_crossentropy_softmax_backward_3;

template<int SG_SIZE>
sycl::event crossentropy_softmax_backward3(sycl::queue &queue, float* dlogits,
                                           const float* dlosses, const float* probs, const int* targets,
                                           const size_t B, const size_t T, const size_t V, const size_t Vp, const int sg_per_wg,
                                           const std::vector<sycl::event> &dependencies = {}) {
    const int wg_size = sg_per_wg * SG_SIZE;
    // backwards through both softmax and crossentropy
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<2> item) {
            const size_t bt  = item.get_global_id(0);
            const size_t lid = item.get_local_linear_id();
            sycl::group gr   = item.get_group();

            float* dlogits_bt     = dlogits + bt * Vp;
            const float* probs_bt = probs + bt * Vp;
            const float dloss     = dlosses[bt];
            const int ix          = targets[bt];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            for (int i = lid; i < V; i += wg_size) {
                const float p = probs_bt[i];
                const float indicator = (i == ix ? float(1.0f) : float(0));
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        };
        cgh.parallel_for<class ker_crossentropy_softmax_backward_3<SG_SIZE>>(
                sycl::nd_range<2>(sycl::range<2>(B * T, wg_size), sycl::range<2>(1, wg_size)), kernel);
    });
    return last;
}

template <int SG_SIZE>
class ker_crossentropy_softmax_backward_4;

template<int SG_SIZE>
sycl::event crossentropy_softmax_backward4(sycl::queue &queue, float* dlogits,
                                           const float* dlosses, const float* probs, const int* targets,
                                           const size_t B, const size_t T, const size_t V, const size_t Vp, const int sg_per_wg,
                                           const std::vector<sycl::event> &dependencies = {}) {
    const int adjusted_sg_per_wg = std::min<size_t>((V + SG_SIZE - 1) / SG_SIZE, sg_per_wg);
    const int wg_size = adjusted_sg_per_wg * SG_SIZE;
    const size_t ceilV = ((V + wg_size - 1) / wg_size) * wg_size;
    // backwards through both softmax and crossentropy
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<2> item) {
            const size_t bt = item.get_global_id(0);
            const size_t v  = item.get_global_id(1);
            sycl::group gr  = item.get_group();
            if (v >= V) return;

            float* dlogits_bt     = dlogits + bt * Vp;
            const float* probs_bt = probs + bt * Vp;
            const float dloss     = dlosses[bt];
            const int ix          = targets[bt];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            const float p = probs_bt[v];
            const float indicator = (v == ix ? float(1.0f) : float(0));
            dlogits_bt[v] += (p - indicator) * dloss;
        };
        cgh.parallel_for<class ker_crossentropy_softmax_backward_4<SG_SIZE>>(
                sycl::nd_range<2>(sycl::range<2>(B * T, ceilV), sycl::range<2>(1, wg_size)), kernel);
    });
    return last;
}

// kernel version dispatch
sycl::event crossentropy_softmax_backward(int kernel_num,
                                          sycl::queue &queue, float* dlogits,
                                          const float* dlosses, const float* probs, const int* targets,
                                          const size_t B, const size_t T, const size_t V, const size_t Vp,
                                          const int sg_per_wg, const int sg_size) {
    switch (kernel_num) {
        case 1: {
            return crossentropy_softmax_backward1(queue, dlogits, dlosses, probs, targets, B, T, V, Vp);
        } break;
        case 2: {
            return crossentropy_softmax_backward2(queue, dlogits, dlosses, probs, targets, B, T, V, Vp);
        } break;
        case 3: {
            if (sg_size == 32) {
                return crossentropy_softmax_backward3<32>(queue, dlogits, dlosses, probs, targets, B, T, V, Vp, sg_per_wg);
            }
#ifndef _DISABLE_SG_SIZE_16
            else if (sg_size == 16) {
                return crossentropy_softmax_backward3<16>(queue, dlogits, dlosses, probs, targets, B, T, V, Vp, sg_per_wg);
            }
#endif
#ifndef _DISABLE_SG_SIZE_8
            else if (sg_size == 8) {
                return crossentropy_softmax_backward3< 8>(queue, dlogits, dlosses, probs, targets, B, T, V, Vp, sg_per_wg);
            }
#endif
            else {
                printf("Invalid sg_size\n");
                exit(2);
            }
        } break;
        case 4: {
            if (sg_size == 32) {
                return crossentropy_softmax_backward4<32>(queue, dlogits, dlosses, probs, targets, B, T, V, Vp, sg_per_wg);
            }
            else if (sg_size == 16) {
                return crossentropy_softmax_backward4<16>(queue, dlogits, dlosses, probs, targets, B, T, V, Vp, sg_per_wg);
            }
#ifndef _DISABLE_SG_SIZE_8
            else if (sg_size == 8) {
                return crossentropy_softmax_backward4< 8>(queue, dlogits, dlosses, probs, targets, B, T, V, Vp, sg_per_wg);
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
    float* dlogits = (float*)hostMallocCheck(B * T * Vp * sizeof(float), queue);
    float* dlosses = (float*)hostMallocCheck(B * T * sizeof(float), queue);
    float* probs   = (float*)hostMallocCheck(B * T * Vp * sizeof(float), queue);
    int* targets   = (int*)hostMallocCheck(B * T * sizeof(int), queue);

    queue.fill<float>(dlogits, float(0), B * T * Vp);
    make_random_float(dlosses, B * T);
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
    float* d_dlogits = (float*)deviceMallocCheck(B * T * Vp * sizeof(float), queue);
    float* d_dlosses = (float*)deviceMallocCheck(B * T * sizeof(float), queue);
    float* d_probs   = (float*)deviceMallocCheck(B * T * Vp * sizeof(float), queue);
    int* d_targets   = (int*)deviceMallocCheck(B * T * sizeof(int), queue);

    queue.fill<float>(d_dlogits, float(0), B * T * Vp);
    queue.memcpy(d_dlosses, dlosses, B * T * sizeof(float));
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
    crossentropy_softmax_backward_cpu(dlogits, dlosses, probs, targets, B, T, V, Vp);

    // Test kernel 1, 2
    for (int i = 1; i <= 2; ++i) {
        if (kernel_num == i) {
            printf("************************************************.\n");
            printf("Checking kernel set #%i.\n", kernel_num);
            crossentropy_softmax_backward(kernel_num, queue, d_dlogits, d_dlosses, d_probs, d_targets, B, T, V, Vp, 0, 0);
            queue.wait();
            validate_result(queue, d_dlogits, dlogits, "dlogits", B * T * Vp, 1e-3f);
            printf("All results match. Benchmarking kernel %i.\n", kernel_num);
            double elapsed_ms = benchmark_kernel(repeat_times, crossentropy_softmax_backward,
                                                 kernel_num, queue, d_dlogits, d_dlosses, d_probs,
                                                 d_targets, B, T, V, Vp, 0, 0);
            double tflops = get_crossentropy_softmax_bwd_tflops(elapsed_ms, B, T, V);
            printf("kernel %2i | time %.4f ms | tflops %.2f\n", kernel_num, elapsed_ms, tflops);

            queue.fill<float>(d_dlogits, float(0), B * T * Vp);
            queue.wait();
        }
        if (run_all) kernel_num++;
        printf("\n");
    }

    for (int i = 3; i <= 4; ++i) {
        // Test kernel 3, 4
        if (kernel_num == i) {
            printf("************************************************.\n");
            printf("Checking kernel set #%i.\n", kernel_num);
            for (int sg_size : supported_sg_sizes) {
                if (sg_size < 8 || sg_size > 32) {printf("Skipping sg_size = %i\n", sg_size); continue;}
                printf("Testing sg_size = %i\n", sg_size);
                for (int sg_per_wg = 1; sg_per_wg <= max_workgroup_size/sg_size; sg_per_wg <<= 1) {
                    printf("Checking kernel %i<%i, %i>.\n", kernel_num, sg_per_wg, sg_size);
                    crossentropy_softmax_backward(kernel_num, queue, d_dlogits, d_dlosses, d_probs, d_targets, B, T, V, Vp, sg_per_wg, sg_size);
                    queue.wait();
                    validate_result(queue, d_dlogits, dlogits, "dlogits", B * T * Vp, 1e-3f);
                    printf("All results match. Benchmarking kernel %i<%i, %i>.\n", kernel_num, sg_per_wg, sg_size);
                    double elapsed_ms = benchmark_kernel(repeat_times, crossentropy_softmax_backward,
                                                         kernel_num, queue, d_dlogits, d_dlosses, d_probs,
                                                         d_targets, B, T, V, Vp, sg_per_wg, sg_size);
                    double tflops = get_crossentropy_softmax_bwd_tflops(elapsed_ms, B, T, V);
                    printf("kernel %2i<%2i, %2i> | time %.4f ms | tflops %.2f\n", kernel_num, sg_per_wg, sg_size, elapsed_ms, tflops);

                    queue.fill<float>(d_dlogits, float(0), B * T * Vp);
                    queue.wait();
                }
                printf("\n");
            }
        }
        if (run_all) kernel_num++;
        printf("\n");
    }

    // free memory
    queue.wait_and_throw();
    sycl::free(dlogits, queue);
    sycl::free(dlosses, queue);
    sycl::free(probs, queue);
    sycl::free(targets, queue);
    sycl::free(d_dlogits, queue);
    sycl::free(d_dlosses, queue);
    sycl::free(d_probs, queue);
    sycl::free(d_targets, queue);
    return 0;
}
