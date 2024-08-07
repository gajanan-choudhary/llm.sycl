/*
Kernels for gelu forward pass.

GPU Compile example for Intel PVC GPU with ahead-of-time compilation:
icpx -O3 -fsycl -fno-sycl-id-queries-fit-in-int -std=c++17 -xhost -DLLMSYCL -fp-model=precise \
     -qopenmp -qopenmp-simd -DOMP  -DTIMEPROFILE \
     -fsycl-targets=spir64_gen -Xs "-device pvc" -D_DISABLE_SG_SIZE_8 \
     gelu_forward.sycl.cpp -lsycl -lOpenCL -liomp5 -o gelu_forward

If encountering "error: identifier "M_PI" is undefined", add the following lines to the top of the file:

#define _USE_MATH_DEFINES
#include <math.h>  OR  #include <cmath>

version 1 is naive CPU port
./gelu_forward 1

version 2 uses sycl::nd_range/workgroups
./gelu_forward 2

Observations as of 08/07/2024: version 2 with SG_SIZE=8, SG_PER_WG={64, ..., 512} perform faster on CPU. version 1 is pretty close
                               version 2 with SG_SIZE=32, SG_PER_WG={1, 2, 4} are faster on GPU @ 0.0932ms

*/

#include <stdio.h>
#include <stdlib.h>
#ifdef OMP
#include <omp.h>
#endif
#include "common.h"
#include <sycl/sycl.hpp>

#define SQRT sqrtf
#define TANH tanhf

// ----------------------------------------------------------------------------
// TFLOP/s
double get_gelu_fwd_tflops(double elapsed_ms, size_t N) {
    // Time is in milliseconds
    return (double) (N * 13) / elapsed_ms * 1e-9;
}

// ----------------------------------------------------------------------------
// CPU code reference

#define CPU_GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
// GPT-2 gelu forward pass
void gelu_forward_cpu(float* out, const float* inp, const int N) {
    timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        // Asume tanhf is equivalent to 4 flops
        out[i] = 0.5f * x * (1.0f + tanhf(CPU_GELU_SCALING_FACTOR * (x + cube)));
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = get_elapsed_ms(start, end);
    double tflops = get_gelu_fwd_tflops(elapsed_ms, N);
    printf("kernel ref OMP | time %.4f ms | tflops %.2f\n", elapsed_ms, tflops);
}

// ----------------------------------------------------------------------------
// GPU kernels

#define GELU_SCALING_FACTOR SQRT(2.0f / M_PI)

// kernel 1 is the most naive gelu kernel
sycl::event gelu_forward1(sycl::queue &queue, float* out,
                          const float* inp, const int N,
                          const std::vector<sycl::event> &dependencies = {}) {
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::id<1> id) {
            const float x = inp[id];
            float cube = 0.044715f * x * x * x;
            out[id] = 0.5f * x * (1.0f + TANH(GELU_SCALING_FACTOR * (x + cube)));
        };
        cgh.parallel_for<class ker_gelu_forward_1>(sycl::range<1>(N), kernel);
    });
    return last;
}

template <int SG_SIZE>
class ker_gelu_forward_2;

template<int SG_SIZE>
sycl::event gelu_forward2(sycl::queue &queue, float* out,
                          const float* inp, const int N, const int sg_per_wg,
                          const std::vector<sycl::event> &dependencies = {}) {
    const int adjusted_sg_per_wg = std::min((N + SG_SIZE - 1) / SG_SIZE, sg_per_wg);
    const int wg_size = adjusted_sg_per_wg * SG_SIZE;
    const int ceilN = ((N + wg_size - 1) & (-wg_size));

    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
            const size_t i = item.get_global_id(0);
            if (i >= N) return;
            const float x = inp[i];
            float cube = 0.044715f * x * x * x;
            out[i] = 0.5f * x * (1.0f + TANH(GELU_SCALING_FACTOR * (x + cube)));
        };
        cgh.parallel_for<class ker_gelu_forward_2<SG_SIZE>>(
                sycl::nd_range<1>(sycl::range<1>(ceilN), sycl::range<1>(wg_size)), kernel);
    });
    return last;
}

// kernel version dispatch
sycl::event gelu_forward(int kernel_num,
                         sycl::queue &queue, float* out,
                         const float* inp, const int N,
                         const int sg_per_wg, const int sg_size) {
    switch (kernel_num) {
        case 1: {
            return gelu_forward1(queue, out, inp, N);
        } break;
        case 2: {
            if (sg_size == 32) {
                return gelu_forward2<32>(queue, out, inp, N, sg_per_wg);
            }
#ifndef _DISABLE_SG_SIZE_16
            else if (sg_size == 16) {
                return gelu_forward2<16>(queue, out, inp, N, sg_per_wg);
            }
#endif
#ifndef _DISABLE_SG_SIZE_8
            else if (sg_size == 8) {
                return gelu_forward2< 8>(queue, out, inp, N, sg_per_wg);
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

    int B = 32;
    int T = 1024;
    int C = 768;
    size_t N = B * T * C;

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
    float* out    = (float*)hostMallocCheck(N * sizeof(float), queue);
    float* inp    = (float*)hostMallocCheck(N * sizeof(float), queue);

    queue.fill<float>(out, float(0), N);
    make_random_float(inp, N);

    // move to GPU
    float* d_out    = (float*)deviceMallocCheck(N * sizeof(float), queue);
    float* d_inp    = (float*)deviceMallocCheck(N * sizeof(float), queue);
    
    queue.fill<float>(d_out, float(0), N);
    queue.memcpy(d_inp, inp, N * sizeof(float));
    queue.wait();

    // read kernel_num from command line
    int kernel_num = 1;
    bool run_all = true;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
        printf("Using kernel %d\n", kernel_num);
        run_all = false;
    }
    const int repeat_times = 1000;

    // first check the correctness of the kernel
    printf("Running reference CPU kernel.\n");
    gelu_forward_cpu(out,inp, N);

    // Test kernel 1
    if (kernel_num == 1) {
        printf("************************************************.\n");
        printf("Checking kernel set #%i.\n", kernel_num);
        gelu_forward(kernel_num, queue, d_out, d_inp, N, 0, 0);
        queue.wait();
        validate_result(queue, d_out, out, "out", N, 1e-5f);
        printf("All results match. Benchmarking kernel %i.\n", kernel_num);
        double elapsed_ms = benchmark_kernel(repeat_times, gelu_forward,
                                             kernel_num, queue, d_out,
                                             d_inp, N, 0, 0);
        double tflops = get_gelu_fwd_tflops(elapsed_ms, N);
        printf("kernel %2i | time %.4f ms | tflops %.2f\n", kernel_num, elapsed_ms, tflops);

        queue.fill<float>(d_out, float(0), N);
        queue.wait();
    }
    if (run_all) kernel_num++;
    printf("\n");

    // Test kernel 2
    if (kernel_num == 2) {
        printf("************************************************.\n");
        printf("Checking kernel set #%i.\n", kernel_num);
        for (int sg_size : supported_sg_sizes) {
            if (sg_size < 8 || sg_size > 32) {printf("Skipping sg_size = %i\n", sg_size); continue;}
            printf("Testing sg_size = %i\n", sg_size);
            for (int sg_per_wg = 1; sg_per_wg <= max_workgroup_size/sg_size; sg_per_wg <<= 1) {
                printf("Checking kernel %i<%i, %i>.\n", kernel_num, sg_per_wg, sg_size);
                gelu_forward(kernel_num, queue, d_out, d_inp, N, sg_per_wg, sg_size);
                queue.wait();
                validate_result(queue, d_out, out, "out", N, 1e-5f);
                printf("All results match. Benchmarking kernel %i<%i, %i>.\n", kernel_num, sg_per_wg, sg_size);
                double elapsed_ms = benchmark_kernel(repeat_times, gelu_forward,
                                                     kernel_num, queue, d_out,
                                                     d_inp, N, sg_per_wg, sg_size);
                double tflops = get_gelu_fwd_tflops(elapsed_ms, N);
                printf("kernel %2i<%2i, %2i> | time %.4f ms | tflops %.2f\n", kernel_num, sg_per_wg, sg_size, elapsed_ms, tflops);

                queue.fill<float>(d_out, float(0), N);
                queue.wait();
            }
            printf("\n");
        }
    }
    if (run_all) kernel_num++;
    printf("\n");

    // free memory
    queue.wait_and_throw();
    sycl::free(out, queue);
    sycl::free(inp, queue);
    sycl::free(d_out, queue);
    sycl::free(d_inp, queue);
    return 0;
}
