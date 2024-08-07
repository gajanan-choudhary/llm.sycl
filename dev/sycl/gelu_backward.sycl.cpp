/*
Kernels for gelu backward pass.

GPU Compile example for Intel PVC GPU with ahead-of-time compilation:
icpx -O3 -fsycl -fno-sycl-id-queries-fit-in-int -std=c++17 -xhost -DLLMSYCL -fp-model=precise \
     -qopenmp -qopenmp-simd -DOMP  -DTIMEPROFILE \
     -fsycl-targets=spir64_gen -Xs "-device pvc" -D_DISABLE_SG_SIZE_8 \
     gelu_backward.sycl.cpp -lsycl -lOpenCL -liomp5 -o gelu_backward

If encountering "error: identifier "M_PI" is undefined", add the following lines to the top of the file:

#define _USE_MATH_DEFINES
#include <math.h>  OR  #include <cmath>

version 1 is naive CPU port
./gelu_backward 1

version 2 uses sycl::nd_range/workgroups
./gelu_backward 2

Observations as of 08/07/2024: both versions have uniform performance on CPU for all sg/wg sizes @ 1.77-1.85ms
                               but SG_SIZE=16, SG_PER_WG=8 is marginally fastest
                               version 2 with SG_SIZE=32, SG_PER_WG={2, 1} are faster on GPU @ 0.47ms

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
#define COSH coshf

// ----------------------------------------------------------------------------
// TFLOP/s
double get_gelu_bwd_tflops(double elapsed_ms, size_t N) {
    // Time is in milliseconds
    return (double) (N * 30) / elapsed_ms * 1e-9;
}

// ----------------------------------------------------------------------------
// CPU code reference

#define CPU_GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
// GPT-2 gelu backward pass
void gelu_backward_cpu(float* dinp, const float* inp, const float* dout, const int N) {
    timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x; // 4 ops
        float tanh_arg = CPU_GELU_SCALING_FACTOR * (x + cube); // 3 ops
        float tanh_out = tanhf(tanh_arg); // assumed 4 ops
        float coshf_out = coshf(tanh_arg); // assumed 4 ops
        float sech_out = 1.0f / (coshf_out * coshf_out); // 2 ops
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * CPU_GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x); // 12 ops
        dinp[i] = (float)(local_grad * (float)dout[i]); // 1 op
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = get_elapsed_ms(start, end);
    double tflops = get_gelu_bwd_tflops(elapsed_ms, N);
    printf("kernel ref OMP | time %.4f ms | tflops %.2f\n", elapsed_ms, tflops);
}

// ----------------------------------------------------------------------------
// GPU kernels

#define GELU_SCALING_FACTOR SQRT(2.0f / M_PI)

// kernel 1 is the most naive gelu kernel
sycl::event gelu_backward1(sycl::queue &queue, float* dinp,
                           const float* inp, const float* dout, const int N,
                           const std::vector<sycl::event> &dependencies = {}) {
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::id<1> id) {
            const float x = inp[id];
            const float cube = 0.044715f * x * x * x;
            const float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            const float tanh_out = TANH(tanh_arg);
            const float coshf_out = COSH(tanh_arg);
            const float sech_out = 1.0f / (coshf_out * coshf_out);
            const float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
            dinp[id] = (float)(local_grad * (float)dout[id]);
        };
        cgh.parallel_for<class ker_gelu_backward_1>(sycl::range<1>(N), kernel);
    });
    return last;
}

template <int SG_SIZE>
class ker_gelu_backward_2;

template<int SG_SIZE>
sycl::event gelu_backward2(sycl::queue &queue, float* dinp,
                           const float* inp, const float* dout, const int N, const int sg_per_wg,
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
            const float cube = 0.044715f * x * x * x;
            const float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            const float tanh_out = TANH(tanh_arg);
            const float coshf_out = COSH(tanh_arg);
            const float sech_out = 1.0f / (coshf_out * coshf_out);
            const float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
            dinp[i] = (float)(local_grad * (float)dout[i]);
        };
        cgh.parallel_for<class ker_gelu_backward_2<SG_SIZE>>(
                sycl::nd_range<1>(sycl::range<1>(ceilN), sycl::range<1>(wg_size)), kernel);
    });
    return last;
}

// kernel version dispatch
sycl::event gelu_backward(int kernel_num,
                          sycl::queue &queue, float* dinp,
                          const float* inp, const float* dout, const int N,
                          const int sg_per_wg, const int sg_size) {
    switch (kernel_num) {
        case 1: {
            return gelu_backward1(queue, dinp, inp, dout, N);
        } break;
        case 2: {
            if (sg_size == 32) {
                return gelu_backward2<32>(queue, dinp, inp, dout, N, sg_per_wg);
            }
#ifndef _DISABLE_SG_SIZE_16
            else if (sg_size == 16) {
                return gelu_backward2<16>(queue, dinp, inp, dout, N, sg_per_wg);
            }
#endif
#ifndef _DISABLE_SG_SIZE_8
            else if (sg_size == 8) {
                return gelu_backward2< 8>(queue, dinp, inp, dout, N, sg_per_wg);
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
    float* dinp   = (float*)hostMallocCheck(N * sizeof(float), queue);
    float* inp    = (float*)hostMallocCheck(N * sizeof(float), queue);
    float* dout   = (float*)hostMallocCheck(N * sizeof(float), queue);

    queue.fill<float>(dinp, float(0), N);
    make_random_float(inp, N);
    make_random_float(dout, N);

    // move to GPU
    float* d_dinp   = (float*)deviceMallocCheck(N * sizeof(float), queue);
    float* d_inp    = (float*)deviceMallocCheck(N * sizeof(float), queue);
    float* d_dout   = (float*)deviceMallocCheck(N * sizeof(float), queue);
    
    queue.fill<float>(d_dinp, float(0), N);
    queue.memcpy(d_inp, inp, N * sizeof(float));
    queue.memcpy(d_dout, dout, N * sizeof(float));
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
    gelu_backward_cpu(dinp, inp, dout, N);

    // Test kernel 1
    if (kernel_num == 1) {
        printf("************************************************.\n");
        printf("Checking kernel set #%i.\n", kernel_num);
        gelu_backward(kernel_num, queue, d_dinp, d_inp, d_dout, N, 0, 0);
        queue.wait();
        validate_result(queue, d_dinp, dinp, "dinp", N, 1e-5f);
        printf("All results match. Benchmarking kernel %i.\n", kernel_num);
        double elapsed_ms = benchmark_kernel(repeat_times, gelu_backward,
                                             kernel_num, queue, d_dinp,
                                             d_inp, d_dout, N, 0, 0);
        double tflops = get_gelu_bwd_tflops(elapsed_ms, N);
        printf("kernel %2i | time %.4f ms | tflops %.2f\n", kernel_num, elapsed_ms, tflops);

        queue.fill<float>(d_dinp, float(0), N);
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
                gelu_backward(kernel_num, queue, d_dinp, d_inp, d_dout, N, sg_per_wg, sg_size);
                queue.wait();
                validate_result(queue, d_dinp, dinp, "dinp", N, 1e-5f);
                printf("All results match. Benchmarking kernel %i<%i, %i>.\n", kernel_num, sg_per_wg, sg_size);
                double elapsed_ms = benchmark_kernel(repeat_times, gelu_backward,
                                                     kernel_num, queue, d_dinp,
                                                     d_inp, d_dout, N, sg_per_wg, sg_size);
                double tflops = get_gelu_bwd_tflops(elapsed_ms, N);
                printf("kernel %2i<%2i, %2i> | time %.4f ms | tflops %.2f\n", kernel_num, sg_per_wg, sg_size, elapsed_ms, tflops);

                queue.fill<float>(d_dinp, float(0), N);
                queue.wait();
            }
            printf("\n");
        }
    }
    if (run_all) kernel_num++;
    printf("\n");

    // free memory
    queue.wait_and_throw();
    sycl::free(dinp, queue);
    sycl::free(inp, queue);
    sycl::free(dout, queue);
    sycl::free(d_dinp, queue);
    sycl::free(d_inp, queue);
    sycl::free(d_dout, queue);
    return 0;
}
