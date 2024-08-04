/*
Kernels for layernorm forward pass.

GPU Compile example for Intel PVC GPU with ahead-of-time compilation:
icpx -O3 -fsycl -fno-sycl-id-queries-fit-in-int -std=c++17 -xhost -DLLMSYCL -fp-model=precise \
     -qopenmp -qopenmp-simd -DOMP  -DTIMEPROFILE \
     -fsycl-targets=spir64_gen -Xs "-device pvc" -D_DISABLE_SG_SIZE_8 \
     layernorm_forward.sycl.cpp -lsycl -lOpenCL -liomp5 -o layernorm_forward

version 1 is sycl::range<> port from CPU code to kernel: parallelizes over B*T, C, and loops over C
./layernorm_forward 1

version 2 is sycl::nd_range<> port of version 1 with different SG_SIZE, sg_per_wg
./layernorm_forward 2

TODO: Try oneDNN / oneMKL for version 3

Observations as of 08/04/2024: version 2 with SG_SIZE = 16, SG_PER_WG={1, 2} is faster on CPU device
                               version 2 with SG_SIZE = 32, SG_PER_WG=4 is faster on GPU device

*/

#include <stdio.h>
#include <stdlib.h>
#ifdef OMP
#include <omp.h>
#endif
#include "common.h"
#include <sycl/sycl.hpp>

#define SQRT sqrtf

// ----------------------------------------------------------------------------
// TFLOP/s
double get_layernorm_fwd_tflops(double elapsed_ms, size_t B, size_t T, size_t C) {
    // Time is in milliseconds
    // m: B * T * (C + 1)
    // v: B * T * (C * 3 + 1)
    // out: B * T * C * 4
    // s: B * T (2 + sqrt) // Assume sqrt involves 4 FLOPS
    return (double) (B * T * (C * 8 + 8)) / elapsed_ms * 1e-9;
}

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 layernorm forward pass
void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                           const float* inp, const float* weight, const float* bias,
                           int B, int T, int C) {
    timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    // Isn't eps = 1e-5 a problem if the variance is itself that
    // small or smaller?
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            const float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = get_elapsed_ms(start, end);
    double tflops = get_layernorm_fwd_tflops(elapsed_ms, B, T, C);
    printf("kernel ref SEQ | time %.4f ms | tflops %.2f\n", elapsed_ms, tflops);
}

// ----------------------------------------------------------------------------
// GPU kernels

// kernel 1 is the most naive layernorm kernel
sycl::event layernorm_forward1(sycl::queue &queue, float* out, float* mean, float* rstd,
                               const float* inp, const float* weight, const float* bias,
                               const int B, const int T, const int C,
                               const std::vector<sycl::event> &dependencies = {}) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    const float eps = float(1e-5);
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<1> item) {
            const size_t bt = item.get_id(0);
                    // seek to the input position inp[b,t,:]
                    const float* x = inp + bt * C;
                    // calculate the mean
                    float m = float(0);
                    for (int i = 0; i < C; i++) {
                        m += x[i];
                    }
                    m = m/C;
                    // calculate the variance (without any bias correction)
                    float v = float(0);
                    for (int i = 0; i < C; i++) {
                        float xshift = x[i] - m;
                        v += xshift * xshift;
                    }
                    v = v/C;
                    // calculate the rstd (reciprocal standard deviation)
                    float s = float(1.0) / SQRT(v + eps);
                    // seek to the output position in out[b,t,:]
                    float* out_bt = out + bt * C;
                    for (int i = 0; i < C; i++) {
                        float n = (s * (x[i] - m)); // normalize
                        float o = n * weight[i] + bias[i]; // scale and shift
                        out_bt[i] = o; // write
                    }
                    // cache the mean and rstd for the backward pass later
                    mean[bt] = m;
                    rstd[bt] = s;
        };
        cgh.parallel_for<class ker_layernorm_forward>(sycl::range<1>(B*T), kernel);
    });
    return last;
}

template <int SG_SIZE>
class ker_layernorm_forward_2;

template<int SG_SIZE>
sycl::event layernorm_forward2(sycl::queue &queue, float* out, float* mean, float* rstd,
                               const float* inp, const float* weight, const float* bias,
                               const int B, const int T, const int C, int sg_per_wg,
                               const std::vector<sycl::event> &dependencies = {}) {
    const float eps = float(1e-5);
    const int wg_size = sg_per_wg * SG_SIZE;
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
            const size_t bt           = item.get_global_id(0);
            const sycl::group gr      = item.get_group();
            const size_t lid          = item.get_local_linear_id(); //sg_group_id * SG_SIZE + sgr_cid;

            // seek to the input position inp[b,t,:]
            const float* x = inp + bt * C;
            // calculate the mean
            float m = float(0);
            for (int i = lid; i < C; i += wg_size) {
                m += x[i];
            }
            m = sycl::reduce_over_group(gr, m, sycl::plus<float>());
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = float(0);
            for (int i = lid; i < C; i += wg_size) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = sycl::reduce_over_group(gr, v, sycl::plus<float>());
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            float s = float(1.0) / SQRT(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + bt * C;
            for (int i = lid; i < C; i += wg_size) {
                float n = (s * (x[i] - m)); // normalize
                float o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[bt] = m;
            rstd[bt] = s;
        };
        cgh.parallel_for<class ker_layernorm_forward_2<SG_SIZE>>(
                sycl::nd_range<2>(sycl::range<2>(B*T, wg_size),
                                  sycl::range<2>(1, wg_size)), kernel);
    });
    return last;
}

// kernel version dispatch
sycl::event layernorm_forward(int kernel_num,
                              sycl::queue &queue, float* out, float* mean, float* rstd,
                              const float* inp, const float* weight, const float* bias,
                              const int B, const int T, const int C,
                              const int sg_per_wg, const int sg_size) {
    switch (kernel_num) {
        case 1: {
            return layernorm_forward1(queue, out, mean, rstd, inp, weight, bias, B, T, C);
        } break;
        case 2: {
            if (sg_size == 32) {
                return layernorm_forward2<32>(queue, out, mean, rstd, inp, weight, bias, B, T, C, sg_per_wg);
            }
#ifndef _DISABLE_SG_SIZE_16
            else if (sg_size == 16) {
                return layernorm_forward2<16>(queue, out, mean, rstd, inp, weight, bias, B, T, C, sg_per_wg);
            }
#endif
#ifndef _DISABLE_SG_SIZE_8
            else if (sg_size == 8) {
                return layernorm_forward2< 8>(queue, out, mean, rstd, inp, weight, bias, B, T, C, sg_per_wg);
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
    float* out    = (float*)hostMallocCheck(B * T * C * sizeof(float), queue);
    float* mean   = (float*)hostMallocCheck(B * T * sizeof(float), queue);
    float* rstd   = (float*)hostMallocCheck(B * T * sizeof(float), queue);
    float* inp    = (float*)hostMallocCheck(B * T * C * sizeof(float), queue);
    float* weight = (float*)hostMallocCheck(C * sizeof(float), queue);
    float* bias   = (float*)hostMallocCheck(C * sizeof(float), queue);

    queue.fill<float>(out, float(0), B * T * C);
    queue.fill<float>(mean, float(0), B * T);
    queue.fill<float>(rstd, float(0), B * T);
    make_random_float(inp, B * T * C);
    make_random_float(weight, C);
    make_random_float(bias, C);

    // move to GPU
    float* d_out    = (float*)deviceMallocCheck(B * T * C * sizeof(float), queue);
    float* d_mean   = (float*)deviceMallocCheck(B * T * sizeof(float), queue);
    float* d_rstd   = (float*)deviceMallocCheck(B * T * sizeof(float), queue);
    float* d_inp    = (float*)deviceMallocCheck(B * T * C * sizeof(float), queue);
    float* d_weight = (float*)deviceMallocCheck(C * sizeof(float), queue);
    float* d_bias   = (float*)deviceMallocCheck(C * sizeof(float), queue);
    
    queue.fill<float>(d_out, float(0), B * T * C);
    queue.fill<float>(d_mean, float(0), B * T);
    queue.fill<float>(d_rstd, float(0), B * T);
    queue.memcpy(d_inp, inp, B * T * C * sizeof(float));
    queue.memcpy(d_weight, weight, C * sizeof(float));
    queue.memcpy(d_bias, bias, C * sizeof(float));
    queue.wait();

    // read kernel_num from command line
    int kernel_num = 1;
    bool run_all = true;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
        printf("Using kernel %d\n", kernel_num);
        run_all = false;
    }
    const int repeat_times = 100;

    // first check the correctness of the kernel
    printf("Running reference CPU kernel.\n");
    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);

    // Test kernel 1
    if (kernel_num == 1) {
        printf("************************************************.\n");
        printf("Checking kernel set #%i.\n", kernel_num);
        layernorm_forward(kernel_num, queue, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, 0, 0);
        queue.wait();
        validate_result(queue, d_out, out, "out", B * T * C, 1e-5f);
        validate_result(queue, d_mean, mean, "mean", B * T, 1e-5f);
        validate_result(queue, d_rstd, rstd, "rstd", B * T, 1e-5f);
        printf("All results match. Benchmarking kernel %i.\n", kernel_num);
        double elapsed_ms = benchmark_kernel(repeat_times, layernorm_forward,
                                             kernel_num, queue, d_out, d_mean, d_rstd,
                                             d_inp, d_weight, d_bias,
                                             B, T, C, 0, 0);
        // napkin math: estimate the flops achieved
        // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
        double tflops = get_layernorm_fwd_tflops(elapsed_ms, B, T, C);
        printf("kernel %2i | time %.4f ms | tflops %.2f\n", kernel_num, elapsed_ms, tflops);

        queue.fill<float>(d_out, float(0), B * T * C);
        queue.fill<float>(d_mean, float(0), B * T);
        queue.fill<float>(d_rstd, float(0), B * T);
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
                layernorm_forward(kernel_num, queue, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, sg_per_wg, sg_size);
                queue.wait();
                validate_result(queue, d_out, out, "out", B * T * C, 1e-5f);
                validate_result(queue, d_mean, mean, "mean", B * T, 1e-5f);
                validate_result(queue, d_rstd, rstd, "rstd", B * T, 1e-5f);
                printf("All results match. Benchmarking kernel %i<%i, %i>.\n", kernel_num, sg_per_wg, sg_size);
                double elapsed_ms = benchmark_kernel(repeat_times, layernorm_forward,
                                                     kernel_num, queue, d_out, d_mean, d_rstd,
                                                     d_inp, d_weight, d_bias,
                                                     B, T, C, sg_per_wg, sg_size);
                // napkin math: estimate the flops achieved
                // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
                double tflops = get_layernorm_fwd_tflops(elapsed_ms, B, T, C);
                printf("kernel %2i<%2i, %2i> | time %.4f ms | tflops %.2f\n", kernel_num, sg_per_wg, sg_size, elapsed_ms, tflops);

                queue.fill<float>(d_out, float(0), B * T * C);
                queue.fill<float>(d_mean, float(0), B * T);
                queue.fill<float>(d_rstd, float(0), B * T);
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
    sycl::free(mean, queue);
    sycl::free(rstd, queue);
    sycl::free(inp, queue);
    sycl::free(weight, queue);
    sycl::free(bias, queue);
    sycl::free(d_out, queue);
    sycl::free(d_mean, queue);
    sycl::free(d_rstd, queue);
    sycl::free(d_inp, queue);
    sycl::free(d_weight, queue);
    sycl::free(d_bias, queue);
    return 0;
}
