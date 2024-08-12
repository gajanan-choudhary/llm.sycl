/*
Kernels for matmul forward pass.

GPU Compile example for Intel PVC GPU with ahead-of-time compilation:
icpx -O3 -fsycl -fno-sycl-id-queries-fit-in-int -std=c++17 -xhost -DLLMSYCL -fp-model=precise \
     -qopenmp -qopenmp-simd -DOMP  -DTIMEPROFILE \
     -fsycl-targets=spir64_gen -Xs "-device pvc" -D_DISABLE_SG_SIZE_8 \
     matmul_forward.sycl.cpp -lsycl -lOpenCL -liomp5 -o matmul_forward

version 1 is sycl::range<> port from CPU code to kernel: parallelizes over B*T, OC, and loops over C
./matmul_forward 1

version 2 is sycl::nd_range<> port of version 1 with different SG_SIZE
./matmul_forward 2

TODO: Try oneDNN / oneMKL for version 3

Observations as of 08/03/2024: version 1 is faster on CPU device @ 0.86 TFLOP/s on 224T SPR, unknown why
                               version 2 with SG_SIZE = 32, SG_PER_WG={4, 8} are fastest on GPU device @ 0.75 TFLOP/s on 1-tile PVC 512 EU
                   08/12/2024: version 5 using oneMKL Interfaces with cuBLAS/Intel oneMKL backends is obviously the fastest by more
                               than an order of magnitude @ 7 ms / 22 TFLOP/s on PVC, and 9.4 ms / 16.5 TFLOP/s on A100!
*/

#include <stdio.h>
#include <stdlib.h>
#ifdef OMP
#include <omp.h>
#endif
#include "common.h"
#include <sycl/sycl.hpp>

#ifdef ONEMKL_INTERFACES
#include "oneapi/mkl/blas.hpp"
#endif

// ----------------------------------------------------------------------------
// GFLOP/s

// ----------------------------------------------------------------------------
// CPU code reference
double get_matmul_fwd_tflops(double elapsed_ms, size_t B, size_t T, size_t C, size_t OC) {
    // Time is in milliseconds
    return (double) (B * T * C * OC * 2) / elapsed_ms * 1e-9;
}

void matmul_forward_cpu(float* out,
                        const float* inp, const float* weight, const float* bias,
                        int B, int T, int C, int OC) {
    timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            const float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                const float* wrow = weight + o*C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = get_elapsed_ms(start, end);
    double tflops = get_matmul_fwd_tflops(elapsed_ms, B, T, C, OC);
    printf("kernel ref OMP | time %.4f ms | tflops %.2f\n", elapsed_ms, tflops);
}

// ----------------------------------------------------------------------------
// GPU kernels

// kernel 1 is the most naive matmul kernel
sycl::event matmul_forward1(sycl::queue &queue, float* out,
                            const float* inp, const float* weight, const float* bias,
                            int B, int T, int C, int OC,
                            const std::vector<sycl::event> &dependencies = {})
{
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<2> item) {
            const size_t bt = item.get_id(0);
            const size_t o  = item.get_id(1);

            const size_t inp_ind_start = bt * C;
            const size_t wt_ind_start  = o * C;
            const size_t out_ind_start = bt * OC;
            float val = (bias != NULL) ? bias[o] : float(0);
            for (int i = 0; i < C; i++) {
                val += inp[inp_ind_start + i] * weight[wt_ind_start + i];
            }
            out[out_ind_start + o] = val;
        };
        cgh.parallel_for<class ker_matmul_forward_1>(sycl::range<2>(B*T, OC), kernel);
    });
    return last;
}

sycl::event matmul_forward2(sycl::queue &queue, float* out,
                            const float* inp, const float* weight, const float* bias,
                            int B, int T, int C, int OC,
                            const std::vector<sycl::event> &dependencies = {})
{
    constexpr std::uint16_t LOOP_UNROLL = 32;
    const size_t BT   = B * T;
    const size_t nOBT = (B*T + LOOP_UNROLL - 1) / LOOP_UNROLL;
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<2> item) {
            const size_t obt  = item.get_id(0) * LOOP_UNROLL;
            const size_t o    = item.get_id(1);
            const std::uint16_t nUnroll = (obt + LOOP_UNROLL < BT ? LOOP_UNROLL : BT - obt);
            const float *wt_o    = weight + o * C;
            const float *inp_obt = inp + obt * C;
            float *out_obt       = out + obt * OC + o;

            // we'll keep LOOP_UNROLL many results in registers
            float result[LOOP_UNROLL];
            // initialize the bias, if it exists
            for (std::uint16_t ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                result[ibt] = (bias != NULL) ? bias[o] : 0.0f;
            }
            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            for (int i = 0; i < C; i++) {
                const float w = wt_o[i];
                #pragma unroll(LOOP_UNROLL)
                for (std::uint16_t ibt = 0; ibt < nUnroll; ibt++) {
                    result[ibt] += inp_obt[ibt * C + i] * w;
                }
            }
            // write back results to main memory
            #pragma unroll(LOOP_UNROLL)
            for (std::uint16_t ibt = 0; ibt < nUnroll; ibt++) {
                out_obt[ibt * OC] = result[ibt];
            }
        };
        cgh.parallel_for<class ker_matmul_forward_2>(sycl::range<2>(nOBT, OC), kernel);
    });
    return last;
}

template <int SG_SIZE>
class ker_matmul_forward_3;

template<int SG_SIZE>
sycl::event matmul_forward3(sycl::queue &queue, float* out,
                            const float* inp, const float* weight, const float* bias,
                            int B, int T, int C, int OC, int sg_per_wg,
                            const std::vector<sycl::event> &dependencies = {})
{
    // Round up next multiple, sg_per_wg always assumed to be a power of 2
    const int ceilBT = (B*T + sg_per_wg - 1) & (-sg_per_wg);
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
            const size_t bt            = item.get_global_id(0);
            const size_t o             = item.get_global_id(1);
            const sycl::sub_group sgr  = item.get_sub_group();
            const std::uint8_t sgr_cid = sgr.get_local_id();

            if (bt >= B*T) return;

            const size_t inp_ind_start = bt * C;
            const size_t wt_ind_start  = o * C;
            const size_t out_ind_start = bt * OC;

            float val = float(0);
            // Split the following into optimzation below
            for (size_t i = sgr_cid; i < C; i += SG_SIZE) {
                val += inp[inp_ind_start + i] * weight[wt_ind_start + i];
            }
            // Reduce values in sub_group to lid == 0
            //val = sycl::reduce_over_group(sgr, val, sycl::plus<float>());
            for (std::uint8_t j = SG_SIZE >> 1; j > 0; j >>= 1) {
                val += sycl::shift_group_left(sgr, val, j);
            }

            if (sgr_cid == 0) {
                const float b = ((bias != NULL) ? bias[o] : float(0));
                out[out_ind_start + o] = val + b;
            }
        };
        cgh.parallel_for<class ker_matmul_forward_3<SG_SIZE>>(
                sycl::nd_range<3>(sycl::range<3>(ceilBT, OC, SG_SIZE),
                                  sycl::range<3>(sg_per_wg, 1, SG_SIZE)), kernel);
    });
    return last;
}

template <int SG_SIZE>
class ker_matmul_forward_4;

template<int SG_SIZE>
sycl::event matmul_forward4(sycl::queue &queue, float* out,
                            const float* inp, const float* weight, const float* bias,
                            int B, int T, int C, int OC, int sg_per_wg,
                            const std::vector<sycl::event> &dependencies = {})
{
    const size_t BT     = B * T;
    const size_t ceilBT = (B*T + SG_SIZE - 1) & (-SG_SIZE);
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
            const size_t o  = item.get_global_id(0);
            const size_t bt = item.get_global_id(1);
            if (bt >= BT) return;

            const float *wt_o   = weight + o * C;
            const float *inp_bt = inp + bt * C;
            float *out_bt       = out + bt * OC;

            // we'll keep LOOP_UNROLL many results in registers
            float result = (bias != NULL) ? bias[o] : 0.0f;

            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            #pragma unroll(4)
            for (int i = 0; i < C; i++) {
                result += inp_bt[i] * wt_o[i];
            }
            // write back results to main memory
            out_bt[o] = result;
        };
        cgh.parallel_for<class ker_matmul_forward_4<SG_SIZE>>(
                sycl::nd_range<2>(sycl::range<2>(OC, ceilBT),
                                  sycl::range<2>(1, SG_SIZE)), kernel);
    });
    return last;
}

#ifdef ONEMKL_INTERFACES
sycl::event matmul_forward_onemkl_interfaces(sycl::queue &queue, float* out,
                                             const float* inp, const float* weight, const float* bias,
                                             int B, int T, int C, int OC,
                                             const std::vector<sycl::event> &dependencies = {})
{
    // inp is (B*T, C), weight is (OC, C). Bias is (OC) and is added on separately later.
    // out is (B*T, OC). All inputs are in row-major format, but apparently
    // row-major is not supported, so we must flip-around some things to get
    // what we want with column-major GEMM.

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    const oneapi::mkl::transpose opA = oneapi::mkl::transpose::trans;
    const oneapi::mkl::transpose opB = oneapi::mkl::transpose::nontrans;
    sycl::event ev_blas_gemm = oneapi::mkl::blas::column_major::gemm(queue, opA, opB,
            OC, B*T, C, alpha, weight, C, inp, C, beta, out, OC, dependencies);

    // Add in bias:
    if (bias != NULL) {
        sycl::event ev_bias = queue.submit([&](sycl::handler &cgh) {
            cgh.depends_on(ev_blas_gemm);
            auto kernel = [=](sycl::item<2> item) {
                const size_t bt = item.get_id(0);
                const size_t o  = item.get_id(1);
                out[bt * OC + o] += bias[o];
            };
            cgh.parallel_for<class ker_matmul_forward_cublas_bias>(sycl::range<2>(B*T, OC), kernel);
        });
        return ev_bias;
    }
    else {
        return ev_blas_gemm;
    }
}
#endif // #ifdef ONEMKL_INTERFACES

// kernel version dispatch
sycl::event matmul_forward(int kernel_num,
                           sycl::queue &queue, float* out,
                           const float* inp, const float* weight, const float* bias,
                           int B, int T, int C, int OC,
                           const int sg_per_wg, const int sg_size) {
    switch (kernel_num) {
        case 1: {
            return matmul_forward1(queue, out, inp, weight, bias, B, T, C, OC);
        } break;
        case 2: {
            return matmul_forward2(queue, out, inp, weight, bias, B, T, C, OC);
        } break;
        case 3: {
            if (sg_size == 32) {
                return matmul_forward3<32>(queue, out, inp, weight, bias, B, T, C, OC, sg_per_wg);
            }
#ifndef _DISABLE_SG_SIZE_16
            else if (sg_size == 16) {
                return matmul_forward3<16>(queue, out, inp, weight, bias, B, T, C, OC, sg_per_wg);
            }
#endif
#ifndef _DISABLE_SG_SIZE_8
            else if (sg_size == 8) {
                return matmul_forward3< 8>(queue, out, inp, weight, bias, B, T, C, OC, sg_per_wg);
            }
#endif
            else {
                printf("Invalid sg_size\n");
                exit(2);
            }
        } break;
        case 4: {
            if (sg_size == 32) {
                return matmul_forward4<32>(queue, out, inp, weight, bias, B, T, C, OC, sg_per_wg);
            }
#ifndef _DISABLE_SG_SIZE_16
            else if (sg_size == 16) {
                return matmul_forward4<16>(queue, out, inp, weight, bias, B, T, C, OC, sg_per_wg);
            }
#endif
#ifndef _DISABLE_SG_SIZE_8
            else if (sg_size == 8) {
                return matmul_forward4< 8>(queue, out, inp, weight, bias, B, T, C, OC, sg_per_wg);
            }
#endif
            else {
                printf("Invalid sg_size\n");
                exit(2);
            }
        } break;
#ifdef ONEMKL_INTERFACES
        case 5: {
            return matmul_forward_onemkl_interfaces(queue, out, inp, weight, bias, B, T, C, OC);
        } break;
#endif // #ifdef ONEMKL_INTERFACES
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
    int OC = 768 * 4; // expansion of 4, e.g. in the MLP

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
    float* out    = (float*)hostMallocCheck(B * T * OC * sizeof(float), queue);
    float* inp    = (float*)hostMallocCheck(B * T * C * sizeof(float), queue);
    float* weight = (float*)hostMallocCheck(OC * C * sizeof(float), queue);
    float* bias   = (float*)hostMallocCheck(OC * sizeof(float), queue);

    queue.fill<float>(out, float(0), B * T * OC);
    make_random_float(inp, B * T * C);
    make_random_float(weight, OC * C);
    make_random_float(bias, OC);

    // move to GPU
    float* d_out    = (float*)deviceMallocCheck(B * T * OC * sizeof(float), queue);
    float* d_inp    = (float*)deviceMallocCheck(B * T * C * sizeof(float), queue);
    float* d_weight = (float*)deviceMallocCheck(C * OC * sizeof(float), queue);
    float* d_bias   = (float*)deviceMallocCheck(OC * sizeof(float), queue);
    
    queue.fill<float>(d_out, float(0), B * T * OC);
    queue.memcpy(d_inp, inp, B * T * C * sizeof(float));
    queue.memcpy(d_weight, weight, C * OC * sizeof(float));
    queue.memcpy(d_bias, bias, OC * sizeof(float));
    queue.wait();

    // read kernel_num from command line
    int kernel_num = 1;
    bool run_all = true;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
        printf("Using kernel %d\n", kernel_num);
        run_all = false;
    }
    const int repeat_times = 20;

    // first check the correctness of the kernel
    printf("Running reference CPU kernel.\n");
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    // Test kernels 1, 2
    for (int i = 1; i <=2; i++) {
        if (kernel_num == i) {
            printf("************************************************.\n");
            printf("Checking kernel set #%i.\n", kernel_num);
            matmul_forward(kernel_num, queue, d_out, d_inp, d_weight, d_bias, B, T, C, OC, 0, 0);
            queue.wait();
            validate_result(queue, d_out, out, "out", B * T * OC, 1e-4f);
            printf("All results match. Benchmarking kernel %i.\n", kernel_num);
            double elapsed_ms = benchmark_kernel(repeat_times, matmul_forward,
                                                 kernel_num, queue, d_out, d_inp, d_weight, d_bias,
                                                 B, T, C, OC, 0, 0);
            // napkin math: estimate the flops achieved
            // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
            double tflops = get_matmul_fwd_tflops(elapsed_ms, B, T, C, OC);
            printf("kernel %2i | time %.4f ms | tflops %.2f\n", kernel_num, elapsed_ms, tflops);

            queue.fill<float>(d_out, float(0), B * T * OC);
            queue.wait();
        }
        if (run_all) kernel_num++;
        printf("\n");
    }

    // Test kernels 3, 4
    for (int i = 3; i <=4; i++) {
        if (kernel_num == i) {
            printf("************************************************.\n");
            printf("Checking kernel set #%i.\n", kernel_num);
            for (int sg_size : supported_sg_sizes) {
                if (sg_size < 8 || sg_size > 32) {printf("Skipping sg_size = %i\n", sg_size); continue;}
                printf("Testing sg_size = %i\n", sg_size);
                for (int sg_per_wg = 1; sg_per_wg <= max_workgroup_size/sg_size; sg_per_wg <<= 1) {
                    printf("Checking kernel %i<%i, %i>.\n", kernel_num, sg_per_wg, sg_size);
                    matmul_forward(kernel_num, queue, d_out, d_inp, d_weight, d_bias, B, T, C, OC, sg_per_wg, sg_size);
                    queue.wait();
                    validate_result(queue, d_out, out, "out", B * T * OC, 1e-4f);
                    printf("All results match. Benchmarking kernel %i<%i, %i>.\n", kernel_num, sg_per_wg, sg_size);
                    double elapsed_ms = benchmark_kernel(repeat_times, matmul_forward,
                                                         kernel_num, queue, d_out, d_inp, d_weight, d_bias,
                                                         B, T, C, OC, sg_per_wg, sg_size);
                    // napkin math: estimate the flops achieved
                    // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
                    double tflops = get_matmul_fwd_tflops(elapsed_ms, B, T, C, OC);
                    printf("kernel %2i<%2i, %2i> | time %.4f ms | tflops %.2f\n", kernel_num, sg_per_wg, sg_size, elapsed_ms, tflops);

                    queue.fill<float>(d_out, float(0), B * T * OC);
                    queue.wait();
                }
                printf("\n");
            }
        }
        if (run_all) kernel_num++;
        printf("\n");
    }

#ifdef ONEMKL_INTERFACES
    if (kernel_num == 5) {
        printf("************************************************.\n");
        printf("Checking kernel set #%i.\n", kernel_num);
        matmul_forward(kernel_num, queue, d_out, d_inp, d_weight, d_bias, B, T, C, OC, 0, 0);
        queue.wait();
        validate_result(queue, d_out, out, "out", B * T * OC, 1e-4f);
        printf("All results match. Benchmarking kernel %i.\n", kernel_num);
        double elapsed_ms = benchmark_kernel(repeat_times, matmul_forward,
                                             kernel_num, queue, d_out, d_inp, d_weight, d_bias,
                                             B, T, C, OC, 0, 0);
        // napkin math: estimate the flops achieved
        // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
        double tflops = get_matmul_fwd_tflops(elapsed_ms, B, T, C, OC);
        printf("kernel %2i | time %.4f ms | tflops %.2f\n", kernel_num, elapsed_ms, tflops);

        queue.fill<float>(d_out, float(0), B * T * OC);
        queue.wait();
    }
    if (run_all) kernel_num++;
    printf("\n");
#endif // #ifdef ONEMKL_INTERFACES

    // free memory
    queue.wait_and_throw();
    sycl::free(out, queue);
    sycl::free(inp, queue);
    sycl::free(weight, queue);
    sycl::free(bias, queue);
    sycl::free(d_out, queue);
    sycl::free(d_inp, queue);
    sycl::free(d_weight, queue);
    sycl::free(d_bias, queue);
    return 0;
}
