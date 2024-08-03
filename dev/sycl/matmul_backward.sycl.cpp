/*
Kernels for matmul backward pass.

GPU Compile example for Intel PVC GPU with ahead-of-time compilation:
icpx -O3 -fsycl -fno-sycl-id-queries-fit-in-int -std=c++17 -xhost -DLLMSYCL -fp-model=precise \
     -qopenmp -qopenmp-simd -DOMP  -DTIMEPROFILE \
     -fsycl-targets=spir64_gen -Xs "-device pvc" -D_DISABLE_SG_SIZE_8 \
     matmul_backward.sycl.cpp -lsycl -lOpenCL -liomp5 -o matmul_backward

version 1 is sycl::range<> port from CPU code to kernel: parallelizes over B*T, OC, and loops over C
./matmul_backward 1

version 2 is sycl::nd_range<> port of version 1 with different SG_SIZE
./matmul_backward 2

TODO: Try oneDNN / oneMKL for version 3

Observations as of 08/03/2024: version 2 with SG_SIZE=32, SG_PER_WG=1 is fastest on CPU device @ 0.27 TFLOP/s on 224T SPR
                               version 2 with SG_SIZE=32, SG_PER_WG={1, 2, 32} are fastest on GPU device @ 0.53 TFLOP/s on 1-tile PVC 512 EU

*/

#include <stdio.h>
#include <stdlib.h>
#ifdef OMP
#include <omp.h>
#endif
#include "common.h"
#include <sycl/sycl.hpp>

// ----------------------------------------------------------------------------
// GFLOP/s

// ----------------------------------------------------------------------------
// CPU code reference
double get_matmul_bwd_tflops(double elapsed_ms, size_t B, size_t T, size_t C, size_t OC) {
    // Time is in milliseconds
    // dinp: B * T * OC * C * 2
    // dweight: B * T * OC * C * 2
    // dbias: B * T * OC * 1
    return (double) ((B * T * C * OC * 4) + (B * T * OC)) / elapsed_ms * 1e-9;
}

void matmul_backward_cpu(float* dinp, float* dweight, float* dbias,
                         float* dout, float* inp, float* weight,
                         int B, int T, int C, int OC) {
    timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * OC + t * OC;
            float* dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float* wrow = weight + o*C;
                float d = dout_bt[o];
                for (int i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #pragma omp parallel for
    for (int o = 0; o < OC; o++) {
        double sum = 0.0;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float* dout_bt = dout + b * T * OC + t * OC;
                float* inp_bt = inp + b * T * C + t * C;
                float* dwrow = dweight + o*C;
                float d = dout_bt[o];
                if (dbias != NULL) { sum += d; }
                for (int i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
        if (dbias != NULL){dbias[o] = sum;}
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = get_elapsed_ms(start, end);
    double tflops = get_matmul_bwd_tflops(elapsed_ms, B, T, C, OC);
    printf("kernel ref OMP | time %.4f ms | tflops %.2f\n", elapsed_ms, tflops);
}

// ----------------------------------------------------------------------------
// GPU kernels

// kernel 1 is the most naive matmul kernel
sycl::event matmul_backward1(sycl::queue &queue, float* dinp, float* dweight, float* dbias,
                             const float* dout, const float* inp, const float* weight,
                             int B, int T, int C, int OC,
                             const std::vector<sycl::event> &dependencies = {})
{
    // backward into dinp first, parallelize over B*T, C
    // dinp depends on dout, weight
    sycl::event ev_dinp = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<2> item) {
            const size_t bt = item.get_id(0);
            const size_t c = item.get_id(1);
            const float* dout_bt = dout + bt * OC;
            float* dinp_bt = dinp + bt * C;
            float val = float(0);
            for (int o = 0; o < OC; o++) {
                const float* wrow = weight + o*C;
                float d = dout_bt[o];
                val += wrow[c] * d;
            }
            dinp_bt[c] += val;
        };
        cgh.parallel_for<class ker_matmul_backward1_dinp>(sycl::range<2>(B*T, C), kernel);
    });

    // backward into weight, parallelize over output channels OC, C
    sycl::event ev_dweight = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<2> item) {
            const size_t o   = item.get_id(0);
            const size_t c   = item.get_id(1);
            const size_t lid = item.get_linear_id();
            float val = float(0);
            for (int bt = 0; bt < B*T; bt++) {
                val += inp[bt * C + c] * dout[bt * OC + o];
            }
            dweight[lid] += val;
        };
        cgh.parallel_for<class ker_matmul_backward1_dweight>(sycl::range<2>(OC, C), kernel);
    });

    // backward into bias, parallelize over output channels OC
    sycl::event ev_dbias = sycl::event();
    if (dbias != NULL) {
        ev_dbias = queue.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dependencies);
            auto kernel = [=](sycl::item<1> item) {
                const size_t o = item.get_id(0);
                float sum = float(0);
                for (int bt = 0; bt < B*T; bt++) {
                    const float* dout_bt = dout + bt * OC;
                    sum += dout_bt[o];
                }
                dbias[o] += sum;
            };
            cgh.parallel_for<class ker_matmul_backward1_dbias>(sycl::range<1>(OC), kernel);
        });
    }
    return queue.ext_oneapi_submit_barrier({ev_dinp, ev_dweight, ev_dbias});
}

template <int SG_SIZE>
class ker_matmul_backward2_dinp;
template <int SG_SIZE>
class ker_matmul_backward2_dweight;
template <int SG_SIZE>
class ker_matmul_backward2_dbias;

template<int SG_SIZE>
sycl::event matmul_backward2(sycl::queue &queue, float* dinp, float* dweight, float* dbias,
                             const float* dout, const float* inp, const float* weight,
                             int B, int T, int C, int OC, int sg_per_wg,
                             const std::vector<sycl::event> &dependencies = {})
{
    // Round up next multiple, sg_per_wg always assumed to be a power of 2
    const int ceilBT = (B*T + sg_per_wg - 1) & (-sg_per_wg);
    const int ceilC  = (C + SG_SIZE - 1) & (-SG_SIZE);
    sycl::event ev_dinp = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
            const size_t bt = item.get_global_id(0);
            const size_t c  = item.get_global_id(1);
            if (c >= C || bt >= B*T) return;
            float val = float(0);
            for (int o = 0; o < OC; ++o) {
                val += weight[o * C + c] * dout[bt * OC + o];
            }
            dinp[bt * C + c] += val;
        };
        cgh.parallel_for<class ker_matmul_backward2_dinp<SG_SIZE>>(
                sycl::nd_range<2>(sycl::range<2>(ceilBT, ceilC),
                                  sycl::range<2>(sg_per_wg, SG_SIZE)), kernel);
    });

    // backward into weight, parallelize over output channels OC, C
    const int ceilOC = (OC + sg_per_wg - 1) & (-sg_per_wg);
    sycl::event ev_dweight = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
            const size_t o = item.get_global_id(0);
            const size_t c = item.get_global_id(1);
            if (o >= OC || c >= C) return;
            float val = float(0);
            for (int bt = 0; bt < B*T; bt++) {
                val += inp[bt * C + c] * dout[bt * OC + o];
            }
            dweight[o * C + c] += val;
        };
        cgh.parallel_for<class ker_matmul_backward2_dweight<SG_SIZE>>(
                sycl::nd_range<2>(sycl::range<2>(ceilOC, ceilC),
                                  sycl::range<2>(sg_per_wg, SG_SIZE)), kernel);
    });

    // backward into bias, parallelize over output channels OC
    sycl::event ev_dbias = sycl::event();
    if (dbias != NULL) {
        ev_dbias = queue.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dependencies);
            auto kernel = [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                const size_t o            = item.get_global_id(0);
                const sycl::sub_group sgr = item.get_sub_group();
                const size_t sgr_cid      = sgr.get_local_id();
                if (o >= OC) return;
                float sum = float(0);
                for (int bt = sgr_cid; bt < B*T; bt += SG_SIZE) {
                    sum += dout[bt * OC + o];
                }
                // Reduce values in each sub_group to sgr_cid == 0
                sum = sycl::reduce_over_group(sgr, sum, sycl::plus<float>());
                // For unknown reason, this reduction is failing on CPU device
                //for (std::uint8_t j = SG_SIZE >> 1; j > 0; j >>= 1) {
                //    sum += sycl::shift_group_left(sgr, sum, j);
                //}
                if (sgr_cid == 0) dbias[o] += sum;
            };
            cgh.parallel_for<class ker_matmul_backward2_dbias<SG_SIZE>>(
                    sycl::nd_range<2>(sycl::range<2>(ceilOC, SG_SIZE),
                                      sycl::range<2>(sg_per_wg, SG_SIZE)), kernel);
        });
    }

    return queue.ext_oneapi_submit_barrier({ev_dinp, ev_dweight, ev_dbias});
}

// kernel version dispatch
sycl::event matmul_backward(int kernel_num,
                            sycl::queue &queue, float* dinp, float* dweight, float* dbias,
                            const float* dout, const float* inp, const float* weight,
                            int B, int T, int C, int OC,
                            const int sg_per_wg, const int sg_size) {
    switch (kernel_num) {
        case 1: {
            return matmul_backward1(queue, dinp, dweight, dbias, dout, inp, weight, B, T, C, OC);
        } break;
        case 2: {
            if (sg_size == 32) {
                return matmul_backward2<32>(queue, dinp, dweight, dbias, dout, inp, weight, B, T, C, OC, sg_per_wg);
            }
#ifndef _DISABLE_SG_SIZE_16
            else if (sg_size == 16) {
                return matmul_backward2<16>(queue, dinp, dweight, dbias, dout, inp, weight, B, T, C, OC, sg_per_wg);
            }
#endif
#ifndef _DISABLE_SG_SIZE_8
            else if (sg_size == 8) {
                return matmul_backward2< 8>(queue, dinp, dweight, dbias, dout, inp, weight, B, T, C, OC, sg_per_wg);
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
    float* dinp    = (float*)hostMallocCheck(B * T * C * sizeof(float), queue);
    float* dweight = (float*)hostMallocCheck(OC * C * sizeof(float), queue);
    float* dbias   = (float*)hostMallocCheck(OC * sizeof(float), queue);
    float* dout    = (float*)hostMallocCheck(B * T * OC * sizeof(float), queue);
    float* inp     = (float*)hostMallocCheck(B * T * C * sizeof(float), queue);
    float* weight  = (float*)hostMallocCheck(OC * C * sizeof(float), queue);

    queue.fill<float>(dinp, float(0), B * T * C);
    queue.fill<float>(dweight, float(0), OC * C);
    queue.fill<float>(dbias, float(0), OC);
    make_random_float(dout, B * T * OC);
    make_random_float(inp, B * T * C);
    make_random_float(weight, OC * C);

    // move to GPU
    float* d_dinp    = (float*)deviceMallocCheck(B * T * C * sizeof(float), queue);
    float* d_dweight = (float*)deviceMallocCheck(C * OC * sizeof(float), queue);
    float* d_dbias   = (float*)deviceMallocCheck(OC * sizeof(float), queue);
    float* d_dout    = (float*)deviceMallocCheck(B * T * OC * sizeof(float), queue);
    float* d_inp     = (float*)deviceMallocCheck(B * T * C * sizeof(float), queue);
    float* d_weight  = (float*)deviceMallocCheck(C * OC * sizeof(float), queue);
    
    queue.fill<float>(d_dinp, float(0), B * T * C);
    queue.fill<float>(d_dweight, float(0), OC * C);
    queue.fill<float>(d_dbias, float(0), OC);
    queue.memcpy(d_dout, dout, B * T * OC * sizeof(float));
    queue.memcpy(d_inp, inp, B * T * C * sizeof(float));
    queue.memcpy(d_weight, weight, OC * C * sizeof(float));
    queue.wait();

    // read kernel_num from command line
    int kernel_num = 1;
    bool run_all = true;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
        printf("Using kernel %d\n", kernel_num);
        run_all = false;
    }
    const int repeat_times = 5;

    // first check the correctness of the kernel
    printf("Running reference CPU kernel.\n");
    matmul_backward_cpu(dinp, dweight, dbias, dout, inp, weight, B, T, C, OC);

    // Test kernel 1
    if (kernel_num == 1) {
        printf("************************************************.\n");
        printf("Checking kernel set #%i.\n", kernel_num);
        matmul_backward(kernel_num, queue, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, B, T, C, OC, 0, 0);
        queue.wait();
        validate_result(queue, d_dinp, dinp, "dinp", B * T * C, 1e-2f);
        validate_result(queue, d_dweight, dweight, "dweight", OC * C, 1e-2f);
        validate_result(queue, d_dbias, dbias, "dbias", OC, 1e-2f);
        printf("All results match. Benchmarking kernel %i.\n", kernel_num);
        double elapsed_ms = benchmark_kernel(repeat_times, matmul_backward, kernel_num, queue,
                                             d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight,
                                             B, T, C, OC, 0, 0);
        double tflops = get_matmul_bwd_tflops(elapsed_ms, B, T, C, OC);
        printf("kernel %2i | time %.4f ms | tflops %.2f\n", kernel_num, elapsed_ms, tflops);

        queue.fill<float>(d_dinp, float(0), B * T * C);
        queue.fill<float>(d_dweight, float(0), OC * C);
        queue.fill<float>(d_dbias, float(0), OC);
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
            for (int sg_per_wg = 1; sg_per_wg <= max_workgroup_size/sg_size && sg_per_wg <= 64; sg_per_wg <<= 1) {
                printf("Checking kernel %i<%i, %i>.\n", kernel_num, sg_per_wg, sg_size);
                matmul_backward(kernel_num, queue, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, B, T, C, OC, sg_per_wg, sg_size);
                queue.wait();
                validate_result(queue, d_dinp, dinp, "dinp", B * T * C, 1e-2f);
                validate_result(queue, d_dweight, dweight, "dweight", OC * C, 1e-2f);
                validate_result(queue, d_dbias, dbias, "dbias", OC, 1e-2f);
                printf("All results match. Benchmarking kernel %i<%i, %i>.\n", kernel_num, sg_per_wg, sg_size);
                double elapsed_ms = benchmark_kernel(repeat_times, matmul_backward, kernel_num, queue,
                                                     d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight,
                                                     B, T, C, OC, sg_per_wg, sg_size);
                double tflops = get_matmul_bwd_tflops(elapsed_ms, B, T, C, OC);
                printf("kernel %2i<%2i, %2i> | time %.4f ms | tflops %.2f\n", kernel_num, sg_per_wg, sg_size, elapsed_ms, tflops);

                queue.fill<float>(d_dinp, float(0), B * T * C);
                queue.fill<float>(d_dweight, float(0), OC * C);
                queue.fill<float>(d_dbias, float(0), OC);
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
    sycl::free(dweight, queue);
    sycl::free(dbias, queue);
    sycl::free(dout, queue);
    sycl::free(inp, queue);
    sycl::free(weight, queue);
    sycl::free(d_dinp, queue);
    sycl::free(d_dweight, queue);
    sycl::free(d_dbias, queue);
    sycl::free(d_dout, queue);
    sycl::free(d_inp, queue);
    sycl::free(d_weight, queue);
    return 0;
}
