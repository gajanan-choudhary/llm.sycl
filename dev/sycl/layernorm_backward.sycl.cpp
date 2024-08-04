/*
Kernels for layernorm backward pass.

GPU Compile example for Intel PVC GPU with ahead-of-time compilation:
icpx -O3 -fsycl -fno-sycl-id-queries-fit-in-int -std=c++17 -xhost -DLLMSYCL -fp-model=precise \
     -qopenmp -qopenmp-simd -DOMP  -DTIMEPROFILE \
     -fsycl-targets=spir64_gen -Xs "-device pvc" -D_DISABLE_SG_SIZE_8 \
     layernorm_backward.sycl.cpp -lsycl -lOpenCL -liomp5 -o layernorm_backward

version 1 is sycl::range<> port from CPU code to kernel: parallelizes over B*T, C, and loops over C
./layernorm_backward 1

version 2 is sycl::nd_range<> port of version 1 with different SG_SIZE, sg_per_wg
./layernorm_backward 2

TODO: Try oneDNN / oneMKL for version 3

Observations as of 08/04/2024: on CPU device, version 1 is 2x faster than version 2, all of which has roughly
                               same timing on it, with {SG_PER_WG, SG_SIZE}={{4, 8}, {2, 16}} the fastest version 2
                               version 2 with {SG_PER_WG, SG_SIZE}={8, 32} are fastest on GPU device

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
double get_layernorm_bwd_tflops(double elapsed_ms, size_t B, size_t T, size_t C) {
    // Time is in milliseconds
    return (double) (B * T * (C * 18 + 2)) / elapsed_ms * 1e-9;
}

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 layernorm forward pass
void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                           const float* inp, const float* weight, const float* bias,
                           int B, int T, int C) {
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
}

// GPT-2 layernorm backward pass
void layernorm_backward_cpu(float* dinp, float* dweight, float* dbias,
                            const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                            int B, int T, int C) {
    timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* dout_bt = dout + b * T * C + t * C;
            const float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            const float mean_bt = mean[b * T + t];
            const float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = get_elapsed_ms(start, end);
    double tflops = get_layernorm_bwd_tflops(elapsed_ms, B, T, C);
    printf("kernel ref SEQ | time %.4f ms | tflops %.2f\n", elapsed_ms, tflops);
}

// ----------------------------------------------------------------------------
// GPU kernels

sycl::event layernorm_backward1(sycl::queue &queue, float* dinp, float* dweight, float* dbias,
                                const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                                const int B, const int T, const int C,
                                const std::vector<sycl::event> &dependencies = {}) {
    sycl::event ev_dbias_dweight = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<1> item) {
            const size_t c = item.get_id(0);
            const float *inp_c  = inp + c;
            const float *dout_c = dout + c;

            float dbiasval = float(0);
            float dweightval = float(0);
            for (int bt = 0; bt < B*T; bt++) {
                const float d        = dout_c[bt * C];
                const float norm_bti = (inp_c[bt * C] - mean[bt]) * rstd[bt];
                // gather'd reduction
                dbiasval   += d;
                dweightval += norm_bti * d;
            }
            // gradient contribution to bias
            dbias[c] += dbiasval;
            // gradient contribution to weight
            dweight[c] += dweightval;
        };
        cgh.parallel_for<class ker_layernorm_backward_1_dbias_dweight>(sycl::range<1>(C), kernel);
    });

    sycl::event ev_dinp = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<1> item) {
            const size_t bt = item.get_id(0);
            const size_t offset = bt * C;

            const float* dout_bt = dout + offset;
            const float* inp_bt  = inp  + offset;
            float* dinp_bt       = dinp + offset;

            const float mean_bt = mean[bt];
            const float rstd_bt = rstd[bt];

            // first: two reduce operations
            float dnorm_mean = float(0);
            float dnorm_norm_mean = float(0);
            for (int i = 0; i < C; i++) {
                const float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                const float dnorm_i  = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean      = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate the remaining gradient
            for (int i = 0; i < C; i++) {
                const float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                const float dnorm_i  = weight[i] * dout_bt[i];
                // gradient contribution to input
                float dval = float(0);
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        };
        cgh.parallel_for<class ker_layernorm_backward_1_dinp>(sycl::range<1>(B*T), kernel);
    });
    return queue.ext_oneapi_submit_barrier({ev_dbias_dweight, ev_dinp});
}

template <int SG_SIZE>
class ker_layernorm_backward_2_dbias_dweight;
template <int SG_SIZE>
class ker_layernorm_backward_2_dinp;

template<int SG_SIZE>
sycl::event layernorm_backward2(sycl::queue &queue, float* dinp, float* dweight, float* dbias,
                                const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                                const int B, const int T, const int C, const int sg_per_wg,
                                const std::vector<sycl::event> &dependencies = {}) {
    const int wg_size = sg_per_wg * SG_SIZE;
    sycl::event ev_dbias_dweight = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
            const size_t c            = item.get_global_id(0);
            const sycl::group gr      = item.get_group();
            const sycl::sub_group sgr = item.get_sub_group();
            const size_t lid          = item.get_local_linear_id();

            const float *inp_c  = inp + c;
            const float *dout_c = dout + c;

            float dbiasval = float(0);
            float dweightval = float(0);
            for (int bt = lid; bt < B*T; bt += wg_size) {
                const float d        = dout_c[bt * C];
                const float norm_bti = (inp_c[bt * C] - mean[bt]) * rstd[bt];
                // gather'd reduction
                dbiasval   += d;
                dweightval += norm_bti * d;
            }
            dbiasval   = sycl::reduce_over_group(gr, dbiasval, sycl::plus<float>());
            dweightval = sycl::reduce_over_group(gr, dweightval, sycl::plus<float>());
            // gradient contribution to bias
            if (lid == 0) dbias[c] += dbiasval;
            // gradient contribution to weight
            if (lid == 0) dweight[c] += dweightval;
        };
        cgh.parallel_for<class ker_layernorm_backward_2_dbias_dweight<SG_SIZE>>(
                sycl::nd_range<2>(sycl::range<2>(C, wg_size),
                                  sycl::range<2>(1, wg_size)), kernel);
    });

    sycl::event ev_dinp = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
            const size_t bt           = item.get_global_id(0);
            const sycl::group gr      = item.get_group();
            const sycl::sub_group sgr = item.get_sub_group();
            const size_t lid          = item.get_local_linear_id();

            const float* dout_bt = dout + bt * C;
            const float* inp_bt = inp + bt * C;
            float* dinp_bt = dinp + bt * C;
            const float mean_bt = mean[bt];
            const float rstd_bt = rstd[bt];

            // first: two reduce operations
            float dnorm_mean = float(0);
            float dnorm_norm_mean = float(0);
            for (int i = lid; i < C; i += wg_size) {
                const float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                const float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = sycl::reduce_over_group(gr, dnorm_mean, sycl::plus<float>());
            dnorm_norm_mean = sycl::reduce_over_group(gr, dnorm_norm_mean, sycl::plus<float>());
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = lid; i < C; i += wg_size) {
                const float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                const float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to input
                float dval = float(0);
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        };
        cgh.parallel_for<class ker_layernorm_backward_2_dinp<SG_SIZE>>(
                sycl::nd_range<2>(sycl::range<2>(B*T, wg_size),
                                  sycl::range<2>(1, wg_size)), kernel);
    });
    return queue.ext_oneapi_submit_barrier({ev_dbias_dweight, ev_dinp});
}

// kernel version dispatch
sycl::event layernorm_backward(int kernel_num,
                               sycl::queue &queue, float* dinp, float* dweight, float* dbias,
                               const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                               const int B, const int T, const int C, const int sg_per_wg, const int sg_size) {
    switch (kernel_num) {
        case 1: {
            return layernorm_backward1(queue, dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
        } break;
        case 2: {
            if (sg_size == 32) {
                return layernorm_backward2<32>(queue, dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, sg_per_wg);
            }
#ifndef _DISABLE_SG_SIZE_16
            else if (sg_size == 16) {
                return layernorm_backward2<16>(queue, dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, sg_per_wg);
            }
#endif
#ifndef _DISABLE_SG_SIZE_8
            else if (sg_size == 8) {
                return layernorm_backward2< 8>(queue, dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, sg_per_wg);
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
    float* out     = (float*)hostMallocCheck(B * T * C * sizeof(float), queue);
    float* mean    = (float*)hostMallocCheck(B * T * sizeof(float), queue);
    float* rstd    = (float*)hostMallocCheck(B * T * sizeof(float), queue);
    float* inp     = (float*)hostMallocCheck(B * T * C * sizeof(float), queue);
    float* weight  = (float*)hostMallocCheck(C * sizeof(float), queue);
    float* bias    = (float*)hostMallocCheck(C * sizeof(float), queue);
    float* dout    = (float*)hostMallocCheck(B * T * C * sizeof(float), queue);
    float* dinp    = (float*)hostMallocCheck(B * T * C * sizeof(float), queue);
    float* dweight = (float*)hostMallocCheck(C * sizeof(float), queue);
    float* dbias   = (float*)hostMallocCheck(C * sizeof(float), queue);

    queue.fill<float>(out, float(0), B * T * C);
    queue.fill<float>(mean, float(0), B * T);
    queue.fill<float>(rstd, float(0), B * T);
    make_random_float(inp, B * T * C);
    make_random_float(weight, C);
    make_random_float(bias, C);
    make_random_float(dout, B * T * C);
    queue.fill<float>(dinp, float(0), B * T * C);
    queue.fill<float>(dweight, float(0), C);
    queue.fill<float>(dbias, float(0), C);
    queue.wait();

    printf("Running one layernorm_forward call to get inputs {inp, weight, bias}.\n");
    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);

    // move to GPU
    float* d_out     = (float*)deviceMallocCheck(B * T * C * sizeof(float), queue);
    float* d_mean    = (float*)deviceMallocCheck(B * T * sizeof(float), queue);
    float* d_rstd    = (float*)deviceMallocCheck(B * T * sizeof(float), queue);
    float* d_inp     = (float*)deviceMallocCheck(B * T * C * sizeof(float), queue);
    float* d_weight  = (float*)deviceMallocCheck(C * sizeof(float), queue);
    float* d_bias    = (float*)deviceMallocCheck(C * sizeof(float), queue);
    float* d_dout    = (float*)deviceMallocCheck(B * T * C * sizeof(float), queue);
    float* d_dinp    = (float*)deviceMallocCheck(B * T * C * sizeof(float), queue);
    float* d_dweight = (float*)deviceMallocCheck(C * sizeof(float), queue);
    float* d_dbias   = (float*)deviceMallocCheck(C * sizeof(float), queue);
    
    queue.memcpy(d_out, out, B * T * C * sizeof(float)); // Copying layernorm_forward calculated vals
    queue.memcpy(d_mean, mean, B * T * sizeof(float));   // Copying layernorm_forward calculated vals
    queue.memcpy(d_rstd, rstd, B * T * sizeof(float));   // Copying layernorm_forward calculated vals
    queue.memcpy(d_inp, inp, B * T * C * sizeof(float));
    queue.memcpy(d_weight, weight, C * sizeof(float));
    queue.memcpy(d_bias, bias, C * sizeof(float));
    queue.memcpy(d_dout, dout, B * T * C * sizeof(float));
    queue.fill<float>(d_dinp, float(0), B * T * C);
    queue.fill<float>(d_dweight, float(0), C);
    queue.fill<float>(d_dbias, float(0), C);
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
    layernorm_backward_cpu(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);

    // Test kernel 1
    if (kernel_num == 1) {
        printf("************************************************.\n");
        printf("Checking kernel set #%i.\n", kernel_num);
        layernorm_backward(kernel_num, queue, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_mean, d_rstd, B, T, C, 0, 0);
        queue.wait();
        validate_result(queue, d_dinp, dinp, "dinp", B * T * C, 1e-2f);
        validate_result(queue, d_dweight, dweight, "dweight", C, 1e-2f);
        validate_result(queue, d_dbias, dbias, "dbias", C, 1e-2f);
        printf("All results match. Benchmarking kernel %i.\n", kernel_num);
        double elapsed_ms = benchmark_kernel(repeat_times, layernorm_backward,
                                             kernel_num, queue, d_dinp, d_dweight, d_dbias,
                                             d_dout, d_inp, d_weight, d_mean, d_rstd,
                                             B, T, C, 0, 0);
        double tflops = get_layernorm_bwd_tflops(elapsed_ms, B, T, C);
        printf("kernel %2i | time %.4f ms | tflops %.2f\n", kernel_num, elapsed_ms, tflops);

        queue.fill<float>(d_dinp, float(0), B * T * C);
        queue.fill<float>(d_dweight, float(0), C);
        queue.fill<float>(d_dbias, float(0), C);
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
                layernorm_backward(kernel_num, queue, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_mean, d_rstd, B, T, C, sg_per_wg, sg_size);
                queue.wait();
                validate_result(queue, d_dinp, dinp, "dinp", B * T * C, 1e-2f);
                validate_result(queue, d_dweight, dweight, "dweight", C, 1e-2f);
                validate_result(queue, d_dbias, dbias, "dbias", C, 1e-2f);
                printf("All results match. Benchmarking kernel %i<%i, %i>.\n", kernel_num, sg_per_wg, sg_size);
                double elapsed_ms = benchmark_kernel(repeat_times, layernorm_backward,
                                                     kernel_num, queue, d_dinp, d_dweight, d_dbias,
                                                     d_dout, d_inp, d_weight, d_mean, d_rstd,
                                                     B, T, C, sg_per_wg, sg_size);
                double tflops = get_layernorm_bwd_tflops(elapsed_ms, B, T, C);
                printf("kernel %2i<%2i, %2i> | time %.4f ms | tflops %.2f\n", kernel_num, sg_per_wg, sg_size, elapsed_ms, tflops);

                queue.fill<float>(d_dinp, float(0), B * T * C);
                queue.fill<float>(d_dweight, float(0), C);
                queue.fill<float>(d_dbias, float(0), C);
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
    sycl::free(dout, queue);
    sycl::free(dinp, queue);
    sycl::free(dweight, queue);
    sycl::free(dbias, queue);
    sycl::free(d_out, queue);
    sycl::free(d_mean, queue);
    sycl::free(d_rstd, queue);
    sycl::free(d_inp, queue);
    sycl::free(d_weight, queue);
    sycl::free(d_bias, queue);
    sycl::free(d_dout, queue);
    sycl::free(d_dinp, queue);
    sycl::free(d_dweight, queue);
    sycl::free(d_dbias, queue);
    return 0;
}
