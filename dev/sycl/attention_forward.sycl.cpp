/*
Kernels for attention forward pass.

GPU Compile example for Intel PVC GPU with ahead-of-time compilation:
icpx -O3 -fsycl -fno-sycl-id-queries-fit-in-int -std=c++17 -xhost -DLLMSYCL -fp-model=precise \
     -qopenmp -qopenmp-simd -DOMP  -DTIMEPROFILE \
     -fsycl-targets=spir64_gen -Xs "-device pvc" -D_DISABLE_SG_SIZE_8 \
     attention_forward.sycl.cpp -lsycl -lOpenCL -liomp5 -o attention_forward

version 1 is sycl::range<> port from CPU code to kernel: parallelizes over B*T, C, and loops over C
./attention_forward 1

version 2 is sycl::nd_range<> port of version 1 with different SG_SIZE, sg_per_wg
./attention_forward 2

TODO: Try oneDNN / oneMKL for version 3

Observations as of 08/05/2024: version 2 with SG_SIZE={8, 16}, SG_PER_WG = {1, 2} are faster on CPU device @ 215ms
                               version 2 with {SG_SIZE, SG_PER_WG} = {2, 32} is faster on GPU device @ 128ms
                               but caution that this is because C / NH = 64, and the parallelization
                               strategy for v2 kernel is over that, so extra workgroups don't help.

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

#define SQRT sqrtf
#define EXP  expf

// ----------------------------------------------------------------------------
// TFLOP/s
double get_attention_fwd_tflops(double elapsed_ms, size_t B, size_t T, size_t C, size_t NH) {
    // Time is in milliseconds
    size_t hs = C/NH;
    // Counting is based on reference kernel
    return (double) (B * NH * (T * (T+1) * (2*hs + 6))) / elapsed_ms * 1e-9;
}

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 attention forward pass
void attention_forward_cpu(float* out, float* preatt, float* att,
                           const float* inp,
                           const int B, const int T, const int C, const int NH) {
    timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // input is (B, T, 3C) Q,K,V
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -FLT_MAX;
                // (T*(T+1)/2) * (2*hs + 1) flops
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    // 2*hs flops
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    // 1 flop
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }
                // pad with -INFINITY outside of autoregressive region for debugging comparisons
                for (int t2 = t+1; t2 < T; t2++) {
                    preatt_bth[t2] = -INFINITY;
                }

                // pass 2: calculate the exp and keep track of sum
                float expsum = 0.0f;
                // (T*(T+1)/2) * 10 flops
                for (int t2 = 0; t2 <= t; t2++) {
                    // 10 flops (assuming EXP = 8 flops)
                    float expv = EXP(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                // 1 flop
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                // (T*(T+1)/2) * 1 flops
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        // 1 flop
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                // (T*(T+1)/2) * 2 * hs flops
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    // 2 * hs flops
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = get_elapsed_ms(start, end);
    double tflops = get_attention_fwd_tflops(elapsed_ms, B, T, C, NH);
    printf("kernel ref OMP | time %.4f ms | tflops %.2f\n", elapsed_ms, tflops);
}

// ----------------------------------------------------------------------------
// GPU kernels

// kernel 1 is the most naive attention kernel
sycl::event attention_forward1(sycl::queue &queue, float* out, float* preatt, float* att,
                               const float* inp,
                               const int B, const int T, const int C, const int NH,
                               const std::vector<sycl::event> &dependencies = {}) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    const int C3 = C*3;
    const int hs = C / NH; // head size
    const float scale = 1.0 / sqrtf(float(hs));

    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<3> item) {
            const size_t b  = item.get_id(0);
            const size_t t  = item.get_id(1);
            const size_t h  = item.get_id(2);
            const size_t bt = b*T+t;
            const float* query_t = inp + bt * C3 + h * hs;
            float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
            float* att_bth = att + b*NH*T*T + h*T*T + t*T;

            // pass 1: calculate query dot key and maxval
            float maxval = -FLT_MAX;
            for (int t2 = 0; t2 <= t; t2++) {
                const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                // (query_t) dot (key_t2)
                float val = float(0);
                for (int i = 0; i < hs; i++) {
                    val += query_t[i] * key_t2[i];
                }
                val *= scale;
                if (val > maxval) {
                    maxval = val;
                }

                preatt_bth[t2] = val;
            }

            // pass 2: calculate the exp and keep track of sum
            // maxval is being calculated and subtracted only for numerical stability
            float expsum = float(0);
            for (int t2 = 0; t2 <= t; t2++) {
                float expv = EXP(preatt_bth[t2] - maxval);
                expsum += expv;
                att_bth[t2] = expv;
            }
            float expsum_inv = expsum == float(0) ? float(0) : float(1.0) / expsum;

            // pass 3: normalize to get the softmax
            for (int t2 = 0; t2 < T; t2++) {
                if (t2 <= t) {
                    att_bth[t2] *= expsum_inv;
                } else {
                    // causal attention mask. not strictly necessary to set to zero here
                    // only doing this explicitly for debugging and checking to PyTorch
                    att_bth[t2] = float(0);
                }
            }

            // pass 4: accumulate weighted values into the output of attention
            float* out_bth = out + bt * C + h * hs;
            for (int i = 0; i < hs; i++) { out_bth[i] = float(0); }
            for (int t2 = 0; t2 <= t; t2++) {
                const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                float att_btht2 = att_bth[t2];
                for (int i = 0; i < hs; i++) {
                    out_bth[i] += att_btht2 * value_t2[i];
                }
            }
        };
        cgh.parallel_for<class ker_attention_forward_1>(sycl::range<3>(B, T, NH), kernel);
    });
    return last;
}

template <int SG_SIZE>
class ker_attention_forward_2;

template<int SG_SIZE>
sycl::event attention_forward2(sycl::queue &queue, float* out, float* preatt, float* att,
                               const float* inp,
                               const int B, const int T, const int C, const int NH, const int sg_per_wg,
                               const std::vector<sycl::event> &dependencies = {}) {
    assert(C % NH == 0);
    const int C3 = C*3;
    const int hs = C / NH; // head size
    const float scale = 1.0 / sqrtf(float(hs));
    // Here we do something slightly different from usual, viz.,
    // we disallow workgroup size to exceed `hs` by too much.
    // It may exceed
    //const int wg_size = sg_per_wg * SG_SIZE;
    const int adjusted_sg_per_wg = std::min((hs + SG_SIZE - 1) / SG_SIZE, sg_per_wg);
    const int wg_size = adjusted_sg_per_wg * SG_SIZE;

    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
            const size_t bt  = item.get_global_id(0);
            const size_t b   = bt / T;
            const size_t t   = bt % T;
            const size_t h   = item.get_global_id(1);
            sycl::group gr   = item.get_group();
            const size_t lid = item.get_local_linear_id(); // sg_gid * SG_SIZE + sg_cid

            const size_t ht      = h * T + t;
            const float* query_t = inp + bt * C3 + h * hs;
            float* preatt_bth    = preatt + b * NH * T * T + ht * T;
            float* att_bth       = att + b * NH * T * T + ht * T;

            // pass 1: calculate query dot key and maxval
            float maxval = -FLT_MAX;
            const float* key_t2_base = inp + b * T * C3 + h * hs + C; // +C because it's key
            for (int t2 = 0; t2 <= t; t2++) {
                const float* key_t2 = key_t2_base + t2 * C3; // +C because it's key

                // (query_t) dot (key_t2)
                float val = float(0);
                for (int i = lid; i < hs; i += wg_size) {
                    val += query_t[i] * key_t2[i];
                }
                val = sycl::reduce_over_group(gr, val, sycl::plus<float>());
                val *= scale;
                if (val > maxval) {
                    maxval = val;
                }

                if (lid == 0) preatt_bth[t2] = val;
            }
            sycl::group_barrier(gr);

            // pass 2: calculate the exp and keep track of sum
            // maxval is being calculated and subtracted only for numerical stability
            float expsum = float(0);
            for (int t2 = lid; t2 <= t; t2 += wg_size) {
                float expv = EXP(preatt_bth[t2] - maxval);
                expsum += expv;
                att_bth[t2] = expv;
            }
            expsum = sycl::reduce_over_group(gr, expsum, sycl::plus<float>());
            float expsum_inv = expsum == float(0) ? float(0) : float(1.0) / expsum;

            // pass 3: normalize to get the softmax
            for (int t2 = lid; t2 < T; t2 += wg_size) {
                if (t2 <= t) {
                    att_bth[t2] *= expsum_inv;
                } else {
                    // causal attention mask. not strictly necessary to set to zero here
                    // only doing this explicitly for debugging and checking to PyTorch
                    att_bth[t2] = float(0);
                }
            }

            // pass 4: accumulate weighted values into the output of attention
            float* out_bth = out + bt * C + h * hs;
            for (int i = lid; i < hs; i += wg_size) { out_bth[i] = float(0); }
            sycl::group_barrier(gr);
            for (int t2 = 0; t2 <= t; t2++) {
                const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                float att_btht2 = att_bth[t2];
                for (int i = lid; i < hs; i += wg_size) {
                    out_bth[i] += att_btht2 * value_t2[i];
                }
            }
        };
        cgh.parallel_for<class ker_attention_forward_2<SG_SIZE>>(
                sycl::nd_range<3>(sycl::range<3>(B * T, NH, wg_size),
                                  sycl::range<3>(1, 1, wg_size)), kernel);
    });
    return last;
}

#ifdef ONEMKL_INTERFACES
sycl::event attention_forward_onemkl_interfaces(sycl::queue &queue, float* out, float* preatt, float* att,
                                                const float* inp,
                                                const int B, const int T, const int C, const int NH,
                                                const std::vector<sycl::event> &dependencies = {}) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    const int C3 = C*3;
    const int hs = C / NH; // head size
    const float scale = 1.0 / sqrtf(float(hs));

    std::vector<sycl::event> depends_qdotk(B);
    for (int b = 0; b < B; b++) {
        // Many triangular Q x K^T operations are performed.
        // Sizes: (T, hs) x (hs, T) = (T, T) matrices. There are B * NH such
        // matrices generated.
        // Upper triangular region of preatt MUST be separately reset to
        // -INFINITY later for this to work as intended later.
        const float *query = inp + b * T * C3;
        const int ldq = C3;
        const float *key = query + C;
        const int ldk = C3;
        float* preatt_bh = preatt + b*NH*T*T;
        const int ldp = T;
        depends_qdotk[b] = oneapi::mkl::blas::column_major::gemm_batch(queue,
                oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                T, T, hs, scale, key, ldk, hs, query, ldq, hs, 0.0, preatt_bh, ldp, T*T, NH, dependencies);
        // row_major gemm_batch apparently not supported in cuBLAS
        //depends_qdotk[b] = oneapi::mkl::blas::row_major::gemm_batch(queue,
        //        oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,
        //        T, T, hs, scale, query, ldq, hs, key, ldk, hs, 0.0, preatt_bh, ldp, T*T, NH, dependencies);
    }

    sycl::event ev_softmax = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends_qdotk);
        auto kernel = [=](sycl::item<3> item) {
            const size_t b  = item.get_id(0);
            const size_t t  = item.get_id(1);
            const size_t h  = item.get_id(2);

            float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
            float* att_bth    = att    + b*NH*T*T + h*T*T + t*T;

            // pass 1: calculate query dot key and maxval
            float maxval = -FLT_MAX;
            for (int t2 = 0; t2 <= t; t2++) {
                const float val = preatt_bth[t2];
                if (val > maxval) {
                    maxval = val;
                }
            }
            // pad preatt with -INFINITY and att with zero outside of autoregressive
            // region for keeping BLAS batched GEMM calls valid
            for (int t2 = t+1; t2 < T; t2++) {
                preatt_bth[t2] = -INFINITY;
            }

            // pass 2: calculate the exp and keep track of sum
            // maxval is being calculated and subtracted only for numerical stability
            float expsum = float(0);
            for (int t2 = 0; t2 <= t; t2++) {
                float expv = EXP(preatt_bth[t2] - maxval);
                expsum += expv;
                att_bth[t2] = expv;
            }
            float expsum_inv = expsum == float(0) ? float(0) : float(1.0) / expsum;

            // pass 3: normalize to get the softmax
            // pad att with zero outside of autoregressive
            // region for keeping BLAS batched GEMM calls valid
            for (int t2 = 0; t2 < T; t2++) {
                att_bth[t2] = (t2 <= t ? att_bth[t2] * expsum_inv : float(0));
            }
        };
        cgh.parallel_for<class ker_attention_forward_onemkl_interfaces_softmax_out>(sycl::range<3>(B, T, NH), kernel);
    });

    std::vector<sycl::event> depends_av(NH);
    for (int h = 0; h < NH; h++) {
        // Many triangular att x V operations are performed.
        // Sizes: (T, T) x (T, hs) = (T, hs) matrices. There are B * NH such
        // matrices generated.
        // Upper triangular region of att_bh MUST be preset to 0 earlier for
        // this to work as intended!
        const float* att_bh = att + h * T * T;
        const int lda = T;
        const float* value = inp + h * hs + C*2; // +C*2 because it's value
        const int ldv = C3;
        float* out_bh = out + h * hs;
        const int ldo = C;
        depends_av[h] = oneapi::mkl::blas::column_major::gemm_batch(queue,
                oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
                hs, T, T, 1.0, value, ldv, T * C3, att_bh, lda, NH*T*T, 0.0, out_bh, ldo, T * C, B, {ev_softmax});
        // row_major gemm_batch apparently not supported in cuBLAS
        //depends_av[h] = oneapi::mkl::blas::row_major::gemm_batch(queue,
        //        oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
        //        T, hs, T, 1.0, att_bh, lda, NH*T*T, value, ldv, T * C3, 0.0, out_bh, ldo, T * C, B, {ev_softmax});
    }
    return queue.ext_oneapi_submit_barrier(depends_av);
}
#endif

// kernel version dispatch
sycl::event attention_forward(int kernel_num,
                              sycl::queue &queue, float* out, float* preatt, float* att,
                              const float* inp,
                              const int B, const int T, const int C, const int NH,
                              const int sg_per_wg, const int sg_size) {
    switch (kernel_num) {
        case 1: {
            return attention_forward1(queue, out, preatt, att, inp, B, T, C, NH);
        } break;
        case 2: {
            if (sg_size == 32) {
                return attention_forward2<32>(queue, out, preatt, att, inp, B, T, C, NH, sg_per_wg);
            }
#ifndef _DISABLE_SG_SIZE_16
            else if (sg_size == 16) {
                return attention_forward2<16>(queue, out, preatt, att, inp, B, T, C, NH, sg_per_wg);
            }
#endif
#ifndef _DISABLE_SG_SIZE_8
            else if (sg_size == 8) {
                return attention_forward2< 8>(queue, out, preatt, att, inp, B, T, C, NH, sg_per_wg);
            }
#endif
            else {
                printf("Invalid sg_size\n");
                exit(2);
            }
        } break;
#ifdef ONEMKL_INTERFACES
        case 3: {
            return attention_forward_onemkl_interfaces(queue, out, preatt, att, inp, B, T, C, NH);
        } break;
#endif
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
    int NH = 12;

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
    float* preatt = (float*)hostMallocCheck(B * NH * T * T * sizeof(float), queue);
    float* att    = (float*)hostMallocCheck(B * NH * T * T * sizeof(float), queue);
    float* inp    = (float*)hostMallocCheck(B * T * 3 * C * sizeof(float), queue);

    queue.fill<float>(out, float(0), B * T * C);
    queue.fill<float>(preatt, float(0), B * NH * T * T);
    queue.fill<float>(att, float(0), B * NH * T * T);
    make_random_float(inp, B * T * 3 * C);

    // move to GPU
    float* d_out    = (float*)deviceMallocCheck(B * T * C * sizeof(float), queue);
    float* d_preatt = (float*)deviceMallocCheck(B * NH * T * T * sizeof(float), queue);
    float* d_att    = (float*)deviceMallocCheck(B * NH * T * T * sizeof(float), queue);
    float* d_inp    = (float*)deviceMallocCheck(B * T * 3 * C * sizeof(float), queue);
    
    queue.fill<float>(d_out, float(0), B * T * C);
    queue.fill<float>(d_preatt, float(0), B * NH * T * T);
    queue.fill<float>(d_att, float(0), B * NH * T * T);
    queue.memcpy(d_inp, inp, B * T * 3 * C * sizeof(float));
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
    attention_forward_cpu(out, preatt, att, inp, B, T, C, NH);

    // Test kernel 1
    if (kernel_num == 1) {
        printf("************************************************.\n");
        printf("Checking kernel set #%i.\n", kernel_num);
        attention_forward(kernel_num, queue, d_out, d_preatt, d_att, d_inp, B, T, C, NH, 0, 0);
        queue.wait();
        validate_result(queue, d_preatt, preatt, "preatt", B * NH * T * T, 1e-5f);
        validate_result(queue, d_att, att, "att", B * NH * T * T, 1e-5f);
        validate_result(queue, d_out, out, "out", B * T * C, 1e-5f);
        printf("All results match. Benchmarking kernel %i.\n", kernel_num);
        double elapsed_ms = benchmark_kernel(repeat_times, attention_forward,
                                             kernel_num, queue, d_out, d_preatt, d_att,
                                             d_inp, B, T, C, NH,0, 0);
        // napkin math: estimate the flops achieved
        // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
        double tflops = get_attention_fwd_tflops(elapsed_ms, B, T, C, NH);
        printf("kernel %2i | time %.4f ms | tflops %.2f\n", kernel_num, elapsed_ms, tflops);

        queue.fill<float>(d_out, float(0), B * T * C);
        queue.fill<float>(d_preatt, float(0), B * NH * T * T);
        queue.fill<float>(d_att, float(0), B * NH * T * T);
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
                attention_forward(kernel_num, queue, d_out, d_preatt, d_att, d_inp, B, T, C, NH, sg_per_wg, sg_size);
                queue.wait();
                validate_result(queue, d_preatt, preatt, "preatt", B * NH * T * T, 1e-5f);
                validate_result(queue, d_att, att, "att", B * NH * T * T, 1e-5f);
                validate_result(queue, d_out, out, "out", B * T * C, 1e-5f);
                printf("All results match. Benchmarking kernel %i<%i, %i>.\n", kernel_num, sg_per_wg, sg_size);
                double elapsed_ms = benchmark_kernel(repeat_times, attention_forward,
                                                     kernel_num, queue, d_out, d_preatt, d_att,
                                                     d_inp, B, T, C, NH, sg_per_wg, sg_size);
                // napkin math: estimate the flops achieved
                // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
                double tflops = get_attention_fwd_tflops(elapsed_ms, B, T, C, NH);
                printf("kernel %2i<%2i, %2i> | time %.4f ms | tflops %.2f\n", kernel_num, sg_per_wg, sg_size, elapsed_ms, tflops);

                queue.fill<float>(d_out, float(0), B * T * C);
                queue.fill<float>(d_preatt, float(0), B * NH * T * T);
                queue.fill<float>(d_att, float(0), B * NH * T * T);
                queue.wait();
            }
            printf("\n");
        }
    }
    if (run_all) kernel_num++;
    printf("\n");

#ifdef ONEMKL_INTERFACES
    // Test kernel 3
    if (kernel_num == 3) {
        printf("************************************************.\n");
        printf("Checking kernel set #%i.\n", kernel_num);
        attention_forward(kernel_num, queue, d_out, d_preatt, d_att, d_inp, B, T, C, NH, 0, 0);
        queue.wait();
        validate_result(queue, d_preatt, preatt, "preatt", B * NH * T * T, 1e-5f);
        validate_result(queue, d_att, att, "att", B * NH * T * T, 1e-5f);
        validate_result(queue, d_out, out, "out", B * T * C, 1e-5f);
        printf("All results match. Benchmarking kernel %i.\n", kernel_num);
        double elapsed_ms = benchmark_kernel(repeat_times, attention_forward,
                                             kernel_num, queue, d_out, d_preatt, d_att,
                                             d_inp, B, T, C, NH,0, 0);
        // napkin math: estimate the flops achieved
        // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
        double tflops = get_attention_fwd_tflops(elapsed_ms, B, T, C, NH);
        printf("kernel %2i | time %.4f ms | tflops %.2f\n", kernel_num, elapsed_ms, tflops);

        queue.fill<float>(d_out, float(0), B * T * C);
        queue.fill<float>(d_preatt, float(0), B * NH * T * T);
        queue.fill<float>(d_att, float(0), B * NH * T * T);
        queue.wait();
    }
    if (run_all) kernel_num++;
    printf("\n");
#endif

    // free memory
    queue.wait_and_throw();
    sycl::free(out, queue);
    sycl::free(preatt, queue);
    sycl::free(att, queue);
    sycl::free(inp, queue);
    sycl::free(d_out, queue);
    sycl::free(d_preatt, queue);
    sycl::free(d_att, queue);
    sycl::free(d_inp, queue);
    return 0;
}
