/*
Kernels for attention backward pass.

GPU Compile example for Intel PVC GPU with ahead-of-time compilation:
icpx -O3 -fsycl -fno-sycl-id-queries-fit-in-int -std=c++17 -xhost -DLLMSYCL -fp-model=precise \
     -qopenmp -qopenmp-simd -DOMP  -DTIMEPROFILE \
     -fsycl-targets=spir64_gen -Xs "-device pvc" -D_DISABLE_SG_SIZE_8 \
     attention_backward.sycl.cpp -lsycl -lOpenCL -liomp5 -o attention_backward

version 1 is sycl::range<> port from CPU code to kernel: parallelizes over B*T, C, and loops over C
./attention_backward 1

version 2 is sycl::nd_range<> port of version 1 with different SG_SIZE, sg_per_wg
./attention_backward 2

TODO: Try oneDNN / oneMKL for version 3

Observations as of 08/06/2024: version 2 with is faster on CPU device
                               version 3 with {SG_SIZE, SG_PER_WG} = {32, 32} is faster on GPU device

*/

#include <stdio.h>
#include <stdlib.h>
#ifdef OMP
#include <omp.h>
#endif
#include "common.h"
#include <sycl/sycl.hpp>

#define SQRT sqrtf
#define EXP  expf

// ----------------------------------------------------------------------------
// TFLOP/s
double get_attention_bwd_tflops(double elapsed_ms, size_t B, size_t T, size_t C, size_t NH) {
    // Time is in milliseconds
    size_t hs = C/NH;
    // Counting is based on reference kernel
    return (double) B * NH * (T*(T+1)*5*hs + T*(T+1)*(T+2)/6*4) / elapsed_ms * 1e-9;
}

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 attention forward pass
void attention_forward_cpu(float* out, float* preatt, float* att,
                           const float* inp,
                           const int B, const int T, const int C, const int NH) {
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
                // pad with -FLT_MAX outside of autoregressive region for debugging comparisons
                for (int t2 = t+1; t2 < T; t2++) {
                    preatt_bth[t2] = -FLT_MAX;
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
}

// GPT-2 attention backward pass
void attention_backward_cpu(float* dinp, float* dpreatt, float* datt,
                            const float* dout, const float* inp, const float* att,
                            const int B, const int T, const int C, const int NH) {
    timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.f / sqrtf(hs);

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < NH; h++) {
            for (int t = 0; t < T; t++) {
                const float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                float* datt_bth = datt + b*NH*T*T + h*T*T + t*T; // easy parallel
                float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T; // easy parallel
                float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs; // easy parallel
                const float* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                const float* dout_bth = dout + b * T * C + t * C + h * hs;
                // t*(t+1)/2 * 4*hs
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                // t*(t+1)*(t+2)/6 * 4
                for (int t2 = 0; t2 <= t; t2++) {
                    // t*(t+1)/2 * 4
                    for (int t3 = 0; t3 <= t; t3++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                // t*(t+1)/2 * 6*hs
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = get_elapsed_ms(start, end);
    double tflops = get_attention_bwd_tflops(elapsed_ms, B, T, C, NH);
    printf("kernel ref OMP | time %.4f ms | tflops %.2f\n", elapsed_ms, tflops);
}

// ----------------------------------------------------------------------------
// GPU kernels

sycl::event attention_backward1(sycl::queue &queue, float* dinp, float* dpreatt, float* datt,
                                const float* dout, const float* inp, const float* att,
                                const int B, const int T, const int C, const int NH,
                                const std::vector<sycl::event> &dependencies = {}) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    const int C3 = C*3;
    const int hs = C / NH; // head size
    const float scale = 1.f / SQRT(float(hs));

    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<2> item) {
            const size_t b = item.get_id(0);
            const size_t h = item.get_id(1);
            for (size_t t = 0; t < T; t++) {
                const float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                float* datt_bth = datt + b*NH*T*T + h*T*T + t*T; // easy parallel
                float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T; // easy parallel
                float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs; // easy parallel
                const float* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                const float* dout_bth = dout + b * T * C + t * C + h * hs;
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    float datt_val = float(0);
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_val     += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                    datt_bth[t2] += datt_val;
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for (int t3 = 0; t3 <= t; t3++) {
                    const float att_bth_t3 = att_bth[t3];
                    float dpreatt_val = float(0);
                    for (int t2 = 0; t2 <= t; t2++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth_t3);
                        dpreatt_val += local_derivative * datt_bth[t2];
                    }
                    dpreatt_bth[t3] += dpreatt_val;
                }

                // backward pass 1, the query @ key matmul
                for (int t2 = 0; t2 <= t; t2++) {
                    const float scaled_dpreatt_t2 = dpreatt_bth[t2] * scale;
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dquery_t[i] += key_t2[i] * scaled_dpreatt_t2;
                        dkey_t2[i] += query_t[i] * scaled_dpreatt_t2;
                    }
                }
            }
        };
        cgh.parallel_for<class ker_attention_backward_1>(sycl::range<2>(B, NH), kernel);
    });
    return last;
}

sycl::event attention_backward2(sycl::queue &queue, float* dinp, float* dpreatt, float* datt,
                                const float* dout, const float* inp, const float* att,
                                const int B, const int T, const int C, const int NH,
                                const std::vector<sycl::event> &dependencies = {}) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    const int C3 = C*3;
    const int hs = C / NH; // head size
    const float scale = 1.f / SQRT(float(hs));

    sycl::event ev_datt = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<2> item) {
            const size_t b = item.get_id(0);
            const size_t h = item.get_id(1);
            for (size_t t = 0; t < T; t++) {

                const float* dout_bth = dout + b * T * C + t * C + h * hs;
                float* datt_bth       = datt + b*NH*T*T + h*T*T + t*T;

                // backward pass 4, through the value accumulation
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* value_t2 = inp  + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float datt_val = float(0);
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_val += value_t2[i] * dout_bth[i];
                    }
                    datt_bth[t2] += datt_val;
                }
            }
        };
        cgh.parallel_for<class ker_attention_backward_2_datt>(sycl::range<2>(B, NH), kernel);
    });

    sycl::event ev_dvalue = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<3> item) {
            const size_t b = item.get_id(0);
            const size_t h = item.get_id(1);
            const size_t i = item.get_id(2);
            for (size_t t = 0; t < T; t++) {

                const float* dout_bth = dout + b * T * C + t * C + h * hs;
                const float* att_bth  = att  + b*NH*T*T + h*T*T + t*T;

                // backward pass 4, through the value accumulation
                for (int t2 = 0; t2 <= t; t2++) {
                    float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    // in the forward pass this was:
                    // out_bth[i] += att_bth[t2] * value_t2[i];
                    // so now we have:
                    dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                }
            }
        };
        cgh.parallel_for<class ker_attention_backward_2_dvalue>(sycl::range<3>(B, NH, hs), kernel);
    });

    sycl::event ev_dpreatt = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on({ev_datt, ev_dvalue});
        auto kernel = [=](sycl::item<2> item) {
            const size_t b = item.get_id(0);
            const size_t h = item.get_id(1);
            for (size_t t = 0; t < T; t++) {

                const float* att_bth  = att  + b*NH*T*T + h*T*T + t*T;
                const float* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for (int t2 = 0; t2 <= t; t2++) {
                    for (int t3 = 0; t3 <= t; t3++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }
            }
        };
        cgh.parallel_for<class ker_attention_backward_2_dpreatt>(sycl::range<2>(B, NH), kernel);
    });

    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(ev_dpreatt);
        auto kernel = [=](sycl::item<3> item) {
            const size_t b = item.get_id(0);
            const size_t h = item.get_id(1);
            const size_t i = item.get_id(2);
            for (size_t t = 0; t < T; t++) {

                const float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                const float* query_t     = inp  + b * T * C3 + t * C3 + h * hs;
                float* dquery_t          = dinp + b * T * C3 + t * C3 + h * hs;

                // backward pass 1, the query @ key matmul
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* key_t2 = inp  + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    float* dkey_t2      = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                }
            }
        };
        cgh.parallel_for<class ker_attention_backward_2_dquery_dkey>(sycl::range<3>(B, NH, hs), kernel);
    });
    return last;
}

template <int SG_SIZE>
class ker_attention_backward_3;

template <typename T>
using atomic_ref_relaxed = sycl::atomic_ref<T, sycl::memory_order::relaxed,
                                            sycl::memory_scope::work_group,
                                            sycl::access::address_space::global_space>;

template<int SG_SIZE>
sycl::event attention_backward3(sycl::queue &queue, float* dinp, float* dpreatt, float* datt,
                                const float* dout, const float* inp, const float* att,
                                const int B, const int T, const int C, const int NH, const int sg_per_wg,
                                const std::vector<sycl::event> &dependencies = {}) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    const int C3 = C*3;
    const int hs = C / NH; // head size
    const float scale = 1.f / SQRT(float(hs));
    const int adjusted_sg_per_wg = std::min((T + SG_SIZE - 1)/SG_SIZE, sg_per_wg);
    const int wg_size = adjusted_sg_per_wg * SG_SIZE;
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
            const size_t b      = item.get_global_id(0);
            const size_t h      = item.get_global_id(1);
            const size_t lid    = item.get_local_linear_id();
            sycl::group gr      = item.get_group();
            sycl::sub_group sgr = item.get_sub_group();
            const size_t sg_gid = sgr.get_group_id();
            const size_t sg_cid = sgr.get_local_id();

            for (size_t t = 0; t < T; t++) {
                const float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                float* datt_bth = datt + b*NH*T*T + h*T*T + t*T; // easy parallel
                float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T; // easy parallel
                float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs; // easy parallel
                const float* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                const float* dout_bth = dout + b * T * C + t * C + h * hs;
                for (int t2 = sg_gid; t2 <= t; t2+= adjusted_sg_per_wg) {
                    const float att_bth_t2 = att_bth[t2];
                    const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    float datt_val = float(0);
                    for (int i = sg_cid; i < hs; i+=SG_SIZE) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        const float dout_bth_i = dout_bth[i];
                        atomic_ref_relaxed<float> dvalue_t2_atomic(dvalue_t2[i]);
                        dvalue_t2_atomic.fetch_add(att_bth_t2 * dout_bth_i);
                        datt_val += value_t2[i] * dout_bth_i;
                        //datt_val     += value_t2[i] * dout_bth[i];
                        //dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                    //datt_bth[t2] += datt_val;
                    datt_val = sycl::reduce_over_group(sgr, datt_val, sycl::plus<float>());
                    if (sg_cid == 0) datt_bth[t2] += datt_val;
                }
                sycl::group_barrier(gr);

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for (int t3 = sg_gid; t3 <= t; t3+=adjusted_sg_per_wg) {
                    const float att_bth_t3  = att_bth[t3];
                    float dpreatt_val = float(0);
                    for (int t2 = sg_cid; t2 <= t; t2+=SG_SIZE) {
                        const float indicator = t2 == t3 ? 1.0f : 0.0f;
                        const float local_derivative = att_bth[t2] * (indicator - att_bth_t3);
                        dpreatt_val += local_derivative * datt_bth[t2];
                        //dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                    dpreatt_val = sycl::reduce_over_group(sgr, dpreatt_val, sycl::plus<float>());
                    if (sg_cid == 0) dpreatt_bth[t3] += dpreatt_val;
                    //atomic_ref_relaxed<float> dpreatt_bth_atomic(dpreatt_bth[t3]);
                    //dpreatt_bth_atomic.fetch_add(local_derivative * datt_bth[t2]);
                }
                sycl::group_barrier(gr);

                // backward pass 1, the query @ key matmul
                for (int t2 = sg_gid; t2 <= t; t2+=adjusted_sg_per_wg) {
                    const float scaled_dpreatt_t2 = dpreatt_bth[t2] * scale;
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    for (int i = sg_cid; i < hs; i+=SG_SIZE) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        atomic_ref_relaxed<float> dquery_t_atomic(dquery_t[i]);
                        atomic_ref_relaxed<float> dkey_t2_atomic(dkey_t2[i]);
                        dquery_t_atomic.fetch_add(key_t2[i] * scaled_dpreatt_t2);
                        dkey_t2_atomic.fetch_add(query_t[i] * scaled_dpreatt_t2);
                        //dquery_t[i] += key_t2[i] * dpreatt_bth_t2 * scale;
                        //dkey_t2[i] += query_t[i] * dpreatt_bth_t2 * scale;
                    }
                }
                sycl::group_barrier(gr);
            }
        };
        cgh.parallel_for<class ker_attention_backward_3<SG_SIZE>>(
                sycl::nd_range<3>(sycl::range<3>(B, NH, wg_size),
                                  sycl::range<3>(1, 1, wg_size)), kernel);
    });
    return last;
}

// kernel version dispatch
sycl::event attention_backward(int kernel_num,
                               sycl::queue &queue, float* dinp, float* dpreatt, float* datt,
                               const float* dout, const float* inp, const float* att,
                               const int B, const int T, const int C, const int NH, const int sg_per_wg,
                               const int sg_size) {
    switch (kernel_num) {
        case 1: {
            return attention_backward1(queue, dinp, dpreatt, datt, dout, inp, att, B, T, C, NH);
        } break;
        case 2: {
            return attention_backward2(queue, dinp, dpreatt, datt, dout, inp, att, B, T, C, NH);
        } break;
        case 3: {
            if (sg_size == 32) {
                return attention_backward3<32>(queue, dinp, dpreatt, datt, dout, inp, att, B, T, C, NH, sg_per_wg);
            }
#ifndef _DISABLE_SG_SIZE_16
            else if (sg_size == 16) {
                return attention_backward3<16>(queue, dinp, dpreatt, datt, dout, inp, att, B, T, C, NH, sg_per_wg);
            }
#endif
#ifndef _DISABLE_SG_SIZE_8
            else if (sg_size == 8) {
                return attention_backward3< 8>(queue, dinp, dpreatt, datt, dout, inp, att, B, T, C, NH, sg_per_wg);
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
    float* inp     = (float*)hostMallocCheck(B * T * 3 * C * sizeof(float), queue);
    float* preatt  = (float*)hostMallocCheck(B * NH * T * T * sizeof(float), queue);
    float* att     = (float*)hostMallocCheck(B * NH * T * T * sizeof(float), queue);
    float* out     = (float*)hostMallocCheck(B * T * C * sizeof(float), queue);
    float* dinp    = (float*)hostMallocCheck(B * T * 3 * C * sizeof(float), queue);
    float* dpreatt = (float*)hostMallocCheck(B * NH * T * T * sizeof(float), queue);
    float* datt    = (float*)hostMallocCheck(B * NH * T * T * sizeof(float), queue);
    float* dout    = (float*)hostMallocCheck(B * T * C * sizeof(float), queue);

    make_random_float(inp, B * T * 3 * C);
    queue.fill<float>(preatt, float(0), B * NH * T * T);
    queue.fill<float>(att, float(0), B * NH * T * T);
    queue.fill<float>(out, float(0), B * T * C);
    queue.fill<float>(dinp, float(0), B * T * 3 * C);
    queue.fill<float>(dpreatt, float(0), B * NH * T * T);
    queue.fill<float>(datt, float(0), B * NH * T * T);
    make_random_float(dout, B * T * C);
    queue.wait();

    printf("Running one attention_forward call to get inputs {out, att}.\n");
    attention_forward_cpu(out, preatt, att, inp, B, T, C, NH);

    // move to GPU
    float* d_inp     = (float*)deviceMallocCheck(B * T * 3 * C * sizeof(float), queue);
    float* d_preatt  = (float*)deviceMallocCheck(B * NH * T * T * sizeof(float), queue);
    float* d_att     = (float*)deviceMallocCheck(B * NH * T * T * sizeof(float), queue);
    float* d_out     = (float*)deviceMallocCheck(B * T * C * sizeof(float), queue);
    float* d_dinp    = (float*)deviceMallocCheck(B * T * 3 * C * sizeof(float), queue);
    float* d_dpreatt = (float*)deviceMallocCheck(B * NH * T * T * sizeof(float), queue);
    float* d_datt    = (float*)deviceMallocCheck(B * NH * T * T * sizeof(float), queue);
    float* d_dout    = (float*)deviceMallocCheck(B * T * C * sizeof(float), queue);
    
    queue.memcpy(d_inp, inp, B * T * 3 * C * sizeof(float));
    // d_preatt not copied as it's not need in attention_backward
    queue.memcpy(d_att, att, B * NH * T * T * sizeof(float));       // Copying attention_forward calculated vals
    // d_out not copied as it's not need in attention_backward
    queue.fill<float>(d_dinp, float(0), B * T * 3 * C);
    queue.fill<float>(d_dpreatt, float(0), B * NH * T * T);
    queue.fill<float>(d_datt, float(0), B * NH * T * T);
    queue.memcpy(d_dout, dout, B * T * C * sizeof(float));
    queue.wait();

    // read kernel_num from command line
    int kernel_num = 1;
    bool run_all = true;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
        printf("Using kernel %d\n", kernel_num);
        run_all = false;
    }
    const int repeat_times = (dev.is_cpu() ? 1 : 5);

    // first check the correctness of the kernel
    printf("Running reference CPU kernel.\n");
    attention_backward_cpu(dinp, dpreatt, datt, dout, inp, att, B, T, C, NH);

    // Test kernels 1, 2
    for (int i = 1; i <= 2; i++) {
        if (kernel_num == i) {
            printf("************************************************.\n");
            printf("Checking kernel set #%i.\n", kernel_num);
            attention_backward(kernel_num, queue, d_dinp, d_dpreatt, d_datt, d_dout, d_inp, d_att, B, T, C, NH, 0, 0);
            queue.wait();
            validate_result(queue, d_dinp, dinp, "dinp", B * T * 3 * C, 1e-5f);
            validate_result(queue, d_dpreatt, dpreatt, "dpreatt", B * NH * T * T, 1e-5f);
            validate_result(queue, d_datt, datt, "datt", B * NH * T * T, 1e-5f);
            printf("All results match. Benchmarking kernel %i.\n", kernel_num);
            double elapsed_ms = benchmark_kernel(repeat_times, attention_backward,
                                                 kernel_num, queue, d_dinp, d_dpreatt, d_datt,
                                                 d_dout, d_inp, d_att,
                                                 B, T, C, NH, 0, 0);
            double tflops = get_attention_bwd_tflops(elapsed_ms, B, T, C, NH);
            printf("kernel %2i | time %.4f ms | tflops %.2f\n", kernel_num, elapsed_ms, tflops);

            queue.fill<float>(d_dinp, float(0), B * T * 3 * C);
            queue.fill<float>(d_dpreatt, float(0), B * NH * T * T);
            queue.fill<float>(d_datt, float(0), B * NH * T * T);
            queue.wait();
        }
        if (run_all) kernel_num++;
        printf("\n");
    }

    // Test kernel 3
    if (kernel_num == 3) {
        printf("************************************************.\n");
        printf("Checking kernel set #%i.\n", kernel_num);
        for (int sg_size : supported_sg_sizes) {
            if (sg_size < 8 || sg_size > 32) {printf("Skipping sg_size = %i\n", sg_size); continue;}
            printf("Testing sg_size = %i\n", sg_size);
            for (int sg_per_wg = 1; sg_per_wg <= max_workgroup_size/sg_size; sg_per_wg <<= 1) {
                printf("Checking kernel %i<%i, %i>.\n", kernel_num, sg_per_wg, sg_size);
                attention_backward(kernel_num, queue, d_dinp, d_dpreatt, d_datt, d_dout, d_inp, d_att, B, T, C, NH, sg_per_wg, sg_size);
                queue.wait();
                validate_result(queue, d_dinp, dinp, "dinp", B * T * 3 * C, 1e-5f);
                validate_result(queue, d_dpreatt, dpreatt, "dpreatt", B * NH * T * T, 1e-5f);
                validate_result(queue, d_datt, datt, "datt", B * NH * T * T, 1e-5f);
                printf("All results match. Benchmarking kernel %i<%i, %i>.\n", kernel_num, sg_per_wg, sg_size);
                double elapsed_ms = benchmark_kernel(repeat_times, attention_backward,
                                                     kernel_num, queue, d_dinp, d_dpreatt, d_datt,
                                                     d_dout, d_inp, d_att,
                                                     B, T, C, NH, sg_per_wg, sg_size);
                double tflops = get_attention_bwd_tflops(elapsed_ms, B, T, C, NH);
                printf("kernel %2i<%2i, %2i> | time %.4f ms | tflops %.2f\n", kernel_num, sg_per_wg, sg_size, elapsed_ms, tflops);

                queue.fill<float>(d_dinp, float(0), B * T * 3 * C);
                queue.fill<float>(d_dpreatt, float(0), B * NH * T * T);
                queue.fill<float>(d_datt, float(0), B * NH * T * T);
                queue.wait();
            }
            printf("\n");
        }
    }
    if (run_all) kernel_num++;
    printf("\n");

    // free memory
    queue.wait_and_throw();
    sycl::free(inp, queue);
    sycl::free(preatt, queue);
    sycl::free(att, queue);
    sycl::free(out, queue);
    sycl::free(dinp, queue);
    sycl::free(dpreatt, queue);
    sycl::free(datt, queue);
    sycl::free(dout, queue);
    sycl::free(d_inp, queue);
    sycl::free(d_preatt, queue);
    sycl::free(d_att, queue);
    sycl::free(d_out, queue);
    sycl::free(d_dinp, queue);
    sycl::free(d_dpreatt, queue);
    sycl::free(d_datt, queue);
    sycl::free(d_dout, queue);
    return 0;
}
