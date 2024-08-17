/*
This file is a modified version of train_gpt2.c CPU C code file that trains
the GPT-2 model. This is a SYCL port meant to be accelerator-agnostic to
- Run on SYCL CPU device,
- Tun on Intel GPU device,
- Run on NVidia GPU device,
- TBD: run on AMD GPU device.
- It is supposed to be using SYCL 2020 Specification compiliant APIs only and
  not any vendor-specific extensions (e.g. Intel ESIMD), although those may
  provide further speedups to the kernels.
*/

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#ifdef OMP
#include <omp.h>
#endif
#include <sycl/sycl.hpp>
#include <vector>

// Flip comment to print some traces
//#define ftrace(fmt, ...) printf("%s:%d" fmt "\n", __func__, __LINE__, ##__VA_ARGS__);
#define ftrace(fmt, ...)

#define SQRT sqrtf /* sycl::sqrt */
#define FABS fabsf /* sycl::fabs */
#define EXP  expf  /* sycl::exp  */
#define LOG  logf  /* sycl::log  */
#define POW  powf  /* sycl::pow  */
#define TANH tanhf /* sycl::tanh */
#define COSH coshf /* sycl::cosh */

// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
//          deviceMallocCheck, sharedMallocCheck, hostMallocCheck
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#include "llmc/dataloader.h"

#ifdef TESTING
// Due to the nature of validation in test_gpt2.sycl.cpp, we currently
// need shared allocation when testing
#define xxxMallocCheck sharedMallocCheck
#else
#define xxxMallocCheck deviceMallocCheck
#endif

// Helper for collecting timing info in milliseconds
inline double get_elapsed_ms(struct timespec &start, struct timespec &end) {
    return 1e3*((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
}

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

template <int SG_SIZE>
class ker_encoder_forward_2;

template<int SG_SIZE>
sycl::event encoder_forward2(sycl::queue &queue, float* out,
                             const int* inp, const float* wte, const float* wpe,
                             const int B, const int T, const int C, const int sg_per_wg,
                             const std::vector<sycl::event> &dependencies = {}) {
    const int wg_size = sg_per_wg * SG_SIZE;
    const int ceilC = (C + wg_size - 1) & (-wg_size);
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const size_t b = item.get_global_id(0);
            const size_t t = item.get_global_id(1);
            const size_t c = item.get_global_id(2);
            const size_t bt = b*T+t;

            if (c >= C) return;

            // seek to the output position in out[b,t,:]
            float* out_bt = out + bt * C;
            // get the index of the token at inp[b, t]
            const int ix = inp[bt];
            // seek to the position in wte corresponding to the token
            const float* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            const float* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            out_bt[c] = wte_ix[c] + wpe_t[c];
        };
        cgh.parallel_for<class ker_encoder_forward_2<SG_SIZE>>(
                sycl::nd_range<3>(sycl::range<3>(B, T, ceilC),
                                  sycl::range<3>(1, 1, wg_size)), kernel);
    });
    return last;
}

sycl::event encoder_forward(sycl::queue &queue, float* out,
                            const int* inp, const float* wte, const float* wpe,
                            const int B, const int T, const int C,
                            const std::vector<sycl::event> &dependencies = {}) {
    ftrace();
#if defined(SYCL_CPU)
    constexpr int sg_per_wg = 32;
    return encoder_forward2<32>(queue, out, inp, wte, wpe, B, T, C, sg_per_wg, dependencies);
#elif defined(SYCL_CUDA)
    constexpr int sg_per_wg = 8;
    return encoder_forward2<32>(queue, out, inp, wte, wpe, B, T, C, sg_per_wg, dependencies);
#else
    constexpr int sg_per_wg = 8;
    return encoder_forward2<32>(queue, out, inp, wte, wpe, B, T, C, sg_per_wg, dependencies);
#endif
}

template <int SG_SIZE>
class ker_encoder_backward_3_dwte;

template <int SG_SIZE>
class ker_encoder_backward_3_dwpe;

template<int SG_SIZE>
sycl::event encoder_backward3(sycl::queue &queue, float* dwte, float* dwpe,
                              const float* dout, const int* inp,
                              const int B, const int T, const int C, const int sg_per_wg,
                              const std::vector<sycl::event> &dependencies = {}) {
    const int adjusted_sg_per_wg = std::min((B + SG_SIZE - 1)/SG_SIZE, sg_per_wg);
    const int wg_size = adjusted_sg_per_wg * SG_SIZE;
    const size_t TC = T * C;
    sycl::event ev_dwpe = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const size_t t   = item.get_global_id(0);
            const size_t c   = item.get_global_id(1);
            const size_t tc  = t * C + c;
            const size_t lid = item.get_local_id(2);
            sycl::group gr   = item.get_group();

            const float *dout_tc = dout + tc;
            float val = 0.0f;
            for (size_t b = lid; b < B; b += wg_size) {
                val += dout_tc[b * TC];
            }
            val = sycl::reduce_over_group(gr, val, sycl::plus<float>());
            if (lid == 0) dwpe[tc] += val;
        };
        // 2D nd_range (T*C instead of T,C) is slightly faster, but unfortunately
        // does not work on A100 for some reason while 3D nd_range works.
        cgh.parallel_for<class ker_encoder_backward_3_dwpe<SG_SIZE>>(
                sycl::nd_range<3>(sycl::range<3>(T, C, wg_size),
                                  sycl::range<3>(1, 1, wg_size)), kernel);
    });

    // Caution: using atomics in this kernel affects bit-wise reproducibility
    // of the output. If bit-wise reproducibility is needed, we must use kernel
    // 2 instead.
    sycl::event ev_dwte = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<3> item) {
            const size_t b = item.get_id(0);
            const size_t t = item.get_id(1);
            const size_t c = item.get_id(2);
            const size_t bt = b * T + t;
            const int ix = inp[bt];
            float* dwte_ix = dwte + ix * C;
            const float* dout_bt = dout + bt * C;
            const float d = dout_bt[c];
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::system,
                             sycl::access::address_space::global_space> dwte_atomic(dwte_ix[c]);
            dwte_atomic.fetch_add(d);
        };
        cgh.parallel_for<class ker_encoder_backward_3_dwte<SG_SIZE>>(sycl::range<3>(B, T, C), kernel);
    });

    return queue.ext_oneapi_submit_barrier({ev_dwpe, ev_dwte});
}

sycl::event encoder_backward(sycl::queue &queue, float* dwte, float* dwpe,
                             const float* dout, const int* inp,
                             const int B, const int T, const int C,
                             const std::vector<sycl::event> &dependencies = {}) {
    ftrace();
#if defined(SYCL_CPU)
    constexpr int sg_per_wg = 2;
    return encoder_backward3<16>(queue, dwte, dwpe, dout, inp, B, T, C, sg_per_wg, dependencies);
#elif defined(SYCL_CUDA)
    constexpr int sg_per_wg = 1; // 2 gives HW-specific runtime errors in this case
    return encoder_backward3<32>(queue, dwte, dwpe, dout, inp, B, T, C, sg_per_wg, dependencies);
#else
    constexpr int sg_per_wg = 2;
    return encoder_backward3<32>(queue, dwte, dwpe, dout, inp, B, T, C, sg_per_wg, dependencies);
#endif
}

template <int SG_SIZE>
class ker_layernorm_forward_2;

template<int SG_SIZE>
sycl::event layernorm_forward2(sycl::queue &queue, float* out, float* mean, float* rstd,
                               const float* inp, const float* weight, const float* bias,
                               const int B, const int T, const int C, int sg_per_wg,
                               const std::vector<sycl::event> &dependencies = {}) {
    const float eps = 1e-5f;
    const int wg_size = sg_per_wg * SG_SIZE;
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const size_t bt           = item.get_global_id(0);
            const sycl::group gr      = item.get_group();
            //const sycl::sub_group sgr = item.get_sub_group();
            //const size_t sg_group_id  = sgr.get_group_id();
            //const size_t sgr_cid      = sgr.get_local_id();
            const size_t lid          = item.get_local_linear_id(); //sg_group_id * SG_SIZE + sgr_cid;

            // seek to the input position inp[b,t,:]
            const float* x = inp + bt * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = lid; i < C; i += wg_size) {
                m += x[i];
            }
            m = sycl::reduce_over_group(gr, m, sycl::plus<float>());
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = lid; i < C; i += wg_size) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = sycl::reduce_over_group(gr, v, sycl::plus<float>());
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            float s = 1.0f / SQRT(v + eps);
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

sycl::event layernorm_forward(sycl::queue &queue, float* out, float* mean, float* rstd,
                              const float* inp, const float* weight, const float* bias,
                              const int B, const int T, const int C,
                              const std::vector<sycl::event> &dependencies = {}) {
    ftrace();
#if defined(SYCL_CPU)
    constexpr int sg_per_wg = 2;
    return layernorm_forward2<16>(queue, out, mean, rstd, inp, weight, bias, B, T, C, sg_per_wg, dependencies);
#elif defined(SYCL_CUDA)
    constexpr int sg_per_wg = 4;
    return layernorm_forward2<32>(queue, out, mean, rstd, inp, weight, bias, B, T, C, sg_per_wg, dependencies);
#else
    constexpr int sg_per_wg = 4;
    return layernorm_forward2<32>(queue, out, mean, rstd, inp, weight, bias, B, T, C, sg_per_wg, dependencies);
#endif
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
        auto kernel = [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const size_t c            = item.get_global_id(0);
            const sycl::group gr      = item.get_group();
            const sycl::sub_group sgr = item.get_sub_group();
            const size_t lid          = item.get_local_linear_id();

            const float *inp_c  = inp + c;
            const float *dout_c = dout + c;

            float dbiasval = 0.0f;
            float dweightval = 0.0f;
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
        auto kernel = [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
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
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
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
                float dval = 0.0f;
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

sycl::event layernorm_backward(sycl::queue &queue, float* dinp, float* dweight, float* dbias,
                               float* dout, float* inp, float* weight, float* mean, float* rstd,
                               const int B, const int T, const int C,
                               const std::vector<sycl::event> &dependencies = {}) {
    ftrace();
#if defined(SYCL_CPU)
    constexpr int sg_per_wg = 2;
    return layernorm_backward2<16>(queue, dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, sg_per_wg, dependencies);
#elif defined(SYCL_CUDA)
    constexpr int sg_per_wg = 8;
    return layernorm_backward2<32>(queue, dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, sg_per_wg, dependencies);
#else
    constexpr int sg_per_wg = 8;
    return layernorm_backward2<32>(queue, dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, sg_per_wg, dependencies);
#endif
}

template <int SG_SIZE>
class ker_matmul_forward2;

template<int SG_SIZE>
sycl::event matmul_forward2(sycl::queue &queue, float* out,
                                 const float* inp, const float* weight, const float* bias,
                                 int B, int T, int C, int OC, int sg_per_wg,
                                 const std::vector<sycl::event> &dependencies = {})
{
    // Round up next multiple, sg_per_wg always assumed to be a power of 2
    const int ceilBT = (B*T + sg_per_wg - 1) & (-sg_per_wg);
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const size_t bt            = item.get_global_id(0);
            const size_t o             = item.get_global_id(1);
            const sycl::sub_group sgr  = item.get_sub_group();
            const std::uint8_t sgr_cid = sgr.get_local_id();

            if (bt >= B*T) return;

            const size_t inp_ind_start = bt * C;
            const size_t wt_ind_start  = o * C;
            const size_t out_ind_start = bt * OC;

            float val = 0.0f;
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
                const float b = ((bias != NULL) ? bias[o] : 0.0f);
                out[out_ind_start + o] = val + b;
            }
        };
        cgh.parallel_for<class ker_matmul_forward2<SG_SIZE>>(
                sycl::nd_range<3>(sycl::range<3>(ceilBT, OC, SG_SIZE),
                                  sycl::range<3>(sg_per_wg, 1, SG_SIZE)), kernel);
    });
    return last;
}

sycl::event matmul_forward(sycl::queue &queue, float* out,
                           const float* inp, const float* weight, const float* bias,
                           const int B, const int T, const int C, const int OC,
                           const std::vector<sycl::event> &dependencies = {}) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct or fallback to naive version
    constexpr int LOOP_UNROLL = 8;
    //if (B*T % LOOP_UNROLL != 0) {
    ftrace();
#if defined(SYCL_CPU)
        // sg_size = 32 had bad performance on CPU device
        constexpr int sg_per_wg = 4;
        return matmul_forward2<16>(queue, out, inp, weight, bias, B, T, C, OC, sg_per_wg, dependencies);
#elif defined(SYCL_CUDA)
        constexpr int sg_per_wg = 4;
        return matmul_forward2<32>(queue, out, inp, weight, bias, B, T, C, OC, sg_per_wg, dependencies);
#else
        constexpr int sg_per_wg = 4;
        return matmul_forward2<32>(queue, out, inp, weight, bias, B, T, C, OC, sg_per_wg, dependencies);
#endif
    //}

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
#ifdef PARALLEL_MATMUL_FWD
    const size_t BT   = B * T;
    const size_t nOBT = (B*T + LOOP_UNROLL - 1) / LOOP_UNROLL;
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<2> item) {
            const size_t obt  = item.get_id(0) * LOOP_UNROLL;
            const size_t o    = item.get_id(1);
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
                for (std::uint16_t ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    result[ibt] += inp_obt[ibt * C + i] * w;
                }
            }
            // write back results to main memory
            #pragma unroll(LOOP_UNROLL)
            for (std::uint16_t ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                out_obt[ibt * OC] = result[ibt];
            }
        };
        cgh.parallel_for<class ker_matmul_forward_2>(sycl::range<2>(nOBT, OC), kernel);
    });
#else
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=]() {
            #pragma omp parallel for
            for (int obt = 0; obt < B * T; obt += LOOP_UNROLL) {
                for (int o = 0; o < OC; o++) {
                    // we'll keep LOOP_UNROLL many results in registers
                    float result[LOOP_UNROLL];
                    // initialize the bias, if it exists
                    for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                        result[ibt] = (bias != NULL) ? bias[o] : 0.0f;
                    }
                    // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
                    // the value of weight[i + o * C] and reuse it.
                    // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
                    for (int i = 0; i < C; i++) {
                        float w = weight[i + o * C];
                        for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                            int bt = obt + ibt;
                            result[ibt] += inp[bt * C + i] * w;
                        }
                    }
                    // write back results to main memory
                    for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                        int bt = obt + ibt;
                        out[bt * OC + o] = result[ibt];
                    }
                }
            }
        };
        cgh.host_task(kernel);
    });
#endif
    return last;
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
        auto kernel = [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const size_t bt = item.get_global_id(0);
            const size_t c  = item.get_global_id(1);
            if (c >= C || bt >= B*T) return;
            float val = 0.0f;
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
        auto kernel = [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const size_t o = item.get_global_id(0);
            const size_t c = item.get_global_id(1);
            if (o >= OC || c >= C) return;
            float val = 0.0f;
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
            auto kernel = [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const size_t o            = item.get_global_id(0);
                const sycl::sub_group sgr = item.get_sub_group();
                const size_t sgr_cid      = sgr.get_local_id();
                if (o >= OC) return;
                float sum = 0.0f;
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

sycl::event matmul_backward(sycl::queue &queue, float* dinp, float* dweight, float* dbias,
                            const float* dout, const float* inp, const float* weight,
                            int B, int T, int C, int OC,
                            const std::vector<sycl::event> &dependencies = {}) {
    ftrace();
#if defined(SYCL_CPU)
    // sg_size = 32 had bad performance on CPU device
    constexpr int sg_per_wg = 1;
    return matmul_backward2<32>(queue, dinp, dweight, dbias, dout, inp, weight, B, T, C, OC, sg_per_wg, dependencies);
#elif defined(SYCL_CUDA)
    constexpr int sg_per_wg = 32;
    return matmul_backward2<32>(queue, dinp, dweight, dbias, dout, inp, weight, B, T, C, OC, sg_per_wg, dependencies);
#else
    constexpr int sg_per_wg = 32;
    return matmul_backward2<32>(queue, dinp, dweight, dbias, dout, inp, weight, B, T, C, OC, sg_per_wg, dependencies);
#endif
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
    const float scale = 1.0 / SQRT(float(hs));
    // Here we do something slightly different from usual, viz.,
    // we disallow workgroup size to exceed `hs` by too much.
    // It may exceed
    //const int wg_size = sg_per_wg * SG_SIZE;
    const int adjusted_sg_per_wg = std::min((hs + SG_SIZE - 1) / SG_SIZE, sg_per_wg);
    const int wg_size = adjusted_sg_per_wg * SG_SIZE;

    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const size_t bt     = item.get_global_id(0);
            const size_t b      = bt / T;
            const size_t t      = bt % T;
            const size_t h      = item.get_global_id(1);

            sycl::group gr      = item.get_group();
            const size_t lid    = item.get_local_linear_id(); // sg_gid * SG_SIZE + sg_cid

            const size_t ht      = h * T + t;
            const float* query_t = inp + bt * C3 + h * hs;
            float* preatt_bth    = preatt + b * NH * T * T + ht * T;
            float* att_bth       = att + b * NH * T * T + ht * T;

            // pass 1: calculate query dot key and maxval
            float maxval = -std::numeric_limits<float>::max();
            const float* key_t2_base = inp + b * T * C3 + h * hs + C; // +C because it's key
            for (int t2 = 0; t2 <= t; t2++) {
                const float* key_t2 = key_t2_base + t2 * C3; // +C because it's key

                // (query_t) dot (key_t2)
                float val = 0.0f;
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
            float expsum = 0.0f;
            for (int t2 = lid; t2 <= t; t2 += wg_size) {
                float expv = EXP(preatt_bth[t2] - maxval);
                expsum += expv;
                att_bth[t2] = expv;
            }
            expsum = sycl::reduce_over_group(gr, expsum, sycl::plus<float>());
            float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

            // pass 3: normalize to get the softmax
            for (int t2 = lid; t2 < T; t2 += wg_size) {
                if (t2 <= t) {
                    att_bth[t2] *= expsum_inv;
                } else {
                    // causal attention mask. not strictly necessary to set to zero here
                    // only doing this explicitly for debugging and checking to PyTorch
                    att_bth[t2] = 0.0f;
                }
            }

            // pass 4: accumulate weighted values into the output of attention
            float* out_bth = out + bt * C + h * hs;
            for (int i = lid; i < hs; i += wg_size) { out_bth[i] = 0.0f; }
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

sycl::event attention_forward(sycl::queue &queue, float* out, float* preatt, float* att,
                              const float* inp,
                              const int B, const int T, const int C, const int NH,
                              const std::vector<sycl::event> &dependencies = {}) {
    ftrace();
#if defined(SYCL_CPU)
    // sg_size = 32 had bad performance on CPU device
    constexpr int sg_per_wg = 2;
    return attention_forward2<16>(queue, out, preatt, att, inp, B, T, C, NH, sg_per_wg, dependencies);
#elif defined(SYCL_CUDA)
    constexpr int sg_per_wg = 2;
    return attention_forward2<32>(queue, out, preatt, att, inp, B, T, C, NH, sg_per_wg, dependencies);
#else
    constexpr int sg_per_wg = 2;
    return attention_forward2<32>(queue, out, preatt, att, inp, B, T, C, NH, sg_per_wg, dependencies);
#endif
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
        auto kernel = [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
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
                    float datt_val = 0.0f;
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
                    float dpreatt_val = 0.0f;
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

sycl::event attention_backward(sycl::queue &queue, float* dinp, float* dpreatt, float* datt,
                               const float* dout, const float* inp, const float* att,
                               const int B, const int T, const int C, const int NH,
                               const std::vector<sycl::event> &dependencies = {}) {
#if defined(SYCL_CPU)
    // This particular attention_backward kernel is actually not too great on
    // CPU device compared to some other kernels, but fantastic on GPU
    constexpr int sg_per_wg = 4;
    return attention_backward3<32>(queue, dinp, dpreatt, datt, dout, inp, att, B, T, C, NH, sg_per_wg, dependencies);
#elif defined(SYCL_CUDA)
    constexpr int sg_per_wg = 32;
    return attention_backward3<32>(queue, dinp, dpreatt, datt, dout, inp, att, B, T, C, NH, sg_per_wg, dependencies);
#else
    constexpr int sg_per_wg = 32;
    return attention_backward3<32>(queue, dinp, dpreatt, datt, dout, inp, att, B, T, C, NH, sg_per_wg, dependencies);
#endif
}

#define GELU_SCALING_FACTOR SQRT(2.0f / M_PI)
template <int SG_SIZE>
class ker_gelu_forward_2;

template<int SG_SIZE>
sycl::event gelu_forward2(sycl::queue &queue, float* out,
                          const float* inp, const int N, const int sg_per_wg,
                          const std::vector<sycl::event> &dependencies = {}) {
    const int adjusted_sg_per_wg = std::min((N + SG_SIZE - 1) / SG_SIZE, sg_per_wg);
    const int wg_size = adjusted_sg_per_wg * SG_SIZE;
    const int ceilN = ((N + wg_size - 1) / wg_size) * wg_size;

    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
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

sycl::event gelu_forward(sycl::queue &queue, float* out,
                         const float* inp, const int N,
                         const std::vector<sycl::event> &dependencies = {}) {
    ftrace();
#if defined(SYCL_CPU)
    constexpr int sg_per_wg = 128;
    return gelu_forward2<16>(queue, out, inp, N, sg_per_wg, dependencies);
#elif defined(SYCL_CUDA)
    constexpr int sg_per_wg = 4;
    return gelu_forward2<32>(queue, out, inp, N, sg_per_wg, dependencies);
#else
    constexpr int sg_per_wg = 4;
    return gelu_forward2<32>(queue, out, inp, N, sg_per_wg, dependencies);
#endif
}

template <int SG_SIZE>
class ker_gelu_backward_2;

// TBD if the following is still needed; Warnings thrown with ICPX compiler
//// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
//#pragma float_control(precise, on, push)
//#if defined(__GNUC__) && !defined(__clang__)
//__attribute__((optimize("no-finite-math-only")))
//#endif
template<int SG_SIZE>
sycl::event gelu_backward2(sycl::queue &queue, float* dinp,
                           const float* inp, const float* dout, const int N, const int sg_per_wg,
                           const std::vector<sycl::event> &dependencies = {}) {
    const int adjusted_sg_per_wg = std::min((N + SG_SIZE - 1) / SG_SIZE, sg_per_wg);
    const int wg_size = adjusted_sg_per_wg * SG_SIZE;
    const int ceilN = ((N + wg_size - 1) / wg_size) * wg_size;

    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
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
//#pragma float_control(pop)

sycl::event gelu_backward(sycl::queue &queue, float* dinp,
                          const float* inp, const float* dout, const int N,
                          const std::vector<sycl::event> &dependencies = {}) {
    ftrace();
#if defined(SYCL_CPU)
    constexpr int sg_per_wg = 8;
    return gelu_backward2<16>(queue, dinp, inp, dout, N, sg_per_wg, dependencies);
#elif defined(SYCL_CUDA)
    constexpr int sg_per_wg = 2;
    return gelu_backward2<32>(queue, dinp, inp, dout, N, sg_per_wg, dependencies);
#else
    constexpr int sg_per_wg = 2;
    return gelu_backward2<32>(queue, dinp, inp, dout, N, sg_per_wg, dependencies);
#endif
}

template <int SG_SIZE>
class ker_residual_forward_2;

template <int SG_SIZE>
sycl::event residual_forward2(sycl::queue &queue, float* out,
                              const float* inp1, const float* inp2, const int N, const int sg_per_wg,
                              const std::vector<sycl::event> &dependencies = {}) {
    const int adjusted_sg_per_wg = std::min((N + SG_SIZE - 1) / SG_SIZE, sg_per_wg);
    const int wg_size = adjusted_sg_per_wg * SG_SIZE;
    const int ceilN = ((N + wg_size - 1) & (-wg_size));

    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const size_t i = item.get_global_id();
            if (i < N) out[i] = inp1[i] + inp2[i];
        };
        cgh.parallel_for<class ker_residual_forward_2<SG_SIZE>>(
                sycl::nd_range<1>(sycl::range<1>(ceilN), sycl::range<1>(wg_size)), kernel);
    });
    return last;
}

sycl::event residual_forward(sycl::queue &queue, float* dinp,
                             const float* inp, const float* dout, const int N,
                             const std::vector<sycl::event> &dependencies = {}) {
    ftrace();
#if defined(SYCL_CPU)
    constexpr int sg_per_wg = 8;
    return residual_forward2<16>(queue, dinp, inp, dout, N, sg_per_wg, dependencies);
#elif defined(SYCL_CUDA)
    constexpr int sg_per_wg = 32;
    return residual_forward2<32>(queue, dinp, inp, dout, N, sg_per_wg, dependencies);
#else
    constexpr int sg_per_wg = 32;
    return residual_forward2<32>(queue, dinp, inp, dout, N, sg_per_wg, dependencies);
#endif
}

template <int SG_SIZE>
class ker_residual_backward_2;

template<int SG_SIZE>
sycl::event residual_backward2(sycl::queue &queue, float* dinp1, float* dinp2,
                               const float* dout, const int N, const int sg_per_wg,
                               const std::vector<sycl::event> &dependencies = {}) {
    const int adjusted_sg_per_wg = std::min((N + SG_SIZE - 1) / SG_SIZE, sg_per_wg);
    const int wg_size = adjusted_sg_per_wg * SG_SIZE;
    const int ceilN = ((N + wg_size - 1) & (-wg_size));

    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const size_t i = item.get_global_id(0);
            if (i < N) {
                const float d = dout[i];
                dinp1[i] += d;
                dinp2[i] += d;
            }
        };
        cgh.parallel_for<class ker_residual_backward_2<SG_SIZE>>(
                sycl::nd_range<1>(sycl::range<1>(ceilN), sycl::range<1>(wg_size)), kernel);
    });
    return last;
}

sycl::event residual_backward(sycl::queue &queue, float* dinp1, float* dinp2,
                              const float* dout, const int N,
                              const std::vector<sycl::event> &dependencies = {}) {
    ftrace();
#if defined(SYCL_CPU)
    constexpr int sg_per_wg = 8;
    return residual_backward2<16>(queue, dinp1, dinp2, dout, N, sg_per_wg, dependencies);
#elif defined(SYCL_CUDA)
    constexpr int sg_per_wg = 32;
    return residual_backward2<32>(queue, dinp1, dinp2, dout, N, sg_per_wg, dependencies);
#else
    constexpr int sg_per_wg = 32;
    return residual_backward2<32>(queue, dinp1, dinp2, dout, N, sg_per_wg, dependencies);
#endif
}

template <int SG_SIZE>
class ker_online_softmax_forward_2;

template<int SG_SIZE>
sycl::event online_softmax_forward2(sycl::queue &queue, float* probs, const float* logits,
                                    const int B, const int T, const int V, const int Vp, const int sg_per_wg,
                                    const std::vector<sycl::event> &dependencies = {}) {
    // Possibly we must force a power-of-2 sg_per_wg instead of adjusting it
    // for reduction loops to work as expected.
    const int adjusted_sg_per_wg = std::min((V + SG_SIZE - 1) / SG_SIZE, sg_per_wg);
    const int wg_size = adjusted_sg_per_wg * SG_SIZE;

    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const size_t bt  = item.get_global_id(0);
            const size_t lid = item.get_local_linear_id();
            sycl::group gr   = item.get_group();

            // probs <- softmax(logits)
            const float* logits_bt = logits + bt * Vp;
            float* probs_bt        = probs + bt * Vp;

            // maxval is only calculated and subtracted for numerical stability
            float local_maxval=-std::numeric_limits<float>::max();
            float local_sum = 0.0f;
            for (int i = lid; i < V; i += wg_size) {
                if (logits_bt[i] > local_maxval) {
                    local_sum = local_sum * EXP(local_maxval - logits_bt[i]) + 1.0f;
                    local_maxval = logits_bt[i];
                }
                else {
                    local_sum += EXP(logits_bt[i] - local_maxval);
                }
            }
            // Perform local reduction within a sub_group
            const float maxval = sycl::reduce_over_group(gr, local_maxval, sycl::maximum<float>());
            local_sum         *= (maxval > local_maxval ? EXP(local_maxval - maxval) : 1.0f);
            const float sum    = sycl::reduce_over_group(gr, local_sum, sycl::plus<float>());

            // note we only loop to V, leaving the padded dimensions
            for (int i = lid; i < V; i += wg_size) {
                probs_bt[i] = EXP(logits_bt[i] - maxval) / sum;
            }

            // for extra super onlinety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            for (int i = V + lid; i < Vp; i += wg_size) {
                probs_bt[i] = 0.0f;
            }
        };
        cgh.parallel_for<class ker_online_softmax_forward_2<SG_SIZE>>(
                sycl::nd_range<2>(sycl::range<2>(B*T, wg_size), sycl::range<2>(1, wg_size)), kernel);
    });
    return last;
}

sycl::event softmax_forward(sycl::queue &queue, float* probs, float* logits,
                            const int B, const int T, const int V, const int Vp,
                            const std::vector<sycl::event> &dependencies = {}) {
    ftrace();
#if defined(SYCL_CPU)
    constexpr int sg_per_wg = 8;
    return online_softmax_forward2<32>(queue, probs, logits, B, T, V, Vp, sg_per_wg, dependencies);
#elif defined(SYCL_CUDA)
    constexpr int sg_per_wg = 32;
    return online_softmax_forward2<32>(queue, probs, logits, B, T, V, Vp, sg_per_wg, dependencies);
#else
    constexpr int sg_per_wg = 32;
    return online_softmax_forward2<32>(queue, probs, logits, B, T, V, Vp, sg_per_wg, dependencies);
#endif
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
        auto kernel = [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
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

sycl::event crossentropy_forward(sycl::queue &queue, float* losses,
                                 const float* probs, const int* targets,
                                 const int B, const int T, const int Vp,
                                 const std::vector<sycl::event> &dependencies = {}) {
    ftrace();
#if defined(SYCL_CPU)
    constexpr int sg_per_wg = 8;
    return crossentropy_forward2<16>(queue, losses, probs, targets, B, T, Vp, sg_per_wg, dependencies);
#elif defined(SYCL_CUDA)
    constexpr int sg_per_wg = 2;
    return crossentropy_forward2<32>(queue, losses, probs, targets, B, T, Vp, sg_per_wg, dependencies);
#else
    constexpr int sg_per_wg = 2;
    return crossentropy_forward2<32>(queue, losses, probs, targets, B, T, Vp, sg_per_wg, dependencies);
#endif
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
            const float indicator = (v == ix ? 1.0f : 0.0f);
            dlogits_bt[v] += (p - indicator) * dloss;
        };
        cgh.parallel_for<class ker_crossentropy_softmax_backward_4<SG_SIZE>>(
                sycl::nd_range<2>(sycl::range<2>(B * T, ceilV), sycl::range<2>(1, wg_size)), kernel);
    });
    return last;
}

sycl::event crossentropy_softmax_backward(sycl::queue &queue, float* dlogits,
                                          const float* dlosses, const float* probs, const int* targets,
                                          const size_t B, const size_t T, const size_t V, const size_t Vp,
                                          const std::vector<sycl::event> &dependencies = {}) {
    ftrace();
#if defined(SYCL_CPU)
    constexpr int sg_per_wg = 8;
    return crossentropy_softmax_backward4<32>(queue, dlogits, dlosses, probs, targets, B, T, V, Vp, sg_per_wg, dependencies);
#elif defined(SYCL_CUDA)
    constexpr int sg_per_wg = 8;
    return crossentropy_softmax_backward4<32>(queue, dlogits, dlosses, probs, targets, B, T, V, Vp, sg_per_wg, dependencies);
#else
    constexpr int sg_per_wg = 8;
    return crossentropy_softmax_backward4<32>(queue, dlogits, dlosses, probs, targets, B, T, V, Vp, sg_per_wg, dependencies);
#endif
}

template <int SG_SIZE>
class ker_adamw_2;

template<int SG_SIZE>
sycl::event adamw2(sycl::queue &queue,
                   float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, const int t, const size_t num_parameters,
                   const float learning_rate, const float beta1, const float beta2, const float eps, const float weight_decay,
                   const int sg_per_wg, const std::vector<sycl::event> &dependencies = {}) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    const int adjusted_sg_per_wg = std::min<size_t>((num_parameters + SG_SIZE - 1) / SG_SIZE, sg_per_wg);
    const int wg_size = adjusted_sg_per_wg * SG_SIZE;
    const size_t ceilN = ((num_parameters + wg_size - 1) & (-wg_size));

    const float oneminusbeta1 = 1.0f - beta1;
    const float oneminusbeta2 = 1.0f - beta2;
    const float beta1_correction = 1.0f - POW(beta1, t);
    const float beta2_correction = 1.0f - POW(beta2, t);

    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const size_t i = item.get_global_id(0);
            if (i >= num_parameters) return;
            const float param = params_memory[i];
            const float grad = grads_memory[i];

            // update the first moment (momentum)
            const float m = beta1 * m_memory[i] + oneminusbeta1 * grad;
            // update the second moment (RMSprop)
            const float v = beta2 * v_memory[i] + oneminusbeta2 * grad * grad;
            // bias-correct both moments
            const float m_hat = m / beta1_correction;
            const float v_hat = v / beta2_correction;

            // update
            m_memory[i] = m;
            v_memory[i] = v;
            params_memory[i] -= learning_rate * (m_hat / (SQRT(v_hat) + eps) + weight_decay * param);
        };
        cgh.parallel_for<class ker_adamw_2<SG_SIZE>>(
                sycl::nd_range<1>(sycl::range<1>(ceilN), sycl::range<1>(wg_size)), kernel);
    });
    return last;
}

sycl::event adamw(sycl::queue &queue,
                  float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, const int t, const size_t num_parameters,
                  const float learning_rate, const float beta1, const float beta2, const float eps, const float weight_decay,
                  const std::vector<sycl::event> &dependencies = {}) {
    ftrace();
#if defined(SYCL_CPU)
    constexpr int sg_per_wg = 64;
    return adamw2<16>(queue, params_memory, grads_memory, m_memory, v_memory, t, num_parameters,
                      learning_rate, beta1, beta2, eps, weight_decay, sg_per_wg, dependencies);
#elif defined(SYCL_CUDA)
    constexpr int sg_per_wg = 32;
    return adamw2<32>(queue, params_memory, grads_memory, m_memory, v_memory, t, num_parameters,
                      learning_rate, beta1, beta2, eps, weight_decay, sg_per_wg, dependencies);
#else
    constexpr int sg_per_wg = 32;
    return adamw2<32>(queue, params_memory, grads_memory, m_memory, v_memory, t, num_parameters,
                      learning_rate, beta1, beta2, eps, weight_decay, sg_per_wg, dependencies);
#endif
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
    size_t max_seq_len; // max sequence length, e.g. 1024
    size_t vocab_size; // vocab size, e.g. 50257
    size_t padded_vocab_size; // padded to e.g. %128==0, 50304
    size_t num_layers; // number of layers, e.g. 12
    size_t num_heads; // number of heads in attention, e.g. 12
    size_t channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
// These need to be device accessible USM pointers
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float* wte; // (V, C)
    float* wpe; // (maxT, C)
    float* ln1w; // (L, C)
    float* ln1b; // (L, C)
    float* qkvw; // (L, 3*C, C)
    float* qkvb; // (L, 3*C)
    float* attprojw; // (L, C, C)
    float* attprojb; // (L, C)
    float* ln2w; // (L, C)
    float* ln2b; // (L, C)
    float* fcw; // (L, 4*C, C)
    float* fcb; // (L, 4*C)
    float* fcprojw; // (L, C, 4*C)
    float* fcprojb; // (L, C)
    float* lnfw; // (C)
    float* lnfb; // (C)
} ParameterTensors;

void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb
}

// allocate memory for the parameters and point the individual tensors to the right places
float* malloc_and_point_parameters(sycl::queue &queue, ParameterTensors* params, size_t* param_sizes) {
    ftrace();
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    float* params_memory = (float*)xxxMallocCheck(num_parameters * sizeof(float), queue);
    // assign all the tensors
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 23
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* qkv; // (L, B, T, 3*C)
    float* atty; // (L, B, T, C)
    float* preatt; // (L, B, NH, T, T)
    float* att; // (L, B, NH, T, T)
    float* attproj; // (L, B, T, C)
    float* residual2; // (L, B, T, C)
    float* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, 4*C)
    float* fch_gelu; // (L, B, T, 4*C)
    float* fcproj; // (L, B, T, C)
    float* residual3; // (L, B, T, C)
    float* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* logits; // (B, T, V)
    float* probs; // (B, T, V)
    float* losses; // (B, T)
} ActivationTensors;

float* malloc_and_point_activations(sycl::queue &queue, ActivationTensors* acts, size_t* act_sizes) {
    ftrace();
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }
    float* acts_memory = (float*)xxxMallocCheck(num_activations * sizeof(float), queue);
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

typedef struct {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    size_t num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    float* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    size_t num_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    float* grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
} GPT2;

void gpt2_build_from_checkpoint(sycl::queue &queue, GPT2 *model, const char* checkpoint_path) {
    ftrace();

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    if (model_file == NULL) { printf("Error opening model file\n"); exit(1); }
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(1); }
    if (model_header[1] != 3) {
        printf("Bad version in model file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(1);
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    model->config.padded_vocab_size = Vp = model_header[7];
    printf("[GPT-2]\n");
    printf("max_seq_len: %zu\n", maxT);
    printf("vocab_size: %zu\n", V);
    printf("padded_vocab_size: %zu\n", Vp);
    printf("num_layers: %zu\n", L);
    printf("num_heads: %zu\n", NH);
    printf("channels: %zu\n", C);

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes,  model->config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    // read in all the parameters from file
    model->params_memory = malloc_and_point_parameters(queue, &model->params, model->param_sizes);
    float* params_memory_host = (float*)hostMallocCheck(num_parameters * sizeof(float), queue);
    freadCheck(params_memory_host, sizeof(float), num_parameters, model_file);
    auto ev_copy_params_h2d = queue.memcpy(model->params_memory, params_memory_host, num_parameters * sizeof(float));
    fcloseCheck(model_file);

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
    ev_copy_params_h2d.wait();
    sycl::free(params_memory_host, queue);
}

sycl::event gpt2_forward(sycl::queue &queue, GPT2 *model, int* inputs, int* targets, size_t B, size_t T,
                         const bool dump_timings,
                         const std::vector<sycl::event> &dependencies = {}) {
    // targets are optional and could be NULL
    ftrace();

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    // convenience parameters (size_t to help prevent int overflow)
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    for(int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) {
            assert(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    if(model->acts_memory == NULL) {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        model->act_sizes[0] = B * T * C; // encoded
        model->act_sizes[1] = L * B * T * C; // ln1
        model->act_sizes[2] = L * B * T;  // ln1_mean
        model->act_sizes[3] = L * B * T;  // ln1_rstd
        model->act_sizes[4] = L * B * T * 3*C; // qkv
        model->act_sizes[5] = L * B * T * C;  // atty
        model->act_sizes[6] = L * B * NH * T * T;  // preatt
        model->act_sizes[7] = L * B * NH * T * T;  // att
        model->act_sizes[8] = L * B * T * C; // attproj
        model->act_sizes[9] = L * B * T * C; // residual2
        model->act_sizes[10] = L * B * T * C; // ln2
        model->act_sizes[11] = L * B * T; // ln2_mean
        model->act_sizes[12] = L * B * T; // ln2_rstd
        model->act_sizes[13] = L * B * T * 4*C; // fch
        model->act_sizes[14] = L * B * T * 4*C; // fch_gelu
        model->act_sizes[15] = L * B * T * C; // fcproj
        model->act_sizes[16] = L * B * T * C; // residual3
        model->act_sizes[17] = B * T * C; // lnf
        model->act_sizes[18] = B * T; // lnf_mean
        model->act_sizes[19] = B * T; // lnf_rstd
        model->act_sizes[20] = B * T * Vp; // logits
        model->act_sizes[21] = B * T * Vp; // probs
        model->act_sizes[22] = B * T; // losses
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        printf("num_activations: %zu\n", num_activations);
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(queue, &model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        model->inputs = (int*)xxxMallocCheck(B * T * sizeof(int), queue);
        model->targets = (int*)xxxMallocCheck(B * T * sizeof(int), queue); // might be unused if we never have targets but it's small
    } else {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        if (B != model->batch_size || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (int)B, (int)T);
            exit(EXIT_FAILURE);
        }
    }

    // cache the inputs/targets
    auto ev_copy_inputs = queue.memcpy(model->inputs, inputs, B * T * sizeof(int), dependencies);

#if (TIMEPROFILE >= 2)
    queue.wait();
    struct timespec start, end;
    // Variables for tracking wall-clock timings
    double rl_ms = 0.0, cs_ms = 0.0, mm_ms = 0.0, ln_ms = 0.0;
    double rs_ms = 0.0, ec_ms = 0.0, at_ms = 0.0, gl_ms = 0.0;
#endif

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    float* residual;
#if (TIMEPROFILE >= 2)
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    auto ev_last = encoder_forward(queue, acts.encoded, model->inputs, params.wte, params.wpe, B, T, C, {ev_copy_inputs}); // encoding goes into residual[0]
#if (TIMEPROFILE >= 2)
    ev_last.wait();
    clock_gettime(CLOCK_MONOTONIC, &end);
    ec_ms += get_elapsed_ms(start, end);
#endif
    for (int l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcb = params.fcb + l * 4*C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        float* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_preatt = acts.preatt + l * B * NH * T * T;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_attproj = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        float* l_fcproj = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;

        // now do the forward pass
#if (TIMEPROFILE >= 2)
        ev_last.wait();
        clock_gettime(CLOCK_MONOTONIC, &start);
#endif
        auto ev_ln = layernorm_forward(queue, l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C, {ev_last});
#if (TIMEPROFILE >= 2)
        ev_ln.wait();
        clock_gettime(CLOCK_MONOTONIC, &end);
        ln_ms += get_elapsed_ms(start, end);
#endif
        auto ev_mm = matmul_forward(queue, l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, {ev_ln});
#if (TIMEPROFILE >= 2)
        ev_mm.wait();
        clock_gettime(CLOCK_MONOTONIC, &start);
        ln_ms += get_elapsed_ms(end, start);
#endif
        auto ev_at = attention_forward(queue, l_atty, l_preatt, l_att, l_qkv, B, T, C, NH, {ev_mm});
#if (TIMEPROFILE >= 2)
        ev_at.wait();
        clock_gettime(CLOCK_MONOTONIC, &end);
        at_ms += get_elapsed_ms(start, end);
#endif
        ev_mm = matmul_forward(queue, l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C, {ev_at});
#if (TIMEPROFILE >= 2)
        ev_mm.wait();
        clock_gettime(CLOCK_MONOTONIC, &start);
        mm_ms += get_elapsed_ms(end, start);
#endif
        auto ev_rs = residual_forward(queue, l_residual2, residual, l_attproj, B*T*C, {ev_mm});
#if (TIMEPROFILE >= 2)
        ev_rs.wait();
        clock_gettime(CLOCK_MONOTONIC, &end);
        rs_ms += get_elapsed_ms(start, end);
#endif
        ev_ln = layernorm_forward(queue, l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C, {ev_rs});
#if (TIMEPROFILE >= 2)
        ev_ln.wait();
        clock_gettime(CLOCK_MONOTONIC, &start);
        ln_ms += get_elapsed_ms(end, start);
#endif
        ev_mm = matmul_forward(queue, l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C, {ev_ln});
#if (TIMEPROFILE >= 2)
        ev_mm.wait();
        clock_gettime(CLOCK_MONOTONIC, &end);
        mm_ms += get_elapsed_ms(start, end);
#endif
        auto ev_gl = gelu_forward(queue, l_fch_gelu, l_fch, B*T*4*C, {ev_mm});
#if (TIMEPROFILE >= 2)
        ev_gl.wait();
        clock_gettime(CLOCK_MONOTONIC, &start);
        gl_ms += get_elapsed_ms(end, start);
#endif
        ev_mm = matmul_forward(queue, l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C, {ev_gl});
#if (TIMEPROFILE >= 2)
        ev_mm.wait();
        clock_gettime(CLOCK_MONOTONIC, &end);
        mm_ms += get_elapsed_ms(start, end);
#endif
        ev_rs = residual_forward(queue, l_residual3, l_residual2, l_fcproj, B*T*C, {ev_mm});
#if (TIMEPROFILE >= 2)
        ev_rs.wait();
        clock_gettime(CLOCK_MONOTONIC, &start);
        rs_ms += get_elapsed_ms(end, start);
#endif
        ev_last = ev_rs;
    }
#if (TIMEPROFILE >= 2)
    ev_last.wait();
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    auto ev_ln = layernorm_forward(queue, acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C, {ev_last});
#if (TIMEPROFILE >= 2)
    ev_ln.wait();
    clock_gettime(CLOCK_MONOTONIC, &end);
    ln_ms += get_elapsed_ms(start, end);
#endif
    auto ev_mm = matmul_forward(queue, acts.logits, acts.lnf, params.wte, NULL, B, T, C, Vp, {ev_ln});
#if (TIMEPROFILE >= 2)
    ev_mm.wait();
    clock_gettime(CLOCK_MONOTONIC, &start);
    mm_ms += get_elapsed_ms(end, start);
#endif
    ev_last = softmax_forward(queue, acts.probs, acts.logits, B, T, V, Vp, {ev_mm});
#if (TIMEPROFILE >= 2)
    ev_last.wait();
    clock_gettime(CLOCK_MONOTONIC, &end);
    cs_ms += get_elapsed_ms(start, end);
#endif

    // also forward the cross-entropy loss function if we have the targets
    if (targets != NULL) {
        //printf("model->acts.losses = %p, model->acts.probs = %p, model->targets = %p\n", model->acts.losses, model->acts.probs, model->targets);
        auto ev_copy_targets = queue.memcpy(model->targets, targets, B * T * sizeof(int), dependencies);
#if (TIMEPROFILE >= 2)
        ev_copy_targets.wait();
        clock_gettime(CLOCK_MONOTONIC, &start);
#endif
        auto ev_ce = crossentropy_forward(queue, model->acts.losses, model->acts.probs, model->targets, B, T, Vp, {ev_copy_targets, ev_last});
#if (TIMEPROFILE >= 2)
        ev_ce.wait();
        clock_gettime(CLOCK_MONOTONIC, &end);
        cs_ms += get_elapsed_ms(start, end);
#endif
        // for convenience also evaluate the mean loss
        const size_t max_workgroup_size = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
        float *mean_loss_host = (float *)hostMallocCheck(sizeof(float), queue);
#if (TIMEPROFILE >= 2)
        clock_gettime(CLOCK_MONOTONIC, &start);
#endif
        auto ev_loss = queue.submit([&](sycl::handler &cgh){
            cgh.depends_on(ev_ce);
            auto losses = model->acts.losses;
            // Single work-group kernel to allow easy reduction
            auto kernel = [=](sycl::nd_item<1> item) {
                const size_t lid = item.get_global_id();
                sycl::group gr   = item.get_group();
                float mean_loss = 0.0f;
                for (size_t i = lid; i < B * T; i += max_workgroup_size) {
                    mean_loss += losses[i];
                }
                mean_loss  = sycl::reduce_over_group(gr, mean_loss, sycl::plus<float>());
                mean_loss /= B*T;
                mean_loss_host[0] = mean_loss;
            };
            cgh.parallel_for<class ker_loss_reduction>(
                    sycl::nd_range<1>(sycl::range<1>(max_workgroup_size), sycl::range<1>(max_workgroup_size)), kernel);
        });
        ev_last = queue.memcpy(&model->mean_loss, mean_loss_host, sizeof(float), ev_loss);
        ev_last.wait(); // Must wait to free mean_loss_host
#if (TIMEPROFILE >= 2)
        clock_gettime(CLOCK_MONOTONIC, &end);
        rl_ms += get_elapsed_ms(start, end);
#endif
        sycl::free(mean_loss_host, queue);
    } else {
        // if we don't have targets, we don't have a loss
        model->mean_loss = -1.0f;
    }

#if (TIMEPROFILE >= 2)
    if (dump_timings) {
        printf("[FWD] rl_ms = %10.3f, cs_ms = %10.3f, mm_ms = %10.3f, ln_ms = %10.3f\n", rl_ms, cs_ms, mm_ms, ln_ms);
        printf("[FWD] rs_ms = %10.3f, gl_ms = %10.3f, at_ms = %10.3f, ec_ms = %10.3f\n", rs_ms, gl_ms, at_ms, ec_ms);
    }
#endif

    return ev_last;
}

sycl::event gpt2_zero_grad(sycl::queue &queue, GPT2 *model,
                           const std::vector<sycl::event> &dependencies = {}) {
    ftrace();
    std::vector<sycl::event> depends = dependencies;
    if(model->grads_memory != NULL) {
        auto ev_zero_grads_memory = queue.fill<float>(model->grads_memory, 0.0f, model->num_parameters, dependencies);
        depends.push_back(ev_zero_grads_memory);
    }
    if(model->grads_acts_memory != NULL) {
        auto ev_zero_grads_acts_memory = queue.fill<float>(model->grads_acts_memory, 0.0f, model->num_activations, dependencies);
        depends.push_back(ev_zero_grads_acts_memory);
    }
    return queue.ext_oneapi_submit_barrier(depends);
}

sycl::event gpt2_backward(sycl::queue &queue, GPT2 *model,
                          const bool dump_timings,
                          const std::vector<sycl::event> &dependencies = {}) {
    ftrace();

    // double check we forwarded previously, with targets
    if (model->mean_loss == -1.0f) {
        printf("Error: must forward with targets before backward\n");
        exit(1);
    }

#if (TIMEPROFILE >= 2)
    queue.wait();
    struct timespec start, end;
    // Variables for tracking wall-clock timings
    double il_ms = 0.0, cs_ms = 0.0, mm_ms = 0.0, ln_ms = 0.0;
    double rs_ms = 0.0, ec_ms = 0.0, at_ms = 0.0, gl_ms = 0.0;
#endif
    // lazily allocate the memory for gradients of the weights and activations, if needed
    std::vector<sycl::event> depends = dependencies;
    if (model->grads_memory == NULL) {
        model->grads_memory = malloc_and_point_parameters(queue, &model->grads, model->param_sizes);
        model->grads_acts_memory = malloc_and_point_activations(queue, &model->grads_acts, model->act_sizes);

#if (TIMEPROFILE >= 2)
        queue.wait();
        clock_gettime(CLOCK_MONOTONIC, &start);
#endif
        auto ev_zero_grad = gpt2_zero_grad(queue, model, depends);
#if (TIMEPROFILE >= 2)
        ev_zero_grad.wait();
        clock_gettime(CLOCK_MONOTONIC, &end);
        double zg_ms = get_elapsed_ms(start, end);
        printf("zg_ms = %10.3f\n", zg_ms);
#endif

        depends.push_back(ev_zero_grad);
    }

    // convenience shortcuts (and size_t to help prevent int overflow)
    size_t B = model->batch_size;
    size_t T = model->seq_len;
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    // do a training step
#if (TIMEPROFILE >= 2)
    queue.wait();
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // technically this is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    const float dloss_mean = 1.0f / (B*T);
    auto ev_init_loss = queue.fill<float>(grads_acts.losses, dloss_mean, B * T, depends);

#if (TIMEPROFILE >= 2)
    ev_init_loss.wait();
    clock_gettime(CLOCK_MONOTONIC, &end);
    il_ms += get_elapsed_ms(start, end);
#endif

    auto ev_ce_sm = crossentropy_softmax_backward(queue, grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V, Vp, {ev_init_loss});
#if (TIMEPROFILE >= 2)
    ev_ce_sm.wait();
    clock_gettime(CLOCK_MONOTONIC, &start);
    cs_ms += get_elapsed_ms(end, start);
#endif
    auto ev_mm = matmul_backward(queue, grads_acts.lnf, grads.wte, NULL, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp, {ev_ce_sm});
#if (TIMEPROFILE >= 2)
    ev_mm.wait();
    clock_gettime(CLOCK_MONOTONIC, &end);
    mm_ms += get_elapsed_ms(start, end);
#endif
    float* residual = acts.residual3 + (L-1) * B * T * C; // last layer's residual
    float* dresidual = grads_acts.residual3 + (L-1) * B * T * C; // write to last layer's residual
    auto ev_ln = layernorm_backward(queue, dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C, {ev_mm});
#if (TIMEPROFILE >= 2)
    ev_ln.wait();
    clock_gettime(CLOCK_MONOTONIC, &start);
    ln_ms += get_elapsed_ms(end, start);
#endif

    sycl::event ev_last = ev_ln;
    for (int l = L-1; l >= 0; l--) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
        dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        float* dl_ln1w = grads.ln1w + l * C;
        float* dl_ln1b = grads.ln1b + l * C;
        float* dl_qkvw = grads.qkvw + l * 3*C * C;
        float* dl_qkvb = grads.qkvb + l * 3*C;
        float* dl_attprojw = grads.attprojw + l * C * C;
        float* dl_attprojb = grads.attprojb + l * C;
        float* dl_ln2w = grads.ln2w + l * C;
        float* dl_ln2b = grads.ln2b + l * C;
        float* dl_fcw = grads.fcw + l * 4*C * C;
        float* dl_fcb = grads.fcb + l * 4*C;
        float* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        float* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // get the pointers of the gradients of the activations for this layer
        float* dl_ln1 = grads_acts.ln1 + l * B * T * C;
        float* dl_qkv = grads_acts.qkv + l * B * T * 3*C;
        float* dl_atty = grads_acts.atty + l * B * T * C;
        float* dl_preatt = grads_acts.preatt + l * B * NH * T * T;
        float* dl_att = grads_acts.att + l * B * NH * T * T;
        float* dl_attproj = grads_acts.attproj + l * B * T * C;
        float* dl_residual2 = grads_acts.residual2 + l * B * T * C;
        float* dl_ln2 = grads_acts.ln2 + l * B * T * C;
        float* dl_fch = grads_acts.fch + l * B * T * 4*C;
        float* dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4*C;
        float* dl_fcproj = grads_acts.fcproj + l * B * T * C;
        float* dl_residual3 = grads_acts.residual3 + l * B * T * C;

#if (TIMEPROFILE >= 2)
        ev_last.wait();
        clock_gettime(CLOCK_MONOTONIC, &end);
#endif
        // backprop this layer
        auto ev_rs = residual_backward(queue, dl_residual2, dl_fcproj, dl_residual3, B*T*C, {ev_last});
#if (TIMEPROFILE >= 2)
        ev_rs.wait();
        clock_gettime(CLOCK_MONOTONIC, &start);
        rs_ms += get_elapsed_ms(end, start);
#endif
        ev_mm = matmul_backward(queue, dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C, {ev_rs});
#if (TIMEPROFILE >= 2)
        ev_mm.wait();
        clock_gettime(CLOCK_MONOTONIC, &end);
        mm_ms += get_elapsed_ms(start, end);
#endif
        auto ev_gl = gelu_backward(queue, dl_fch, l_fch, dl_fch_gelu, B*T*4*C, {ev_mm});
#if (TIMEPROFILE >= 2)
        ev_gl.wait();
        clock_gettime(CLOCK_MONOTONIC, &start);
        gl_ms += get_elapsed_ms(end, start);
#endif
        ev_mm = matmul_backward(queue, dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C, {ev_gl});
#if (TIMEPROFILE >= 2)
        ev_mm.wait();
        clock_gettime(CLOCK_MONOTONIC, &end);
        mm_ms += get_elapsed_ms(start, end);
#endif
        ev_ln = layernorm_backward(queue, dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C, {ev_mm});
#if (TIMEPROFILE >= 2)
        ev_ln.wait();
        clock_gettime(CLOCK_MONOTONIC, &start);
        ln_ms += get_elapsed_ms(end, start);
#endif
        ev_rs = residual_backward(queue, dresidual, dl_attproj, dl_residual2, B*T*C, {ev_ln});
#if (TIMEPROFILE >= 2)
        ev_rs.wait();
        clock_gettime(CLOCK_MONOTONIC, &end);
        rs_ms += get_elapsed_ms(start, end);
#endif
        ev_mm = matmul_backward(queue, dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C, {ev_rs});
#if (TIMEPROFILE >= 2)
        ev_mm.wait();
        clock_gettime(CLOCK_MONOTONIC, &start);
        mm_ms += get_elapsed_ms(end, start);
#endif
        auto ev_at = attention_backward(queue, dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH, {ev_mm});
#if (TIMEPROFILE >= 2)
        ev_at.wait();
        clock_gettime(CLOCK_MONOTONIC, &end);
        at_ms += get_elapsed_ms(start, end);
#endif
        ev_mm = matmul_backward(queue, dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C, {ev_at});
#if (TIMEPROFILE >= 2)
        ev_mm.wait();
        clock_gettime(CLOCK_MONOTONIC, &start);
        mm_ms += get_elapsed_ms(end, start);
#endif
        ev_ln = layernorm_backward(queue, dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C, {ev_mm});
#if (TIMEPROFILE >= 2)
        ev_ln.wait();
        clock_gettime(CLOCK_MONOTONIC, &end);
        ln_ms += get_elapsed_ms(start, end);
#endif
        ev_last = ev_ln;
    }

#if (TIMEPROFILE >= 2)
    ev_last.wait();
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    auto ev_enc = encoder_backward(queue, grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C, {ev_last});
#if (TIMEPROFILE >= 2)
    ev_enc.wait();
    clock_gettime(CLOCK_MONOTONIC, &end);
    ec_ms += get_elapsed_ms(start, end);
#endif

#if (TIMEPROFILE >= 2)
    if (dump_timings) {
        printf("[BWD] il_ms = %10.3f, cs_ms = %10.3f, mm_ms = %10.3f, ln_ms = %10.3f\n", il_ms, cs_ms, mm_ms, ln_ms);
        printf("[BWD] rs_ms = %10.3f, gl_ms = %10.3f, at_ms = %10.3f, ec_ms = %10.3f\n", rs_ms, gl_ms, at_ms, ec_ms);
    }
#endif

    return ev_enc;
}

sycl::event gpt2_update(sycl::queue &queue, GPT2 *model,
                        const float learning_rate, const float beta1, const float beta2,
                        const float eps, const float weight_decay, const int t,
                        const std::vector<sycl::event> &dependencies = {})
{
    ftrace();
    std::vector<sycl::event> depends = dependencies;
    // lazily allocate the memory for m_memory and v_memory
    if (model->m_memory == NULL) {
        model->m_memory = (float*)xxxMallocCheck(model->num_parameters * sizeof(float), queue);
        model->v_memory = (float*)xxxMallocCheck(model->num_parameters * sizeof(float), queue);
        auto ev_fill_m_memory = queue.fill<float>(model->m_memory, 0.0f, model->num_parameters, dependencies);
        auto ev_fill_v_memory = queue.fill<float>(model->v_memory, 0.0f, model->num_parameters, dependencies);
        depends.push_back(ev_fill_m_memory);
        depends.push_back(ev_fill_v_memory);
    }

    return adamw(queue, model->params_memory, model->grads_memory, model->m_memory, model->v_memory,
                 t, model->num_parameters, learning_rate, beta1, beta2, eps, weight_decay, depends);
}

void gpt2_free(sycl::queue &queue, GPT2 *model,
               const std::vector<sycl::event> &dependencies = {}) {
    queue.ext_oneapi_submit_barrier(dependencies).wait();
    if (model->params_memory)     sycl::free(model->params_memory, queue);
    if (model->grads_memory)      sycl::free(model->grads_memory, queue);
    if (model->m_memory)          sycl::free(model->m_memory, queue);
    if (model->v_memory)          sycl::free(model->v_memory, queue);
    if (model->acts_memory)       sycl::free(model->acts_memory, queue);
    if (model->grads_acts_memory) sycl::free(model->grads_acts_memory, queue);
    if (model->inputs)            sycl::free(model->inputs, queue);
    if (model->targets)           sycl::free(model->targets, queue);
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler

unsigned int random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
// TODO: specialize for different floating point types if needed
float random_f32(uint64_t *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// main training loop
int main() {

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

    //sycl::device dev(sycl::gpu_selector_v);
    sycl::device dev;
    sycl::queue queue(dev, exception_handler);

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(queue, &model, "gpt2_124M.bin");

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    const char* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    const char* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    const char* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    const char* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    int B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
    int T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_tokens, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);
    printf("train dataset num_batches: %zu\n", train_loader.num_tokens / (B*T));
    printf("val dataset num_batches: %zu\n", val_loader.num_tokens / (B*T));
    int val_num_batches = 5;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    uint64_t rng_state = 1337;
    int* gen_tokens = (int*)hostMallocCheck(B * T * sizeof(int), queue);
    const int genT = 64; // number of steps of inference we will do

    // train
    struct timespec start, end;
    sycl::event last = sycl::event();
    // probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // probs_host is the host copy of those
    float *probs_host = (float *)hostMallocCheck(B * T * model.config.padded_vocab_size * sizeof(float), queue);
    for (int step = 0; step <= 40; step++) {

        // once in a while estimate the validation loss
        if (step % 10 == 0) {
            float val_loss = 0.0f;
            last.wait(); // Wait before resetting
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                last.wait(); // Wait before loading next batch
                dataloader_next_batch(&val_loader);
                last = gpt2_forward(queue, &model, val_loader.inputs, val_loader.targets, B, T, /* dump_timings = */ false, {last});
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            printf("val loss %f\n", val_loss);
        }

        // once in a while do model inference to print generated text
        if (step > 0 && step % 20 == 0) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            last = queue.fill<int>(gen_tokens, tokenizer.eot_token, B * T, {last});
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            for (int t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                last.wait(); // gen_tokens is "inputs" variable required to be available on host
                last = gpt2_forward(queue, &model, gen_tokens, NULL, B, T, /* dump_timings = */ false, {last});
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the Vp-dimensional vector probs[0, t-1, :]
                last = queue.memcpy(probs_host, model.acts.probs, B * T * model.config.padded_vocab_size * sizeof(float), {last});
                last.wait();
                float* probs = probs_host + (t-1) * model.config.padded_vocab_size;
                float coin = random_f32(&rng_state);
                // note we're only sampling from the first V elements, ignoring padding
                // (the probabilities in the padded region should be zero anyway)
                int next_token = sample_mult(probs, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                if (tokenizer.init_ok) {
                    const char* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } else {
                    // fall back to printing the token id
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }

        // do a training step
        last.wait(); // Wait before loading next batch
        clock_gettime(CLOCK_MONOTONIC, &start);

        dataloader_next_batch(&train_loader);

#if (TIMEPROFILE >= 1)
        last.wait();
        clock_gettime(CLOCK_MONOTONIC, &end);
        double dataloader_ms = get_elapsed_ms(start, end);
#endif

        last = gpt2_forward(queue, &model, train_loader.inputs, train_loader.targets, B, T, /* dump_timings = */ true, {last});

#if (TIMEPROFILE >= 1)
        last.wait();
        clock_gettime(CLOCK_MONOTONIC, &start);
        double gpt2_forward_ms = get_elapsed_ms(end, start);
#endif

        last = gpt2_zero_grad(queue, &model, {last});

#if (TIMEPROFILE >= 1)
        last.wait();
        clock_gettime(CLOCK_MONOTONIC, &end);
        double gpt2_zero_grad_ms = get_elapsed_ms(start, end);
#endif

        last = gpt2_backward(queue, &model, /* dump_timings = */ true, {last});

#if (TIMEPROFILE >= 1)
        last.wait();
        clock_gettime(CLOCK_MONOTONIC, &start);
        double gpt2_backward_ms = get_elapsed_ms(end, start);
#endif

        last = gpt2_update(queue, &model, 1e-4, 0.9, 0.999, 1e-8, 0.0, step+1, {last});

        last.wait();
        clock_gettime(CLOCK_MONOTONIC, &end);

#if (TIMEPROFILE >= 1)
        double gpt2_update_ms = get_elapsed_ms(start, end);
        double total_ms = dataloader_ms + gpt2_forward_ms + gpt2_zero_grad_ms + gpt2_backward_ms + gpt2_update_ms;
        printf("step %4d: train loss %.6f (took %10.3f ms: [%10.3f, %10.3f, %10.3f, %10.3f, %10.3f])\n", step,
                model.mean_loss, total_ms, dataloader_ms,
                gpt2_forward_ms, gpt2_zero_grad_ms, gpt2_backward_ms, gpt2_update_ms);
#else
        double total_ms = get_elapsed_ms(start, end);
        printf("step %4d: train loss %.6f (took %10.3f ms)\n", step, model.mean_loss, total_ms);
#endif

    }

    last.wait_and_throw();
    queue.wait_and_throw();

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(queue, &model, {last});
    if (gen_tokens) sycl::free(gen_tokens, queue);
    if (probs_host) sycl::free(probs_host, queue);
    return 0;
}
#endif
