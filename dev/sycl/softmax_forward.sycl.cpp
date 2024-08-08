/*
Kernels for softmax forward pass.

GPU Compile example for Intel PVC GPU with ahead-of-time compilation:
icpx -O3 -fsycl -fno-sycl-id-queries-fit-in-int -std=c++17 -xhost -DLLMSYCL -fp-model=precise \
     -qopenmp -qopenmp-simd -DOMP  -DTIMEPROFILE \
     -fsycl-targets=spir64_gen -Xs "-device pvc" -D_DISABLE_SG_SIZE_8 \
     softmax_forward.sycl.cpp -lsycl -lOpenCL -liomp5 -o softmax_forward

If encountering "error: identifier "M_PI" is undefined", add the following lines to the top of the file:

#define _USE_MATH_DEFINES
#include <math.h>  OR  #include <cmath>

version 1 is naive CPU port
./softmax_forward 1

version 2 is online softmax from the paper, "Online normalizer calculation for softmax"
./softmax_forward 3

version 3 and 4 are similar to vesrions 1 and 2 respectively, but use sycl::nd_range/workgroups
./softmax_forward {3, 4}

Observations as of 08/08/2024: version 3/4 all roughly same @ 142.5 ms on CPU, with SG_SIZE={32},
                               SG_PER_WG={1, 8} being marginally faster
                               version 4 with SG_SIZE=32, SG_PER_WG=32 is faster on GPU @ 15 ms.
                               Version 3 for same sg/wg size is close at 16 ms on GPU, but is up to 40%
                               slower for other SG_SIZE/SG_PER_WG values.


*/

#include <stdio.h>
#include <stdlib.h>
#ifdef OMP
#include <omp.h>
#endif
#include "common.h"
#include <sycl/sycl.hpp>

#define EXP expf /* sycl::exp  */

// ----------------------------------------------------------------------------
// TFLOP/s
double get_softmax_fwd_tflops(double elapsed_ms, size_t B, size_t T, size_t V) {
    // Time is in milliseconds
    return (double) (B * T * V) / elapsed_ms * 1e-9;
}

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 softmax forward pass
void safe_softmax_forward_cpu(float* probs, const float* logits,
                              const int B, const int T, const int V, const size_t Vp) {
    timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < B; b++) {
        for (size_t t = 0; t < T; t++) {
            // probs <- softmax(logits)
            const float* logits_bt = logits + b * T * Vp + t * Vp;
            float* probs_bt = probs + b * T * Vp + t * Vp;

            // maxval is only calculated and subtracted for numerical stability
            float maxval = -FLT_MAX;
            for (size_t i = 0; i < V; i++) {
                if (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            // note we only loop to V, leaving the padded dimensions
            for (int i = 0; i < V; i++) {
                probs_bt[i] /= sum;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            for (int i = V; i < Vp; i++) {
                probs_bt[i] = 0.0f;
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = get_elapsed_ms(start, end);
    double tflops = get_softmax_fwd_tflops(elapsed_ms, B, T, V);
    printf("kernel ref OMP | time %.4f ms | tflops %.2f\n", elapsed_ms, tflops);
}

// ----------------------------------------------------------------------------
// GPU kernels

// kernel 1 is the most naive softmax kernel
sycl::event safe_softmax_forward1(sycl::queue &queue, float* probs, const float* logits,
                                  const int B, const int T, const int V, const int Vp,
                                  const std::vector<sycl::event> &dependencies = {}) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<1> item) {
            const size_t bt = item.get_id(0);
            // probs <- softmax(logits)
            const float* logits_bt = logits + bt * Vp;
            float* probs_bt = probs + bt * Vp;

            // maxval is only calculated and subtracted for numerical stability
            float maxval = -std::numeric_limits<float>::max();
            for (int i = 0; i < V; i++) {
                if (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            float sum = float(0);
            for (int i = 0; i < V; i++) {
                probs_bt[i] = EXP(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            // note we only loop to V, leaving the padded dimensions
            for (int i = 0; i < V; i++) {
                probs_bt[i] /= sum;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            for (int i = V; i < Vp; i++) {
                probs_bt[i] = float(0);
            }
        };
        cgh.parallel_for<class ker_safe_softmax_forward_1>(sycl::range<1>(B*T), kernel);
    });
    return last;
}

// kernel 2 is like kernel 1 but rewritten as the  online softmax kernel
// from the paper "Online normalizer calculation for softmax"
sycl::event online_softmax_forward1(sycl::queue &queue, float* probs, const float* logits,
                                    const int B, const int T, const int V, const int Vp,
                                    const std::vector<sycl::event> &dependencies = {}) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<1> item) {
            const size_t bt = item.get_id(0);
            // probs <- softmax(logits)
            const float* logits_bt = logits + bt * Vp;
            float* probs_bt = probs + bt * Vp;

            // maxval is only calculated and subtracted for numerical stability
            float maxval = -std::numeric_limits<float>::max();
            float sum = float(0);
            for (int i = 0; i < V; i++) {
                if (logits_bt[i] > maxval) {
                    sum = sum * EXP(maxval - logits_bt[i]) + 1.0f;
                    maxval = logits_bt[i];
                }
                else {
                    sum += EXP(logits_bt[i] - maxval);
                }
            }
            // note we only loop to V, leaving the padded dimensions
            for (int i = 0; i < V; i++) {
                probs_bt[i] = EXP(logits_bt[i] - maxval) / sum;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            for (int i = V; i < Vp; i++) {
                probs_bt[i] = float(0);
            }
        };
        cgh.parallel_for<class ker_online_softmax_forward_1>(sycl::range<1>(B*T), kernel);
    });
    return last;
}

template <int SG_SIZE>
class ker_safe_softmax_forward_2;

template<int SG_SIZE>
sycl::event safe_softmax_forward2(sycl::queue &queue, float* probs, const float* logits,
                                  const int B, const int T, const int V, const int Vp, const int sg_per_wg,
                                  const std::vector<sycl::event> &dependencies = {}) {
    const int adjusted_sg_per_wg = std::min((V + SG_SIZE - 1) / SG_SIZE, sg_per_wg);
    const int wg_size = adjusted_sg_per_wg * SG_SIZE;

    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
            const size_t bt  = item.get_global_id(0);
            const size_t lid = item.get_local_linear_id();
            sycl::group gr = item.get_group();
            // probs <- softmax(logits)
            const float* logits_bt = logits + bt * Vp;
            float* probs_bt        = probs + bt * Vp;

            // maxval is only calculated and subtracted for numerical stability
            float maxval = -std::numeric_limits<float>::max();
            for (int i = lid; i < V; i += wg_size) {
                if (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            maxval = sycl::reduce_over_group(gr, maxval, sycl::maximum<float>());

            float sum = float(0);
            for (int i = lid; i < V; i += wg_size) {
                probs_bt[i] = EXP(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            sum = sycl::reduce_over_group(gr, sum, sycl::plus<float>());

            // note we only loop to V, leaving the padded dimensions
            for (int i = lid; i < V; i += wg_size) {
                probs_bt[i] /= sum;
            }

            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            for (int i = V + lid; i < Vp; i += wg_size) {
                probs_bt[i] = float(0);
            }
        };
        cgh.parallel_for<class ker_safe_softmax_forward_2<SG_SIZE>>(
                sycl::nd_range<2>(sycl::range<2>(B*T, wg_size), sycl::range<2>(1, wg_size)), kernel);
    });
    return last;
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
        auto kernel = [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
            const size_t bt  = item.get_global_id(0);
            const size_t lid = item.get_local_linear_id();
            sycl::group gr   = item.get_group();

            // probs <- softmax(logits)
            const float* logits_bt = logits + bt * Vp;
            float* probs_bt        = probs + bt * Vp;

            // maxval is only calculated and subtracted for numerical stability
            float local_maxval=-std::numeric_limits<float>::max();
            float local_sum = float(0);
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
                probs_bt[i] = float(0);
            }
        };
        cgh.parallel_for<class ker_online_softmax_forward_2<SG_SIZE>>(
                sycl::nd_range<2>(sycl::range<2>(B*T, wg_size), sycl::range<2>(1, wg_size)), kernel);
    });
    return last;
}

// kernel version dispatch
sycl::event softmax_forward(int kernel_num,
                            sycl::queue &queue, float* probs, const float* logits,
                            const int B, const int T, const int V, const int Vp,
                            const int sg_per_wg, const int sg_size) {
    switch (kernel_num) {
        case 1: {
            return safe_softmax_forward1(queue, probs, logits, B, T, V, Vp);
        } break;
        case 2: {
            return online_softmax_forward1(queue, probs, logits, B, T, V, Vp);
        } break;
        case 3: {
            if (sg_size == 32) {
                return safe_softmax_forward2<32>(queue, probs, logits, B, T, V, Vp, sg_per_wg);
            }
#ifndef _DISABLE_SG_SIZE_16
            else if (sg_size == 16) {
                return safe_softmax_forward2<16>(queue, probs, logits, B, T, V, Vp, sg_per_wg);
            }
#endif
#ifndef _DISABLE_SG_SIZE_8
            else if (sg_size == 8) {
                return safe_softmax_forward2< 8>(queue, probs, logits, B, T, V, Vp, sg_per_wg);
            }
#endif
            else {
                printf("Invalid sg_size\n");
                exit(2);
            }
        } break;
        case 4: {
            if (sg_size == 32) {
                return online_softmax_forward2<32>(queue, probs, logits, B, T, V, Vp, sg_per_wg);
            }
            else if (sg_size == 16) {
                return online_softmax_forward2<16>(queue, probs, logits, B, T, V, Vp, sg_per_wg);
            }
#ifndef _DISABLE_SG_SIZE_8
            else if (sg_size == 8) {
                return online_softmax_forward2< 8>(queue, probs, logits, B, T, V, Vp, sg_per_wg);
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
    float* probs  = (float*)hostMallocCheck(B * T * Vp * sizeof(float), queue);
    float* logits = (float*)hostMallocCheck(B * T * Vp * sizeof(float), queue);

    queue.fill<float>(probs, float(0), B * T * Vp);
    make_random_float(logits, B * T * Vp);
    // Reset the padding to 0 as expected
    #pragma omp parallel for collapse(3)
    for(size_t b = 0; b < B; ++b) {
        for(size_t t = 0; t < T; ++t) {
            for (size_t v = V; v < Vp; ++v) {
                logits[b * T * Vp + t * Vp + v] = 0;
            }
        }
    }

    // make the input less uniformly random: Otherwise, all probabilities will be basically zero,
    // and the tests are not actually meaningful. Say roughly 33% values are adjusted excluding
    // padded regions
    // Vfrac and the multiplier empirically determined to get some good range of
    // probabilities as output.
    const size_t Vfrac = V / 100;
    const float multiplier = 4.0f;
    int* outliers = (int*)hostMallocCheck(B * T * Vfrac * sizeof(int), queue);
    // Generate random numbers in the range [0, V)
    make_random_int(outliers, B * T * Vfrac, V);
    // Can't collape 3 since multiple elements in outliers[v] may have same
    // value meaning they point to same memory location in logits
    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            const size_t logits_base_offset   = b * T * Vp + t * Vp;
            const size_t outliers_base_offset = b * T * Vfrac + t * Vfrac;
            for (size_t v = 0; v < Vfrac; ++v) {
                logits[logits_base_offset + outliers[outliers_base_offset + v]] *= multiplier;
            }
        }
    }
    sycl::free(outliers, queue);

    // move to GPU
    float* d_probs  = (float*)deviceMallocCheck(B * T * Vp * sizeof(float), queue);
    float* d_logits = (float*)deviceMallocCheck(B * T * Vp * sizeof(float), queue);
    
    queue.fill<float>(d_probs, float(0), B * T * Vp);
    queue.memcpy(d_logits, logits, B * T * Vp * sizeof(float));
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
    safe_softmax_forward_cpu(probs, logits, B, T, V, Vp);

    // To check output probabilities
    //for(size_t b = 0; b < B; ++b) {
    //    for(size_t t = 0; t < T; ++t) {
    //        for (size_t v = 0; v < V; ++v) {
    //            const float probval = probs[b * T * Vp + t * Vp + v];
    //            if (fabsf(probval) > 1e-1) printf("probs[%zu][%zu][%zu] = %.2f\n", b, t, v, probval);
    //        }
    //    }
    //}

    // Test kernel 1, 2
    for (int i = 1; i < 3; i++) {
        if (kernel_num == i) {
            printf("************************************************.\n");
            printf("Checking kernel set #%i.\n", kernel_num);
            softmax_forward(kernel_num, queue, d_probs, d_logits, B, T, V, Vp, 0, 0);
            queue.wait();
            validate_result(queue, d_probs, probs, "probs", B * T * Vp, 1e-2f);
            printf("All results match. Benchmarking kernel %i.\n", kernel_num);
            double elapsed_ms = benchmark_kernel(repeat_times, softmax_forward,
                                                 kernel_num, queue, d_probs,
                                                 d_logits, B, T, V, Vp, 0, 0);
            double tflops = get_softmax_fwd_tflops(elapsed_ms, B, T, V);
            printf("kernel %2i | time %.4f ms | tflops %.2f\n", kernel_num, elapsed_ms, tflops);

            queue.fill<float>(d_probs, float(0), B * T * Vp);
            queue.wait();
        }
        if (run_all) kernel_num++;
        printf("\n");
    }

    // Test kernel 3, 4
    for (int i = 3; i < 5; i++) {
        if (kernel_num == i) {
            printf("************************************************.\n");
            printf("Checking kernel set #%i.\n", kernel_num);
            for (int sg_size : supported_sg_sizes) {
                if (sg_size < 8 || sg_size > 32) {printf("Skipping sg_size = %i\n", sg_size); continue;}
                printf("Testing sg_size = %i\n", sg_size);
                for (int sg_per_wg = 1; sg_per_wg <= max_workgroup_size/sg_size; sg_per_wg <<= 1) {
                    printf("Checking kernel %i<%i, %i>.\n", kernel_num, sg_per_wg, sg_size);
                    softmax_forward(kernel_num, queue, d_probs, d_logits, B, T, V, Vp, sg_per_wg, sg_size);
                    queue.wait();
                    validate_result(queue, d_probs, probs, "probs", B * T * Vp, 1e-2f);
                    printf("All results match. Benchmarking kernel %i<%i, %i>.\n", kernel_num, sg_per_wg, sg_size);
                    double elapsed_ms = benchmark_kernel(repeat_times, softmax_forward,
                                                         kernel_num, queue, d_probs,
                                                         d_logits, B, T, V, Vp, sg_per_wg, sg_size);
                    double tflops = get_softmax_fwd_tflops(elapsed_ms, B, T, V);
                    printf("kernel %2i<%2i, %2i> | time %.4f ms | tflops %.2f\n", kernel_num, sg_per_wg, sg_size, elapsed_ms, tflops);

                    queue.fill<float>(d_probs, float(0), B * T * Vp);
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
    sycl::free(probs, queue);
    sycl::free(logits, queue);
    sycl::free(d_probs, queue);
    sycl::free(d_logits, queue);
    return 0;
}
