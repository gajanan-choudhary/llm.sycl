/*
Kernels for encoder backward pass.

GPU Compile example for Intel PVC GPU with ahead-of-time compilation:
icpx -O3 -fsycl -fno-sycl-id-queries-fit-in-int -std=c++17 -xhost -DLLMSYCL -fp-model=precise \
     -qopenmp -qopenmp-simd -DOMP  -DTIMEPROFILE \
     -fsycl-targets=spir64_gen -Xs "-device pvc" -D_DISABLE_SG_SIZE_8 \
     encoder_backward.sycl.cpp -lsycl -lOpenCL -liomp5 -o encoder_backward

version 1 is sycl::range<> port from CPU code to kernel: parallelizes over B*T, C, and loops over C
./encoder_backward 1

version 2 simply splits the kernel 1 into 2 kernels to parallelize differently
./encoder_backward 2

version 3 is sycl::nd_range<> port of version 2 with different SG_SIZE, sg_per_wg and uses atomics
./encoder_backward 3

TODO: Try oneDNN / oneMKL for version 4

Observations as of 08/05/2024: version 1 with SG_SIZE={16}, SG_PER_WG = {1, 2, 4} are faster on CPU device
                               version 2 with {SG_SIZE, SG_PER_WG} = {2, 32} is faster on GPU device; note
                               that even though B = 64 is perfect for sg_size/sg_per_wg, even with B = 157
                               the perf for that is best

*/

#include <stdio.h>
#include <stdlib.h>
#ifdef OMP
#include <omp.h>
#endif
#include "common.h"
#include <sycl/sycl.hpp>

// ----------------------------------------------------------------------------
// TFLOP/s
double get_encoder_bwd_tflops(double elapsed_ms, size_t B, size_t T, size_t C) {
    // Time is in milliseconds
    return (double) (B * T * C * 2) / elapsed_ms * 1e-9;
}

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 positional encoder backward pass
void encoder_backward_cpu(float* dwte, float* dwpe,
                          const float* dout, const int* inp,
                          const int B, const int T, const int C) {
    timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* dout_bt = dout + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* dwte_ix = dwte + ix * C;
            float* dwpe_t = dwpe + t * C;
            for (int i = 0; i < C; i++) {
                const float d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = get_elapsed_ms(start, end);
    double tflops = get_encoder_bwd_tflops(elapsed_ms, B, T, C);
    printf("kernel ref SEQ | time %.4f ms | tflops %.2f\n", elapsed_ms, tflops);
}

// ----------------------------------------------------------------------------
// GPU kernels

// kernel 1 is the most naive encoder kernel
sycl::event encoder_backward1(sycl::queue &queue, float* dwte, float* dwpe,
                              const float* dout, const int* inp,
                              const int B, const int T, const int C,
                              const std::vector<sycl::event> &dependencies = {}) {
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<1> item) {
            const size_t c = item.get_id(0);
            for (int b = 0; b < B; b++) {
                for (int t = 0; t < T; t++) {
                    const size_t bt = b * T + t;
                    const int ix = inp[bt];
                    float* dwte_ix = dwte + ix * C;
                    float* dwpe_t = dwpe + t * C;
                    const float* dout_bt = dout + bt * C;
                    const float d = dout_bt[c];
                    dwte_ix[c] += d;
                    dwpe_t[c] += d;
                }
            }
        };
        cgh.parallel_for<class ker_encoder_backward_1>(sycl::range<1>(C), kernel);
    });
    return last;
}

sycl::event encoder_backward2(sycl::queue &queue, float* dwte, float* dwpe,
                              const float* dout, const int* inp,
                              const int B, const int T, const int C,
                              const std::vector<sycl::event> &dependencies = {}) {
    const size_t TC = T * C;
    sycl::event ev_dwpe = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<1> item) {
            const size_t tc = item.get_id(0);
            const float *dout_tc = dout + tc;
            float val = float(0);
            for (int b = 0; b < B; b++) {
                val += dout_tc[b * TC];
            }
            dwpe[tc] += val;
        };
        cgh.parallel_for<class ker_encoder_backward_2_dwpe>(sycl::range<1>(TC), kernel);
    });

    sycl::event ev_dwte = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<1> item) {
            const size_t c = item.get_id(0);
            for (int b = 0; b < B; b++) {
                for (int t = 0; t < T; t++) {
                    const size_t bt = b * T + t;
                    const int ix = inp[bt];
                    float* dwte_ix = dwte + ix * C;
                    const float* dout_bt = dout + bt * C;
                    const float d = dout_bt[c];
                    dwte_ix[c] += d;
                }
            }
        };
        cgh.parallel_for<class ker_encoder_backward_2_dwte>(sycl::range<1>(C), kernel);
    });
    return queue.ext_oneapi_submit_barrier({ev_dwpe, ev_dwte});
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
        auto kernel = [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
            const size_t t   = item.get_global_id(0);
            const size_t c   = item.get_global_id(1);
            const size_t tc  = t * C + c;
            const size_t lid = item.get_local_id(2);
            sycl::group gr   = item.get_group();

            const float *dout_tc = dout + tc;
            float val = float(0);
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

// kernel version dispatch
sycl::event encoder_backward(int kernel_num,
                             sycl::queue &queue, float* dwte, float* dwpe,
                             const float* dout, const int* inp,
                             const int B, const int T, const int C,
                             const int sg_per_wg, const int sg_size) {
    switch (kernel_num) {
        case 1: {
            return encoder_backward1(queue, dwte, dwpe, dout, inp, B, T, C);
        } break;
        case 2: {
            return encoder_backward2(queue, dwte, dwpe, dout, inp, B, T, C);
        } break;
        case 3: {
            if (sg_size == 32) {
                return encoder_backward3<32>(queue, dwte, dwpe, dout, inp, B, T, C, sg_per_wg);
            }
#ifndef _DISABLE_SG_SIZE_16
            else if (sg_size == 16) {
                return encoder_backward3<16>(queue, dwte, dwpe, dout, inp, B, T, C, sg_per_wg);
            }
#endif
#ifndef _DISABLE_SG_SIZE_8
            else if (sg_size == 8) {
                return encoder_backward3< 8>(queue, dwte, dwpe, dout, inp, B, T, C, sg_per_wg);
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

    int B = 64;
    int T = 1024;
    int C = 768;
    int V = 50257;

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
    float* dout = (float*)hostMallocCheck(B * T * C * sizeof(float), queue);
    int* inp    = (int*)hostMallocCheck(B * T * sizeof(int), queue);
    float* dwte = (float*)hostMallocCheck(V * C * sizeof(float), queue);
    float* dwpe = (float*)hostMallocCheck(T * C * sizeof(float), queue);

    make_random_float(dout, B * T * C);
    make_random_int(inp, B * T, V);
    queue.fill<float>(dwte, float(0), V * C);
    queue.fill<float>(dwpe, float(0), T * C);

    // move to GPU
    float* d_dout = (float*)deviceMallocCheck(B * T * C * sizeof(float), queue);
    int* d_inp    = (int*)deviceMallocCheck(B * T * sizeof(int), queue);
    float* d_dwte = (float*)deviceMallocCheck(V * C * sizeof(float), queue);
    float* d_dwpe = (float*)deviceMallocCheck(T * C * sizeof(float), queue);
    
    queue.memcpy(d_dout, dout, B * T * C * sizeof(float));
    queue.memcpy(d_inp, inp, B * T * sizeof(int));
    queue.fill<float>(d_dwte, float(0), V * C);
    queue.fill<float>(d_dwpe, float(0), T * C);
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
    encoder_backward_cpu(dwte, dwpe, dout, inp, B, T, C);

    // Test kernels 1, 2
    for (int i = 1; i <= 2; i++) {
        if (kernel_num == i) {
            printf("************************************************.\n");
            printf("Checking kernel set #%i.\n", kernel_num);
            encoder_backward(kernel_num, queue, d_dwte, d_dwpe, d_dout, d_inp, B, T, C, 0, 0);
            queue.wait();
            validate_result(queue, d_dwte, dwte, "dwte", V * C, 1e-5f);
            validate_result(queue, d_dwpe, dwpe, "dwpe", T * C, 1e-5f);
            printf("All results match. Benchmarking kernel %i.\n", kernel_num);
            double elapsed_ms = benchmark_kernel(repeat_times, encoder_backward,
                                                 kernel_num, queue, d_dwte, d_dwpe, d_dout, d_inp,
                                                 B, T, C, 0, 0);
            double tflops = get_encoder_bwd_tflops(elapsed_ms, B, T, C);
            printf("kernel %2i | time %.4f ms | tflops %.2f\n", kernel_num, elapsed_ms, tflops);

            queue.fill<float>(d_dwte, float(0), V * C);
            queue.fill<float>(d_dwpe, float(0), T * C);
            queue.wait();
        }
        if (run_all) kernel_num++;
        printf("\n");
    }

    // Test kernel 2
    if (kernel_num == 3) {
        printf("************************************************.\n");
        printf("Checking kernel set #%i.\n", kernel_num);
        for (int sg_size : supported_sg_sizes) {
            if (sg_size < 8 || sg_size > 32) {printf("Skipping sg_size = %i\n", sg_size); continue;}
            printf("Testing sg_size = %i\n", sg_size);
            for (int sg_per_wg = 1; sg_per_wg <= max_workgroup_size/sg_size; sg_per_wg <<= 1) {
                printf("Checking kernel %i<%i, %i>.\n", kernel_num, sg_per_wg, sg_size);
                encoder_backward(kernel_num, queue, d_dwte, d_dwpe, d_dout, d_inp, B, T, C, sg_per_wg, sg_size);
                queue.wait();
                validate_result(queue, d_dwte, dwte, "dwte", V * C, 1e-5f);
                validate_result(queue, d_dwpe, dwpe, "dwpe", T * C, 1e-5f);
                printf("All results match. Benchmarking kernel %i.\n", kernel_num);
                double elapsed_ms = benchmark_kernel(repeat_times, encoder_backward,
                                                     kernel_num, queue, d_dwte, d_dwpe, d_dout, d_inp,
                                                     B, T, C, sg_per_wg, sg_size);
                double tflops = get_encoder_bwd_tflops(elapsed_ms, B, T, C);
                printf("kernel %2i<%2i, %2i> | time %.4f ms | tflops %.2f\n", kernel_num, sg_per_wg, sg_size, elapsed_ms, tflops);

                queue.fill<float>(d_dwte, float(0), V * C);
                queue.fill<float>(d_dwpe, float(0), T * C);
                queue.wait();
            }
            printf("\n");
        }
    }
    if (run_all) kernel_num++;
    printf("\n");

    // free memory
    queue.wait_and_throw();
    sycl::free(dout, queue);
    sycl::free(inp, queue);
    sycl::free(dwte, queue);
    sycl::free(dwpe, queue);
    sycl::free(d_dout, queue);
    sycl::free(d_inp, queue);
    sycl::free(d_dwte, queue);
    sycl::free(d_dwpe, queue);
    return 0;
}
