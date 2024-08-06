/*
Kernels for encoder forward pass.

GPU Compile example for Intel PVC GPU with ahead-of-time compilation:
icpx -O3 -fsycl -fno-sycl-id-queries-fit-in-int -std=c++17 -xhost -DLLMSYCL -fp-model=precise \
     -qopenmp -qopenmp-simd -DOMP  -DTIMEPROFILE \
     -fsycl-targets=spir64_gen -Xs "-device pvc" -D_DISABLE_SG_SIZE_8 \
     encoder_forward.sycl.cpp -lsycl -lOpenCL -liomp5 -o encoder_forward

version 1 is sycl::range<> port from CPU code to kernel: parallelizes over B*T, C, and loops over C
./encoder_forward 1

version 2 is sycl::nd_range<> port of version 1 with different SG_SIZE, sg_per_wg
./encoder_forward 2

TODO: Try oneDNN / oneMKL for version 3

Observations as of 08/05/2024: version ? with {SG_SIZE, SG_PER_WG} = {32+, 16+} are faster on CPU device
                               version 2 with {SG_SIZE, SG_PER_WG} = {8, 32} is faster on GPU device

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
double get_encoder_fwd_tflops(double elapsed_ms, size_t B, size_t T, size_t C) {
    // Time is in milliseconds
    return (double) (B * T * C) / elapsed_ms * 1e-9;
}

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 positional encoder forward pass
void encoder_forward_cpu(float* out,
                         const int* inp, const float* wte, const float* wpe,
                         int B, int T, int C) {
    timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    //#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * C + t * C;
            int ix = inp[b * T + t];
            const float* wte_ix = wte + ix * C;
            const float* wpe_t = wpe + t * C;
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = get_elapsed_ms(start, end);
    double tflops = get_encoder_fwd_tflops(elapsed_ms, B, T, C);
    printf("kernel ref SEQ | time %.4f ms | tflops %.2f\n", elapsed_ms, tflops);
}

// ----------------------------------------------------------------------------
// GPU kernels

// kernel 1 is the most naive encoder kernel
sycl::event encoder_forward1(sycl::queue &queue, float* out,
                             const int* inp, const float* wte, const float* wpe,
                             const int B, const int T, const int C,
                             const std::vector<sycl::event> &dependencies = {}) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<3> item) {
            const size_t b = item.get_id(0);
            const size_t t = item.get_id(1);
            const size_t c = item.get_id(2);
            const size_t bt = b*T+t;

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
        cgh.parallel_for<class ker_encoder_forward_1>(sycl::range<3>(B, T, C), kernel);
    });
    return last;
}

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
        auto kernel = [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
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

// kernel version dispatch
sycl::event encoder_forward(int kernel_num,
                            sycl::queue &queue, float* out,
                            const int* inp, const float* wte, const float* wpe,
                            const int B, const int T, const int C,
                            const int sg_per_wg, const int sg_size) {
    switch (kernel_num) {
        case 1: {
            return encoder_forward1(queue, out, inp, wte, wpe, B, T, C);
        } break;
        case 2: {
            if (sg_size == 32) {
                return encoder_forward2<32>(queue, out, inp, wte, wpe, B, T, C, sg_per_wg);
            }
#ifndef _DISABLE_SG_SIZE_16
            else if (sg_size == 16) {
                return encoder_forward2<16>(queue, out, inp, wte, wpe, B, T, C, sg_per_wg);
            }
#endif
#ifndef _DISABLE_SG_SIZE_8
            else if (sg_size == 8) {
                return encoder_forward2< 8>(queue, out, inp, wte, wpe, B, T, C, sg_per_wg);
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
    float* out = (float*)hostMallocCheck(B * T * C * sizeof(float), queue);
    int* inp   = (int*)hostMallocCheck(B * T * sizeof(int), queue);
    float* wte = (float*)hostMallocCheck(V * C * sizeof(float), queue);
    float* wpe = (float*)hostMallocCheck(T * C * sizeof(float), queue);

    queue.fill<float>(out, float(0), B * T * C);
    make_random_int(inp, B * T, V);
    make_random_float(wte, V * C);
    make_random_float(wpe, T * C);

    // move to GPU
    float* d_out = (float*)deviceMallocCheck(B * T * C * sizeof(float), queue);
    int* d_inp   = (int*)deviceMallocCheck(B * T * sizeof(int), queue);
    float* d_wte = (float*)deviceMallocCheck(V * C * sizeof(float), queue);
    float* d_wpe = (float*)deviceMallocCheck(T * C * sizeof(float), queue);
    
    queue.fill<float>(d_out, float(0), B * T * C);
    queue.memcpy(d_inp, inp, B * T * sizeof(int));
    queue.memcpy(d_wte, wte, V * C * sizeof(float));
    queue.memcpy(d_wpe, wpe, T * C * sizeof(float));
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
    encoder_forward_cpu(out, inp, wte, wpe, B, T, C);

    // Test kernel 1
    if (kernel_num == 1) {
        printf("************************************************.\n");
        printf("Checking kernel set #%i.\n", kernel_num);
        encoder_forward(kernel_num, queue, d_out, d_inp, d_wte, d_wpe, B, T, C, 0, 0);
        queue.wait();
        validate_result(queue, d_out, out, "out", B * T * C, 1e-5f);
        printf("All results match. Benchmarking kernel %i.\n", kernel_num);
        double elapsed_ms = benchmark_kernel(repeat_times, encoder_forward,
                                             kernel_num, queue, d_out, d_inp, d_wte, d_wpe,
                                             B, T, C, 0, 0);
        double tflops = get_encoder_fwd_tflops(elapsed_ms, B, T, C);
        printf("kernel %2i | time %.4f ms | tflops %.2f\n", kernel_num, elapsed_ms, tflops);

        queue.fill<float>(d_out, float(0), B * T * C);
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
                encoder_forward(kernel_num, queue, d_out, d_inp, d_wte, d_wpe, B, T, C, sg_per_wg, sg_size);
                queue.wait();
                validate_result(queue, d_out, out, "out", B * T * C, 1e-5f);
                printf("All results match. Benchmarking kernel %i.\n", kernel_num);
                double elapsed_ms = benchmark_kernel(repeat_times, encoder_forward,
                                                     kernel_num, queue, d_out, d_inp, d_wte, d_wpe,
                                                     B, T, C, sg_per_wg, sg_size);
                double tflops = get_encoder_fwd_tflops(elapsed_ms, B, T, C);
                printf("kernel %2i<%2i, %2i> | time %.4f ms | tflops %.2f\n", kernel_num, sg_per_wg, sg_size, elapsed_ms, tflops);

                queue.fill<float>(d_out, float(0), B * T * C);
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
    sycl::free(inp, queue);
    sycl::free(wte, queue);
    sycl::free(wpe, queue);
    sycl::free(d_out, queue);
    sycl::free(d_inp, queue);
    sycl::free(d_wte, queue);
    sycl::free(d_wpe, queue);
    return 0;
}
