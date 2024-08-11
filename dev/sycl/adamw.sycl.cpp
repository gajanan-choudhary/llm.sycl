/*
Kernels for AdamW optimizer.

GPU Compile example for Intel PVC GPU with ahead-of-time compilation:
icpx -O3 -fsycl -fno-sycl-id-queries-fit-in-int -std=c++17 -xhost -DLLMSYCL -fp-model=precise \
     -qopenmp -qopenmp-simd -DOMP  -DTIMEPROFILE \
     -fsycl-targets=spir64_gen -Xs "-device pvc" -D_DISABLE_SG_SIZE_8 \
     adamw.sycl.cpp -lsycl -lOpenCL -liomp5 -o adamw

version 1 is naive CPU port
./adamw 1

version 2 uses sycl::nd_range/workgroups
./adamw 2

Observations as of 08/10/2024: version 2 with SG_SIZE=32, SG_PER_WG={64, 32} are faster on CPU
                               version 2 with SG_SIZE=32, SG_PER_WG=32 is faster on GPU

*/

#include <stdio.h>
#include <stdlib.h>
#ifdef OMP
#include <omp.h>
#endif
#include "common.h"
#include <sycl/sycl.hpp>

#define SQRT sqrtf
#define POW  powf

// ----------------------------------------------------------------------------
// TFLOP/s
double get_adamw_tflops(double elapsed_ms, size_t num_parameters) {
    // Time is in milliseconds
    // Assumes sqrt is 4 flops
    return (double) (num_parameters * 19) / elapsed_ms * 1e-9;
}

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 AdamW optimizer
void adamw_cpu(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, const int t, const size_t num_parameters,
               const float learning_rate, const float beta1, const float beta2, const float eps, const float weight_decay) {
    timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    const float oneminusbeta1 = 1.0f - beta1;
    const float oneminusbeta2 = 1.0f - beta2;
    const float beta1_correction = 1.0f - POW(beta1, t);
    const float beta2_correction = 1.0f - POW(beta2, t);

    #pragma omp parallel for
    for (size_t i = 0; i < num_parameters; i++) {
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
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = get_elapsed_ms(start, end);
    double tflops = get_adamw_tflops(elapsed_ms, num_parameters);
    printf("kernel ref OMP | time %.4f ms | tflops %.2f\n", elapsed_ms, tflops);
}

// ----------------------------------------------------------------------------
// GPU kernels

// kernel 1 is the most naive residual kernel
sycl::event adamw1(sycl::queue &queue,
                   float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, const int t, const size_t num_parameters,
                   const float learning_rate, const float beta1, const float beta2, const float eps, const float weight_decay,
                   const std::vector<sycl::event> &dependencies = {}) {
    const float oneminusbeta1 = 1.0f - beta1;
    const float oneminusbeta2 = 1.0f - beta2;
    const float beta1_correction = 1.0f - POW(beta1, t);
    const float beta2_correction = 1.0f - POW(beta2, t);

    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::item<1> item) {
            const size_t i = item.get_id(0);
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
        cgh.parallel_for<class ker_adamw_1>(sycl::range<1>(num_parameters), kernel);
    });
    return last;
}

template <int SG_SIZE>
class ker_adamw_2;

template<int SG_SIZE>
sycl::event adamw2(sycl::queue &queue,
                   float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, const int t, const size_t num_parameters,
                   const float learning_rate, const float beta1, const float beta2, const float eps, const float weight_decay,
                   const int sg_per_wg, const std::vector<sycl::event> &dependencies = {}) {
    const int adjusted_sg_per_wg = std::min<size_t>((num_parameters + SG_SIZE - 1) / SG_SIZE, sg_per_wg);
    const int wg_size = adjusted_sg_per_wg * SG_SIZE;
    const size_t ceilN = ((num_parameters + wg_size - 1) & (-wg_size));

    const float oneminusbeta1 = 1.0f - beta1;
    const float oneminusbeta2 = 1.0f - beta2;
    const float beta1_correction = 1.0f - POW(beta1, t);
    const float beta2_correction = 1.0f - POW(beta2, t);

    sycl::event last = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        auto kernel = [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
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

// kernel version dispatch
sycl::event adamw(int kernel_num,
                  sycl::queue &queue,
                  float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, const int t, const size_t num_parameters,
                  const float learning_rate, const float beta1, const float beta2, const float eps, const float weight_decay,
                  const int sg_per_wg, const int sg_size) {
    switch (kernel_num) {
        case 1: {
            return adamw1(queue, params_memory, grads_memory, m_memory, v_memory, t, num_parameters,
                          learning_rate, beta1, beta2, eps, weight_decay);
        } break;
        case 2: {
            if (sg_size == 32) {
                return adamw2<32>(queue, params_memory, grads_memory, m_memory, v_memory, t, num_parameters,
                                  learning_rate, beta1, beta2, eps, weight_decay, sg_per_wg);
            }
#ifndef _DISABLE_SG_SIZE_16
            else if (sg_size == 16) {
                return adamw2<16>(queue, params_memory, grads_memory, m_memory, v_memory, t, num_parameters,
                                  learning_rate, beta1, beta2, eps, weight_decay, sg_per_wg);
            }
#endif
#ifndef _DISABLE_SG_SIZE_8
            else if (sg_size == 8) {
                return adamw2< 8>(queue, params_memory, grads_memory, m_memory, v_memory, t, num_parameters,
                                  learning_rate, beta1, beta2, eps, weight_decay, sg_per_wg);
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

    const size_t num_parameters = 1048576;
    const int t = 10;

    const float learning_rate = 1e-3f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float weight_decay = 0.0f;

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

    // create host memory of random numbers - inputs
    float* inp_params_memory = (float*)hostMallocCheck(num_parameters * sizeof(float), queue);
    float* grads_memory      = (float*)hostMallocCheck(num_parameters * sizeof(float), queue);
    float* inp_m_memory      = (float*)hostMallocCheck(num_parameters * sizeof(float), queue);
    float* inp_v_memory      = (float*)hostMallocCheck(num_parameters * sizeof(float), queue);

    make_random_float(inp_params_memory, num_parameters);
    make_random_float(grads_memory, num_parameters);
    make_random_float(inp_m_memory, num_parameters);
    make_random_float_01(inp_v_memory, num_parameters);

    // create variables for reference solution on CPU
    float* params_memory = (float*)hostMallocCheck(num_parameters * sizeof(float), queue);
    float* m_memory      = (float*)hostMallocCheck(num_parameters * sizeof(float), queue);
    float* v_memory      = (float*)hostMallocCheck(num_parameters * sizeof(float), queue);

    queue.memcpy(params_memory, inp_params_memory, num_parameters * sizeof(float));
    queue.memcpy(m_memory, inp_m_memory, num_parameters * sizeof(float));
    queue.memcpy(v_memory, inp_v_memory, num_parameters * sizeof(float));

    // move to GPU
    float* d_params_memory = (float*)deviceMallocCheck(num_parameters * sizeof(float), queue);
    float* d_grads_memory  = (float*)deviceMallocCheck(num_parameters * sizeof(float), queue);
    float* d_m_memory      = (float*)deviceMallocCheck(num_parameters * sizeof(float), queue);
    float* d_v_memory      = (float*)deviceMallocCheck(num_parameters * sizeof(float), queue);
    
    queue.memcpy(d_params_memory, inp_params_memory, num_parameters * sizeof(float));
    queue.memcpy(d_grads_memory, grads_memory, num_parameters * sizeof(float));
    queue.memcpy(d_m_memory, inp_m_memory, num_parameters * sizeof(float));
    queue.memcpy(d_v_memory, inp_v_memory, num_parameters * sizeof(float));
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
    adamw_cpu(params_memory, grads_memory, m_memory, v_memory, t, num_parameters, learning_rate, beta1, beta2, eps, weight_decay);

    // Test kernel 1
    if (kernel_num == 1) {
        printf("************************************************.\n");
        printf("Checking kernel set #%i.\n", kernel_num);
        adamw(kernel_num, queue, d_params_memory, d_grads_memory, d_m_memory, d_v_memory, t, num_parameters,
              learning_rate, beta1, beta2, eps, weight_decay, 0, 0);
        queue.wait();
        validate_result(queue, d_params_memory, params_memory, "params_memory", num_parameters, 1e-5f);
        validate_result(queue, d_m_memory, m_memory, "m_memory", num_parameters, 1e-5f);
        validate_result(queue, d_v_memory, v_memory, "v_memory", num_parameters, 1e-5f);
        printf("All results match. Benchmarking kernel %i.\n", kernel_num);
        double elapsed_ms = benchmark_kernel(repeat_times, adamw,
                                             kernel_num, queue,
                                             d_params_memory, d_grads_memory, d_m_memory, d_v_memory, t, num_parameters,
                                             learning_rate, beta1, beta2, eps, weight_decay, 0, 0);
        double tflops = get_adamw_tflops(elapsed_ms, num_parameters);
        printf("kernel %2i | time %.4f ms | tflops %.2f\n", kernel_num, elapsed_ms, tflops);

        queue.memcpy(d_params_memory, inp_params_memory, num_parameters * sizeof(float));
        queue.memcpy(d_m_memory, inp_m_memory, num_parameters * sizeof(float));
        queue.memcpy(d_v_memory, inp_v_memory, num_parameters * sizeof(float));
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
                adamw(kernel_num, queue, d_params_memory, d_grads_memory, d_m_memory, d_v_memory, t, num_parameters,
                      learning_rate, beta1, beta2, eps, weight_decay, sg_per_wg, sg_size);
                queue.wait();
                validate_result(queue, d_params_memory, params_memory, "params_memory", num_parameters, 1e-5f);
                validate_result(queue, d_m_memory, m_memory, "m_memory", num_parameters, 1e-5f);
                validate_result(queue, d_v_memory, v_memory, "v_memory", num_parameters, 1e-5f);
                printf("All results match. Benchmarking kernel %i<%i, %i>.\n", kernel_num, sg_per_wg, sg_size);
                double elapsed_ms = benchmark_kernel(repeat_times, adamw,
                                                     kernel_num, queue,
                                                     d_params_memory, d_grads_memory, d_m_memory, d_v_memory, t, num_parameters,
                                                     learning_rate, beta1, beta2, eps, weight_decay, sg_per_wg, sg_size);
                double tflops = get_adamw_tflops(elapsed_ms, num_parameters);
                printf("kernel %2i<%2i, %2i> | time %.4f ms | tflops %.2f\n", kernel_num, sg_per_wg, sg_size, elapsed_ms, tflops);

                queue.memcpy(d_params_memory, inp_params_memory, num_parameters * sizeof(float));
                queue.memcpy(d_m_memory, inp_m_memory, num_parameters * sizeof(float));
                queue.memcpy(d_v_memory, inp_v_memory, num_parameters * sizeof(float));
                queue.wait();
            }
            printf("\n");
        }
    }
    if (run_all) kernel_num++;
    printf("\n");

    // free memory
    queue.wait_and_throw();
    sycl::free(inp_params_memory, queue);
    sycl::free(inp_m_memory, queue);
    sycl::free(inp_v_memory, queue);
    sycl::free(params_memory, queue);
    sycl::free(grads_memory, queue);
    sycl::free(m_memory, queue);
    sycl::free(v_memory, queue);
    sycl::free(d_params_memory, queue);
    sycl::free(d_grads_memory, queue);
    sycl::free(d_m_memory, queue);
    sycl::free(d_v_memory, queue);
    return 0;
}
