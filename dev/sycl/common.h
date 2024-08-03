#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <sycl/sycl.hpp>

template<class T>
T ceil_div(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

inline double get_elapsed_ms(struct timespec &start, struct timespec &end) {
    return 1e3*((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9); /* milli-seconds */
}

// ----------------------------------------------------------------------------
// sycl::malloc_host/malloc_shared/malloc_device error-handling wrapper util

#ifdef LLMSYCL
extern inline void *sycl_malloc_check(size_t size, sycl::queue &queue, sycl::usm::alloc alloc_type, const char *file, int line) {
    void *ptr;
    if (alloc_type == sycl::usm::alloc::device)  ptr = malloc_device(size, queue);
    else if (alloc_type == sycl::usm::alloc::shared)  ptr = malloc_shared(size, queue);
    else if (alloc_type == sycl::usm::alloc::host)    ptr = malloc_host(size, queue);
    else {
        fprintf(stderr, "  Unexpected SYCL memory allocation type requested\n");
        exit(EXIT_FAILURE);
    }
    if (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

#define deviceMallocCheck(size, queue) sycl_malloc_check(size, queue, sycl::usm::alloc::device, __FILE__, __LINE__)
#define sharedMallocCheck(size, queue) sycl_malloc_check(size, queue, sycl::usm::alloc::shared, __FILE__, __LINE__)
#define hostMallocCheck(size, queue)   sycl_malloc_check(size, queue, sycl::usm::alloc::host, __FILE__, __LINE__)
#endif // #ifdef LLMSYCL

// ----------------------------------------------------------------------------
// random utils

void make_random_float_01(float *arr, size_t N) {
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / (float)RAND_MAX); // range 0..1
    }
}

void make_random_float(float *arr, size_t N) {
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / (float)RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
}

void make_random_int(int *arr, size_t N, int V) {
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
}

void make_zeros_float(float *arr, size_t N) {
    memset(arr, 0, N * sizeof(float)); // all zero
}

void make_ones_float(float *arr, size_t N) {
    for (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
}

template<class D, class T>
void validate_result(sycl::queue &queue, D* device_result, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)hostMallocCheck(num_elements * sizeof(D), queue);
    queue.memcpy(out_gpu, device_result, num_elements * sizeof(D)).wait();
    int nfaults = 0;
#ifndef ENABLE_BF16
    float epsilon = FLT_EPSILON;
#else
    float epsilon = 0.079;
#endif
    for (int i = 0; i < num_elements; i++) {
        // Skip masked elements
        if(!isfinite(cpu_reference[i]))
            continue;

        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        float t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        if (fabs(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs device: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            if (nfaults >= 10) {
                sycl::free(out_gpu, queue);
                exit(EXIT_FAILURE);
            }
        }
    }

    if (nfaults > 0) {
        sycl::free(out_gpu, queue);
        exit(EXIT_FAILURE);
    }

    sycl::free(out_gpu, queue);
}

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, int kernel_num, sycl::queue &queue, KernelArgs&&... kernel_args) {
    timespec start, end;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    const size_t dev_cache_size = queue.get_device().get_info<sycl::info::device::global_mem_cache_size>();
    const size_t flush_buffer_bytes = 4 * dev_cache_size;
    void* flush_buffer = deviceMallocCheck(flush_buffer_bytes, queue);

    // Single warmup run, untimed
    auto last = kernel(kernel_num, queue, std::forward<KernelArgs>(kernel_args)...);
    last.wait();

    double elapsed_time_ms = 0.0;
    for (int i = 0; i < repeats; i++) {
        // clear L2
        queue.memset(flush_buffer, 0, flush_buffer_bytes).wait();
        // now we can start recording the timing of the kernel
        clock_gettime(CLOCK_MONOTONIC, &start);
        last = kernel(kernel_num, queue, std::forward<KernelArgs>(kernel_args)...);
        last.wait();
        clock_gettime(CLOCK_MONOTONIC, &end);

        double single_ms = get_elapsed_ms(start, end);
        elapsed_time_ms += single_ms;
    }

    sycl::free(flush_buffer, queue);

    return elapsed_time_ms / repeats;
}
