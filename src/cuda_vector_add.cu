#include "benchkit/cuda_benchmarks.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

namespace benchkit {

__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

std::vector<double> run_cuda_vector_add()
{
    constexpr int N = 1 << 20;
    constexpr size_t bytes = N * sizeof(float);
    

    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];

    for (int i = 0; i < N; ++i)
    {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(
        d_a,
        h_a,
        bytes,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_b,
        h_b,
        bytes,
        cudaMemcpyHostToDevice
    );

    constexpr int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    // Warm-up
    vector_add_kernel<<<blocks, threads>>>(
        d_a,
        d_b,
        d_c,
        N
    );

    cudaDeviceSynchronize();

    // CUDA timing events
    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    constexpr int iterations = 20;
    std::vector<double> timings;
    timings.reserve(iterations);

    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        cudaEventRecord(start);

        vector_add_kernel<<<blocks, threads>>>(
            d_a,
            d_b,
            d_c,
            N
        );

        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float milliseconds = 0.0f;

        cudaEventElapsedTime(
            &milliseconds,
            start,
            stop
        );

        // double microseconds = milliseconds * 1000.0;

        timings.push_back(static_cast<double>(milliseconds));
    }

    cudaMemcpy(
        h_c,
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    );

    bool correct = true;

    for (int i = 0; i < N; ++i)
    {
        if (h_c[i] != 3.0f)
        {
            correct = false;
            break;
        }
    }

    std::cout
        << "CUDA Vector Add: "
        << (correct ? "PASS" : "FAIL")
        << '\n';

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return timings;
}

} // namespace benchkit