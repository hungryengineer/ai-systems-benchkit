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

CudaBenchmarkResult run_cuda_vector_add()
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

    // --- H2D (Host-to-Device) transfer latency ---

    constexpr int h2d_iterations = 20;
    std::vector<double> h2d_timings;
    h2d_timings.reserve(h2d_iterations);

    cudaEvent_t h2d_start;
    cudaEvent_t h2d_stop;

    cudaEventCreate(&h2d_start);
    cudaEventCreate(&h2d_stop);

    for (int iteration = 0; iteration < h2d_iterations; ++iteration)
    {
        // Mark the stream position before the transfer starts.
        cudaEventRecord(h2d_start);

        // Copy both input buffers (a and b) to the device as one 2-MB
        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

        // Mark the stream position right after the copies finish. Because
        // cudaMemcpy is synchronous, the host blocks until the data arrives.
        cudaEventRecord(h2d_stop);

        // Wait until the GPU reaches the stop event, guaranteeing the copy
        cudaEventSynchronize(h2d_stop);

        float milliseconds = 0.0f;

        // Time between the two stream milestones = H2D transfer latency.
        cudaEventElapsedTime(
            &milliseconds,
            h2d_start,
            h2d_stop
        );

        h2d_timings.push_back(static_cast<double>(milliseconds));
    }

    cudaEventDestroy(h2d_start);
    cudaEventDestroy(h2d_stop);

    // --- D2H (Device-to-Host) transfer latency ---
    // Same technique as H2D: blocking copies surrounded by CUDA events.
    // d_c already holds the kernel result from the timed loop above, so
    // copying it back repeatedly is idempotent and measures the real
    // device->host path latency.
    constexpr int d2h_iterations = 20;
    std::vector<double> d2h_timings;
    d2h_timings.reserve(d2h_iterations);

    cudaEvent_t d2h_start;
    cudaEvent_t d2h_stop;

    cudaEventCreate(&d2h_start);
    cudaEventCreate(&d2h_stop);

    for (int iteration = 0; iteration < d2h_iterations; ++iteration)
    {
        // Mark the stream position before the transfer starts.
        cudaEventRecord(d2h_start);

        // Copy the result buffer back to the host. cudaMemcpy is blocking,
        // so the host waits until the full buffer has arrived.
        cudaMemcpy(
            h_c,
            d_c,
            bytes,
            cudaMemcpyDeviceToHost
        );

        // Mark the stream position after the copy completes.
        cudaEventRecord(d2h_stop);

        // Block the host until the GPU reaches the stop event, so the
        // elapsed time below reflects a finished transfer.
        cudaEventSynchronize(d2h_stop);

        float milliseconds = 0.0f;

        // Time between the two stream milestones = D2H transfer latency.
        cudaEventElapsedTime(
            &milliseconds,
            d2h_start,
            d2h_stop
        );

        d2h_timings.push_back(static_cast<double>(milliseconds));
    }

    cudaEventDestroy(d2h_start);
    cudaEventDestroy(d2h_stop);

    // --- End-to-end (e2e) latency ---
    // The full round trip of one benchmark iteration: copy the inputs to the
    // device, launch the kernel, copy the result back. Because the H2D/D2H
    // copies are blocking, the elapsed time covers the complete pipeline the
    // application would observe (transfer + compute + transfer).
    constexpr int e2e_iterations = 20;
    std::vector<double> e2e_timings;
    e2e_timings.reserve(e2e_iterations);

    cudaEvent_t e2e_start;
    cudaEvent_t e2e_stop;

    cudaEventCreate(&e2e_start);
    cudaEventCreate(&e2e_stop);

    for (int iteration = 0; iteration < e2e_iterations; ++iteration)
    {
        // Mark the stream position before the full pipeline starts.
        cudaEventRecord(e2e_start);

        // Phase 1: host -> device (inputs).
        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

        // Phase 2: GPU computation.
        vector_add_kernel<<<blocks, threads>>>(
            d_a,
            d_b,
            d_c,
            N
        );

        // Phase 3: device -> host (result). Blocking, so once it returns the
        // entire iteration has completed on the GPU.
        cudaMemcpy(
            h_c,
            d_c,
            bytes,
            cudaMemcpyDeviceToHost
        );

        cudaEventRecord(e2e_stop);

        cudaEventSynchronize(e2e_stop);

        float milliseconds = 0.0f;

        // Time between start and stop = full H2D + kernel + D2H latency.
        cudaEventElapsedTime(
            &milliseconds,
            e2e_start,
            e2e_stop
        );

        e2e_timings.push_back(static_cast<double>(milliseconds));
    }

    cudaEventDestroy(e2e_start);
    cudaEventDestroy(e2e_stop);

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

    return CudaBenchmarkResult{
        timings,
        h2d_timings,
        d2h_timings,
        e2e_timings
    };
}

} // namespace benchkit