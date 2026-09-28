#include "benchkit/cuda_benchmarks.hpp"
#include "benchkit/statistics.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace benchkit {

constexpr int RMSNORM_BLOCK_SIZE = 128;

__global__ void rmsnorm_kernel(
    const float* x,
    const float* gamma,
    float* output,
    int hidden,
    float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float shared[RMSNORM_BLOCK_SIZE];

    float sum = 0.0f;

    for (int col = tid; col < hidden; col += blockDim.x)
    {
        int idx = row * hidden + col;
        sum += x[idx] * x[idx];
    }

    shared[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2)
    {
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    float rms = sqrtf(
        shared[0] / hidden + eps
    );

    for (int col = tid; col < hidden; col += blockDim.x)
    {
        int idx = row * hidden + col;

        output[idx] =
            (x[idx] / rms) * gamma[col];
    }
}


BenchmarkResult run_cuda_rmsnorm(
    const EnvironmentInfo& environment)
{
    constexpr int rows = 2;
    constexpr int hidden = 4096;
    constexpr float eps = 1e-5f;

    constexpr size_t elements =
        static_cast<size_t>(rows) * hidden;

    constexpr size_t bytes =
        elements * sizeof(float);

    // Host buffers
    std::vector<float> h_x(elements);
    std::vector<float> h_gamma(hidden);
    std::vector<float> h_output(elements);

    for (size_t i = 0; i < elements; ++i)
    {
        h_x[i] = 0.001f * static_cast<float>(i + 1);
    }

    for (int i = 0; i < hidden; ++i)
    {
        h_gamma[i] = 1.0f;
    }

    // Device buffers
    float* d_x = nullptr;
    float* d_gamma = nullptr;
    float* d_output = nullptr;

    cudaMalloc(&d_x, bytes);
    cudaMalloc(
        &d_gamma,
        hidden * sizeof(float)
    );
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(
        d_x,
        h_x.data(),
        bytes,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_gamma,
        h_gamma.data(),
        hidden * sizeof(float),
        cudaMemcpyHostToDevice
    );

    constexpr int blocks = rows;

    // Warm-up
    rmsnorm_kernel<<<blocks, RMSNORM_BLOCK_SIZE>>>(
        d_x,
        d_gamma,
        d_output,
        hidden,
        eps
    );

    cudaDeviceSynchronize();

    // Timing
    constexpr int iterations = 20;

    std::vector<double> timings;
    timings.reserve(iterations);

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int iteration = 0;
         iteration < iterations;
         ++iteration)
    {
        cudaEventRecord(start);

        rmsnorm_kernel<<<blocks, RMSNORM_BLOCK_SIZE>>>(
            d_x,
            d_gamma,
            d_output,
            hidden,
            eps
        );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0.0f;

        cudaEventElapsedTime(
            &milliseconds,
            start,
            stop
        );

        timings.push_back(
            static_cast<double>(milliseconds)
        );
    }

    // Copy result back
    cudaMemcpy(
        h_output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    );

    // Basic correctness check:
    // recompute the RMSNorm reference on CPU.
    std::vector<float> h_reference(elements);

    for (int row = 0; row < rows; ++row)
    {
        float sum = 0.0f;

        for (int col = 0; col < hidden; ++col)
        {
            int idx = row * hidden + col;
            sum += h_x[idx] * h_x[idx];
        }

        float rms =
            std::sqrt(sum / hidden + eps);

        for (int col = 0; col < hidden; ++col)
        {
            int idx = row * hidden + col;

            h_reference[idx] =
                (h_x[idx] / rms) * h_gamma[col];
        }
    }

    float max_error = 0.0f;

    for (size_t i = 0; i < elements; ++i)
    {
        float error =
            std::fabs(
                h_output[i] - h_reference[i]
            );

        max_error =
            std::max(max_error, error);
    }

    bool correct = max_error < 1e-5f;

    std::cout
        << "CUDA RMSNorm: "
        << (correct ? "PASS" : "FAIL")
        << " (max error: "
        << max_error
        << ")\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_x);
    cudaFree(d_gamma);
    cudaFree(d_output);

    BenchmarkResult result;

    result.benchmark = "cuda_rmsnorm";
    result.status = correct ? "pass" : "fail";
    result.environment = environment;
    result.sample_count =
        static_cast<int>(timings.size());
    result.raw_samples = timings;

    std::vector<Measurement> stats =
        calculate_statistics(timings, "ms");

    for (Measurement& stat : stats)
    {
        stat.name = "kernel_" + stat.name;
        result.measurements.push_back(
            std::move(stat)
        );
    }

    result.measurements.push_back(
        Measurement{
            .name = "max_error",
            .unit = "",
            .value = max_error
        }
    );

    return result;
}

} // namespace benchkit