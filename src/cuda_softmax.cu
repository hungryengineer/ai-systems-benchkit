#include "benchkit/cuda_benchmarks.hpp"
#include "benchkit/statistics.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace benchkit {

constexpr int SOFTMAX_BLOCK_SIZE = 128;

__global__ void softmax_kernel(
    const float* input,
    float* output,
    int hidden)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float shared[SOFTMAX_BLOCK_SIZE];

    // ------------------------------------------------------------
    // 1. Find row maximum
    // ------------------------------------------------------------
    float local_max = -INFINITY;

    for (int col = tid; col < hidden; col += blockDim.x) {
        int idx = row * hidden + col;
        local_max = fmaxf(local_max, input[idx]);
    }

    shared[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = fmaxf(
                shared[tid],
                shared[tid + stride]
            );
        }

        __syncthreads();
    }

    float max_val = shared[0];

    // ------------------------------------------------------------
    // 2. Compute exp(x - max)
    // ------------------------------------------------------------
    float local_sum = 0.0f;

    for (int col = tid; col < hidden; col += blockDim.x) {
        int idx = row * hidden + col;

        float value = expf(input[idx] - max_val);

        output[idx] = value;
        local_sum += value;
    }

    shared[tid] = local_sum;
    __syncthreads();

    // ------------------------------------------------------------
    // 3. Reduce sum
    // ------------------------------------------------------------
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    float sum = shared[0];

    // ------------------------------------------------------------
    // 4. Normalize
    // ------------------------------------------------------------
    for (int col = tid; col < hidden; col += blockDim.x) {
        int idx = row * hidden + col;
        output[idx] /= sum;
    }
}


// ------------------------------------------------------------
// CPU reference
// ------------------------------------------------------------

static void softmax_cpu(
    const std::vector<float>& input,
    std::vector<float>& output,
    int rows,
    int hidden)
{
    for (int row = 0; row < rows; ++row) {

        int offset = row * hidden;

        float max_val = input[offset];

        for (int col = 1; col < hidden; ++col) {
            max_val = std::max(
                max_val,
                input[offset + col]
            );
        }

        float sum = 0.0f;

        for (int col = 0; col < hidden; ++col) {
            float value = std::exp(
                input[offset + col] - max_val
            );

            output[offset + col] = value;
            sum += value;
        }

        for (int col = 0; col < hidden; ++col) {
            output[offset + col] /= sum;
        }
    }
}


// ------------------------------------------------------------
// BenchKit benchmark wrapper
// ------------------------------------------------------------

BenchmarkResult run_cuda_softmax(
    const EnvironmentInfo& environment)
{
    constexpr int rows = 2;
    constexpr int hidden = 4096;
    constexpr int warmup = 5;
    constexpr int samples = 20;

    const size_t element_count =
        static_cast<size_t>(rows) * hidden;

    const size_t bytes =
        element_count * sizeof(float);

    std::vector<float> h_input(element_count);
    std::vector<float> h_output(element_count);
    std::vector<float> h_reference(element_count);

    // Deterministic input.
    for (size_t i = 0; i < element_count; ++i) {
        h_input[i] =
            static_cast<float>((i % 100) - 50) / 10.0f;
    }

    softmax_cpu(
        h_input,
        h_reference,
        rows,
        hidden
    );

    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(
        d_input,
        h_input.data(),
        bytes,
        cudaMemcpyHostToDevice
    );

    dim3 grid(rows);
    dim3 block(SOFTMAX_BLOCK_SIZE);

    // ------------------------------------------------------------
    // Warmup
    // ------------------------------------------------------------

    for (int i = 0; i < warmup; ++i) {
        softmax_kernel<<<grid, block>>>(
            d_input,
            d_output,
            hidden
        );
    }

    cudaDeviceSynchronize();

    // ------------------------------------------------------------
    // Benchmark
    // ------------------------------------------------------------

    std::vector<double> samples_ms;
    samples_ms.reserve(samples);

    for (int i = 0; i < samples; ++i) {

        cudaEvent_t start;
        cudaEvent_t stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        softmax_kernel<<<grid, block>>>(
            d_input,
            d_output,
            hidden
        );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms = 0.0f;

        cudaEventElapsedTime(
            &elapsed_ms,
            start,
            stop
        );

        samples_ms.push_back(elapsed_ms);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // ------------------------------------------------------------
    // Copy result back
    // ------------------------------------------------------------

    cudaMemcpy(
        h_output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    );

    // ------------------------------------------------------------
    // Correctness
    // ------------------------------------------------------------

    float max_error = 0.0f;

    for (size_t i = 0; i < element_count; ++i) {
        max_error = std::max(
            max_error,
            std::fabs(
                h_output[i] - h_reference[i]
            )
        );
    }

    cudaFree(d_input);
    cudaFree(d_output);

    BenchmarkResult result;

    result.benchmark = "cuda_softmax";
    result.status =
        max_error < 1e-5f ? "pass" : "fail";
    result.environment = environment;

    result.raw_samples = samples_ms;

    result.measurements = calculate_statistics(
        samples_ms,
        "kernel"
    );

    result.measurements.push_back({
        "max_error",
        "",
        max_error
    });

    return result;
}

} // namespace benchkit