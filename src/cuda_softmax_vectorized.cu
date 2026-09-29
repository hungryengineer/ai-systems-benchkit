#include "benchkit/cuda_benchmarks.hpp"
#include "benchkit/statistics.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace benchkit {

constexpr int SOFTMAX_VECTOR_BLOCK_SIZE = 256;

__global__ void softmax_kernel_vectorized(
    const float* input,
    float* output,
    int hidden)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float shared[SOFTMAX_VECTOR_BLOCK_SIZE];

    const float4* input4 =
        reinterpret_cast<const float4*>(input + row * hidden);

    float4* output4 =
        reinterpret_cast<float4*>(output + row * hidden);

    int vectors_per_row = hidden / 4;

    // ------------------------------------------------------------
    // 1. Vectorized max pass
    // ------------------------------------------------------------
    float local_max = -INFINITY;

    for (int v = tid; v < vectors_per_row; v += blockDim.x) {
        float4 x = input4[v];

        local_max = fmaxf(local_max, x.x);
        local_max = fmaxf(local_max, x.y);
        local_max = fmaxf(local_max, x.z);
        local_max = fmaxf(local_max, x.w);
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
    // 2. Vectorized exp + store + sum
    // ------------------------------------------------------------
    float local_sum = 0.0f;

    for (int v = tid; v < vectors_per_row; v += blockDim.x) {
        float4 x = input4[v];

        float e0 = expf(x.x - max_val);
        float e1 = expf(x.y - max_val);
        float e2 = expf(x.z - max_val);
        float e3 = expf(x.w - max_val);

        output4[v] = make_float4(e0, e1, e2, e3);

        local_sum += e0 + e1 + e2 + e3;
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
    // 4. Vectorized final normalization (read-back from output)
    // ------------------------------------------------------------
    for (int v = tid; v < vectors_per_row; v += blockDim.x) {
        float4 y = output4[v];

        y.x /= sum;
        y.y /= sum;
        y.z /= sum;
        y.w /= sum;

        output4[v] = y;
    }
}

// ------------------------------------------------------------
// BenchKit benchmark wrapper
// ------------------------------------------------------------

BenchmarkResult run_cuda_softmax_vectorized(
    const EnvironmentInfo& environment)
{
    constexpr int rows = 512;
    constexpr int hidden = 4096;
    constexpr int warmup = 5;
    constexpr int samples = 20;

    const size_t element_count =
        static_cast<size_t>(rows) * hidden;

    const size_t bytes =
        element_count * sizeof(float);

    std::vector<float> h_input(element_count);
    std::vector<float> h_output(element_count);

    // Deterministic input. Matches the baseline's generator so the
    // Nsight comparison is apples-to-apples.
    for (size_t i = 0; i < element_count; ++i) {
        h_input[i] =
            static_cast<float>((i % 100) - 50) / 10.0f;
    }

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
    dim3 block(SOFTMAX_VECTOR_BLOCK_SIZE);

    // ------------------------------------------------------------
    // Warmup
    // ------------------------------------------------------------

    for (int i = 0; i < warmup; ++i) {
        softmax_kernel_vectorized<<<grid, block>>>(
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

        softmax_kernel_vectorized<<<grid, block>>>(
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

    cudaFree(d_input);
    cudaFree(d_output);

    BenchmarkResult result;

    result.benchmark = "cuda_softmax_vectorized";
    result.status = "pass";
    result.environment = environment;

    result.raw_samples = samples_ms;

    result.measurements = calculate_statistics(
        samples_ms,
        "kernel"
    );

    result.measurements.push_back({
        "hidden",
        "",
        static_cast<double>(hidden)
    });

    return result;
}

} // namespace benchkit