#include "benchkit/cuda_benchmarks.hpp"
#include "benchkit/statistics.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace benchkit {

constexpr int SOFTMAX_VECTOR_BLOCK_SIZE = 256;

__device__ float warp_reduce_sum(float value)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(
            0xffffffff,
            value,
            offset
        );
    }

    return value;
}

__device__ float warp_reduce_max(float value)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        value = fmaxf(
            value,
            __shfl_down_sync(
                0xffffffff,
                value,
                offset
            )
        );
    }

    return value;
}

__device__ float block_reduce_sum(float value)
{
    __shared__ float warp_sums[8];
    __shared__ float total;

    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    value = warp_reduce_sum(value);

    if (lane == 0) {
        warp_sums[warp] = value;
    }

    __syncthreads();

    value = (threadIdx.x < 8)
        ? warp_sums[lane]
        : 0.0f;

    if (warp == 0) {
        value = warp_reduce_sum(value);
    }

    if (threadIdx.x == 0) {
        total = value;
    }

    __syncthreads();

    return total;
}

__device__ float block_reduce_max(float value)
{
    __shared__ float warp_max[8];
    __shared__ float total_max;

    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    value = warp_reduce_max(value);

    if (lane == 0) {
        warp_max[warp] = value;
    }

    __syncthreads();

    value = (threadIdx.x < 8)
        ? warp_max[lane]
        : -INFINITY;

    if (warp == 0) {
        value = warp_reduce_max(value);
    }

    if (threadIdx.x == 0) {
        total_max = value;
    }

    __syncthreads();

    return total_max;
}

__global__ void softmax_kernel_vectorized(
    const float* input,
    float* output,
    int hidden)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

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

    float max_val = block_reduce_max(local_max);

    // ------------------------------------------------------------
    // 2. Fused exp + sum, results cached in registers (V4)
    //    exp -> registers -> reduce -> normalize from registers.
    //    No global write/read-back of the exp values.
    // ------------------------------------------------------------
    float4 cached[4];
    float local_sum = 0.0f;

    for (int k = 0; k < 4; ++k) {
        int v = tid + k * blockDim.x;

        if (v >= vectors_per_row) {
            break;
        }

        float4 x = input4[v];

        float4 e = make_float4(
            expf(x.x - max_val),
            expf(x.y - max_val),
            expf(x.z - max_val),
            expf(x.w - max_val)
        );

        cached[k] = e;

        local_sum += e.x + e.y + e.z + e.w;
    }

    float sum = block_reduce_sum(local_sum);

    // ------------------------------------------------------------
    // 3. Normalize straight from registers
    // ------------------------------------------------------------
    for (int k = 0; k < 4; ++k) {
        int v = tid + k * blockDim.x;

        if (v >= vectors_per_row) {
            break;
        }

        float4 y = cached[k];

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