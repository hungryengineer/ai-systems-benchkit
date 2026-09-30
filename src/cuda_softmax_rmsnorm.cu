#include "benchkit/cuda_benchmarks.hpp"
#include "benchkit/statistics.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace benchkit {

constexpr int SOFTMAX_RMSNORM_BLOCK_SIZE = 256;
constexpr int SOFTMAX_RMSNORM_ROWS = 512;
constexpr int SOFTMAX_RMSNORM_HIDDEN = 4096;
constexpr float SOFTMAX_RMSNORM_EPS = 1e-5f;

// Existing kernels reused by the unfused (two-launch) baseline.
__global__ void softmax_kernel_vectorized(
    const float* input,
    float* output,
    int hidden);

__global__ void rmsnorm_kernel(
    const float* x,
    const float* gamma,
    float* output,
    int hidden,
    float eps);

// ------------------------------------------------------------
// Local reduction helpers (TU-local copies; no RDC in this build,
// but kept static so linkage is unambiguous).
// ------------------------------------------------------------

__device__ float sr_warp_reduce_sum(float value)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }

    return value;
}

__device__ float sr_warp_reduce_max(float value)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }

    return value;
}

__device__ float sr_block_reduce_sum(float value)
{
    __shared__ float warp_sums[8];
    __shared__ float total;

    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    value = sr_warp_reduce_sum(value);

    if (lane == 0) {
        warp_sums[warp] = value;
    }

    __syncthreads();

    value = (threadIdx.x < 8) ? warp_sums[lane] : 0.0f;

    if (warp == 0) {
        value = sr_warp_reduce_sum(value);
    }

    if (threadIdx.x == 0) {
        total = value;
    }

    __syncthreads();

    return total;
}

__device__ float sr_block_reduce_max(float value)
{
    __shared__ float warp_max[8];
    __shared__ float total_max;

    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    value = sr_warp_reduce_max(value);

    if (lane == 0) {
        warp_max[warp] = value;
    }

    __syncthreads();

    value = (threadIdx.x < 8) ? warp_max[lane] : -INFINITY;

    if (warp == 0) {
        value = sr_warp_reduce_max(value);
    }

    if (threadIdx.x == 0) {
        total_max = value;
    }

    __syncthreads();

    return total_max;
}

// ------------------------------------------------------------
// Fused softmax + RMSNorm (Day 4).
//
// out[row][col] = softmax(x)_i / rms(softmax) * gamma[col]
//
// One block per row. Registers hold the exp values between the
// reduction and the normalize pass (the V4 pattern), so the
// intermediate softmax never round-trips through global memory.
//
// Math:  softmax_i = e_i / sum,  e_i = exp(x_i - max)
//        mean(softmax^2) = sum(e_i^2) / (hidden * sum^2)
//        rms = sqrt(mean + eps)
//        out_i = e_i * gamma_i / (sum * rms)
// ------------------------------------------------------------
__global__ void softmax_rmsnorm_kernel(
    const float* input,
    const float* gamma,
    float* output,
    int hidden,
    float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float4* input4 =
        reinterpret_cast<const float4*>(input + row * hidden);

    const float4* gamma4 =
        reinterpret_cast<const float4*>(gamma);

    float4* output4 =
        reinterpret_cast<float4*>(output + row * hidden);

    int vectors_per_row = hidden / 4;

    // Pass 1: row max (uses register-cached values? no - max only)
    float local_max = -INFINITY;

    for (int v = tid; v < vectors_per_row; v += blockDim.x) {
        float4 x = input4[v];

        local_max = fmaxf(local_max, x.x);
        local_max = fmaxf(local_max, x.y);
        local_max = fmaxf(local_max, x.z);
        local_max = fmaxf(local_max, x.w);
    }

    float max_val = sr_block_reduce_max(local_max);

    // Pass 2: exp + sum + sum-of-squares, values cached in registers
    float4 cached[4];
    float local_sum = 0.0f;
    float local_sum2 = 0.0f;

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
        local_sum2 += e.x * e.x + e.y * e.y + e.z * e.z + e.w * e.w;
    }

    float sum = sr_block_reduce_sum(local_sum);
    float sum2 = sr_block_reduce_sum(local_sum2);

    float rms = sqrtf(sum2 / (static_cast<float>(hidden) * sum * sum) + eps);
    float factor = 1.0f / (sum * rms);

    // Pass 3: normalize straight from registers, scale by gamma
    for (int k = 0; k < 4; ++k) {
        int v = tid + k * blockDim.x;

        if (v >= vectors_per_row) {
            break;
        }

        float4 e = cached[k];
        float4 g = gamma4[v];

        output4[v] = make_float4(
            e.x * factor * g.x,
            e.y * factor * g.y,
            e.z * factor * g.z,
            e.w * factor * g.w
        );
    }
}

BenchmarkResult run_cuda_softmax_rmsnorm_impl(
    const EnvironmentInfo& environment,
    bool fused)
{
    constexpr int rows = SOFTMAX_RMSNORM_ROWS;
    constexpr int hidden = SOFTMAX_RMSNORM_HIDDEN;
    constexpr float eps = SOFTMAX_RMSNORM_EPS;
    constexpr int warmup = 5;
    constexpr int samples = 20;

    const size_t element_count =
        static_cast<size_t>(rows) * hidden;

    const size_t bytes =
        element_count * sizeof(float);

    const size_t gamma_bytes =
        static_cast<size_t>(hidden) * sizeof(float);

    std::vector<float> h_input(element_count);
    std::vector<float> h_gamma(hidden);

    // Same deterministic generator as the softmax benchmarks.
    for (size_t i = 0; i < element_count; ++i) {
        h_input[i] =
            static_cast<float>((i % 100) - 50) / 10.0f;
    }

    for (int i = 0; i < hidden; ++i) {
        h_gamma[i] = 1.0f + 0.01f * static_cast<float>(i % 7);
    }

    float* d_input = nullptr;
    float* d_gamma = nullptr;
    float* d_tmp = nullptr;
    float* d_output = nullptr;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_gamma, gamma_bytes);
    cudaMalloc(&d_tmp, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(
        d_input,
        h_input.data(),
        bytes,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_gamma,
        h_gamma.data(),
        gamma_bytes,
        cudaMemcpyHostToDevice
    );

    dim3 grid(rows);
    dim3 block_softmax(SOFTMAX_RMSNORM_BLOCK_SIZE);
    dim3 block_rmsnorm(128);

    auto launch = [&](float* out) {
        if (fused) {
            softmax_rmsnorm_kernel<<<grid, block_softmax>>>(
                d_input,
                d_gamma,
                out,
                hidden,
                eps
            );
        } else {
            softmax_kernel_vectorized<<<grid, block_softmax>>>(
                d_input,
                d_tmp,
                hidden
            );

            rmsnorm_kernel<<<grid, block_rmsnorm>>>(
                d_tmp,
                d_gamma,
                out,
                hidden,
                eps
            );
        }
    };

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        launch(d_output);
    }

    cudaDeviceSynchronize();

    // Benchmark
    std::vector<double> samples_ms;
    samples_ms.reserve(samples);

    for (int i = 0; i < samples; ++i) {
        cudaEvent_t start;
        cudaEvent_t stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        launch(d_output);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms = 0.0f;

        cudaEventElapsedTime(&elapsed_ms, start, stop);

        samples_ms.push_back(elapsed_ms);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_tmp);
    cudaFree(d_output);

    BenchmarkResult result;

    result.benchmark =
        fused ? "cuda_softmax_rmsnorm_fused"
              : "cuda_softmax_rmsnorm_unfused";
    result.status = "pass";
    result.environment = environment;

    result.raw_samples = samples_ms;

    result.measurements = calculate_statistics(
        samples_ms,
        "kernel"
    );

    result.measurements.push_back({
        "launches",
        "",
        fused ? 1.0 : 2.0
    });

    result.measurements.push_back({
        "hidden",
        "",
        static_cast<double>(hidden)
    });

    return result;
}

BenchmarkResult run_cuda_softmax_rmsnorm_fused(
    const EnvironmentInfo& environment)
{
    return run_cuda_softmax_rmsnorm_impl(environment, true);
}

BenchmarkResult run_cuda_softmax_rmsnorm_unfused(
    const EnvironmentInfo& environment)
{
    return run_cuda_softmax_rmsnorm_impl(environment, false);
}

} // namespace benchkit