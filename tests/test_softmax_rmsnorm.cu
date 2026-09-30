#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace benchkit {

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

__global__ void softmax_rmsnorm_kernel(
    const float* input,
    const float* gamma,
    float* output,
    int hidden,
    float eps);

} // namespace benchkit

namespace {

constexpr int TEST_BLOCK_SIZE = 256;

constexpr float EPS = 1e-5f;

// CPU reference: softmax(x) then RMSNorm(softmax, gamma).
void softmax_rmsnorm_cpu(
    const std::vector<float>& input,
    const std::vector<float>& gamma,
    std::vector<float>& output,
    int rows,
    int hidden)
{
    for (int row = 0; row < rows; ++row) {

        int offset = row * hidden;

        float max_val = input[offset];

        for (int col = 1; col < hidden; ++col) {
            max_val = std::max(max_val, input[offset + col]);
        }

        float sum = 0.0f;

        for (int col = 0; col < hidden; ++col) {
            float value = std::exp(input[offset + col] - max_val);
            output[offset + col] = value;
            sum += value;
        }

        for (int col = 0; col < hidden; ++col) {
            output[offset + col] /= sum;
        }

        float sum2 = 0.0f;

        for (int col = 0; col < hidden; ++col) {
            float value = output[offset + col];
            sum2 += value * value;
        }

        float rms =
            std::sqrt(sum2 / hidden + EPS);

        for (int col = 0; col < hidden; ++col) {
            output[offset + col] =
                (output[offset + col] / rms) * gamma[col];
        }
    }
}

float run_impl(
    int rows,
    int hidden,
    bool fused)
{
    const size_t element_count =
        static_cast<size_t>(rows) * hidden;

    const size_t bytes =
        element_count * sizeof(float);

    const size_t gamma_bytes =
        static_cast<size_t>(hidden) * sizeof(float);

    std::vector<float> h_input(element_count);
    std::vector<float> h_gamma(hidden);
    std::vector<float> h_output(element_count);
    std::vector<float> h_reference(element_count);

    // Deterministic input matching the benchmarks.
    for (size_t i = 0; i < element_count; ++i) {
        h_input[i] =
            static_cast<float>((i % 100) - 50) / 10.0f;
    }

    for (int i = 0; i < hidden; ++i) {
        h_gamma[i] = 1.0f + 0.01f * static_cast<float>(i % 7);
    }

    softmax_rmsnorm_cpu(
        h_input,
        h_gamma,
        h_reference,
        rows,
        hidden
    );

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
    dim3 block(TEST_BLOCK_SIZE);
    dim3 block_rmsnorm(128);

    if (fused) {
        benchkit::softmax_rmsnorm_kernel<<<grid, block>>>(
            d_input,
            d_gamma,
            d_output,
            hidden,
            EPS
        );
    } else {
        benchkit::softmax_kernel_vectorized<<<grid, block>>>(
            d_input,
            d_tmp,
            hidden
        );

        benchkit::rmsnorm_kernel<<<grid, block_rmsnorm>>>(
            d_tmp,
            d_gamma,
            d_output,
            hidden,
            EPS
        );
    }

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    );

    float max_error = 0.0f;

    for (size_t i = 0; i < element_count; ++i) {
        max_error = std::max(
            max_error,
            std::fabs(h_output[i] - h_reference[i])
        );
    }

    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_tmp);
    cudaFree(d_output);

    return max_error;
}

} // namespace

int main()
{
    // hidden must be divisible by 4 (float4 chunks).
    const int sizes[] = { 4, 8, 16, 32, 4096 };
    const int rows = 3;

    bool all_pass = true;

    for (int hidden : sizes) {

        if (hidden % 4 != 0) {
            std::printf(
                "hidden=%d  -- skipped (requires hidden %% 4 == 0)\n",
                hidden
            );
            all_pass = false;
            continue;
        }

        float fused_error = run_impl(rows, hidden, true);
        float unfused_error = run_impl(rows, hidden, false);

        const bool fused_pass = fused_error < 1e-5f;
        const bool unfused_pass = unfused_error < 1e-5f;

        std::printf(
            "hidden=%-5d rows=%d  fused_max_error=%-12g unfused_max_error=%-12g  %s / %s\n",
            hidden,
            rows,
            fused_error,
            unfused_error,
            fused_pass ? "PASS" : "FAIL",
            unfused_pass ? "PASS" : "FAIL"
        );

        all_pass = all_pass && fused_pass && unfused_pass;
    }

    std::printf("Result: %s\n", all_pass ? "ALL PASS" : "FAIL");

    return all_pass ? 0 : 1;
}