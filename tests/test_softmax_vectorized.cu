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

} // namespace benchkit

namespace {

constexpr int TEST_BLOCK_SIZE = 256;

// CPU reference (same math as the kernel, scalar).
void softmax_cpu(
    const std::vector<float>& input,
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
    }
}

float run_case(int rows, int hidden)
{
    const size_t element_count =
        static_cast<size_t>(rows) * hidden;

    const size_t bytes =
        element_count * sizeof(float);

    std::vector<float> h_input(element_count);
    std::vector<float> h_output(element_count);
    std::vector<float> h_reference(element_count);

    // Deterministic input spanning positive and negative values.
    for (size_t i = 0; i < element_count; ++i) {
        h_input[i] =
            static_cast<float>((i % 100) - 50) / 10.0f;
    }

    softmax_cpu(h_input, h_reference, rows, hidden);

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
    dim3 block(TEST_BLOCK_SIZE);

    benchkit::softmax_kernel_vectorized<<<grid, block>>>(
        d_input,
        d_output,
        hidden
    );

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
    cudaFree(d_output);

    return max_error;
}

} // namespace

int main()
{
    // hidden must be divisible by 4 (float4 chunks) and >= 1 vector.
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

        float max_error = run_case(rows, hidden);

        const bool pass = max_error < 1e-5f;

        std::printf(
            "hidden=%-5d rows=%d  max_error=%-12g  %s\n",
            hidden,
            rows,
            max_error,
            pass ? "PASS" : "FAIL"
        );

        all_pass = all_pass && pass;
    }

    std::printf("Result: %s\n", all_pass ? "ALL PASS" : "FAIL");

    return all_pass ? 0 : 1;
}