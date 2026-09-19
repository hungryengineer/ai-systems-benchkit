#include "benchkit/cuda_matmul.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <cublas_v2.h>

__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N)
{
    int row =
        blockIdx.y * blockDim.y + threadIdx.y;

    int col =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k]
                 * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

// Tiled kernel using shared memory. Requires TILE_SIZE x TILE_SIZE blocks.
constexpr int TILE_SIZE = 16;

__global__ void matmul_tiled_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N)
{
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < K; tile += TILE_SIZE)
    {
        // Load A and B tiles into shared memory. Each thread of the 16x16
        // block copies exactly one element of each tile.
        tile_A[threadIdx.y][threadIdx.x] =
            A[(blockIdx.y * TILE_SIZE + threadIdx.y) * K
              + tile + threadIdx.x];

        tile_B[threadIdx.y][threadIdx.x] =
            B[(tile + threadIdx.y) * N
              + blockIdx.x * TILE_SIZE + threadIdx.x];

        // Make sure every thread has finished writing its shared-memory
        // element before any thread starts reading another thread's data.
        __syncthreads();

        // Accumulate the partial dot product over this tile's K dimension.
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += tile_A[threadIdx.y][k]
                 * tile_B[k][threadIdx.x];
        }

        // Wait again before the next iteration overwrites shared memory.
        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}

namespace {

using MatMulKernel = void (*)(
    const float*,
    const float*,
    float*,
    int,
    int,
    int);

// Runs warm-up plus `samples` timed iterations of the given kernel and
// returns the per-iteration elapsed times in milliseconds.
std::vector<double> time_kernel(
    MatMulKernel kernel,
    const dim3& blocks,
    const dim3& threads,
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int K,
    int N,
    int samples)
{
    // Warm-up.
    kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);

    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<double> timings;
    timings.reserve(samples);

    for (int i = 0; i < samples; ++i)
    {
        cudaEventRecord(start);

        kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);

        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float milliseconds = 0.0f;

        cudaEventElapsedTime(
            &milliseconds,
            start,
            stop);

        timings.push_back(
            static_cast<double>(milliseconds));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return timings;
}

// All buffers, grid configuration and input copies shared by the naive and
// tiled kernel benchmarks. A = 1, B = 2, so C is expected to equal 2048.
struct MatMulBuffers {
    std::vector<float> h_A;
    std::vector<float> h_B;
    std::vector<float> h_C;

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    dim3 threads;
    dim3 blocks;
};

MatMulBuffers setup_matmul(int M, int K, int N)
{
    MatMulBuffers buffers;

    buffers.h_A.assign(
        static_cast<std::size_t>(M) * K,
        1.0f);

    buffers.h_B.assign(
        static_cast<std::size_t>(K) * N,
        2.0f);

    buffers.h_C.assign(
        static_cast<std::size_t>(M) * N,
        0.0f);

    const std::size_t bytes_A =
        static_cast<std::size_t>(M) * K * sizeof(float);

    const std::size_t bytes_B =
        static_cast<std::size_t>(K) * N * sizeof(float);

    const std::size_t bytes_C =
        static_cast<std::size_t>(M) * N * sizeof(float);

    cudaMalloc(&buffers.d_A, bytes_A);
    cudaMalloc(&buffers.d_B, bytes_B);
    cudaMalloc(&buffers.d_C, bytes_C);

    cudaMemcpy(
        buffers.d_A,
        buffers.h_A.data(),
        bytes_A,
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        buffers.d_B,
        buffers.h_B.data(),
        bytes_B,
        cudaMemcpyHostToDevice);

    buffers.threads = dim3(16, 16);

    buffers.blocks = dim3(
        (N + buffers.threads.x - 1) / buffers.threads.x,
        (M + buffers.threads.y - 1) / buffers.threads.y
    );

    return buffers;
}

bool verify_matmul(const std::vector<float>& h_C)
{
    bool correct = true;

    for (std::size_t i = 0; i < h_C.size(); ++i)
    {
        if (std::fabs(h_C[i] - 2048.0f) > 1e-3f)
        {
            correct = false;
            break;
        }
    }

    return correct;
}

} // namespace

namespace benchkit {

std::vector<double> run_cuda_matmul()
{
    constexpr int M = 1024;
    constexpr int K = 1024;
    constexpr int N = 1024;

    constexpr int samples = 20;

    MatMulBuffers buffers = setup_matmul(M, K, N);

    std::vector<double> timings =
        time_kernel(
            matmul_kernel,
            buffers.blocks,
            buffers.threads,
            buffers.d_A,
            buffers.d_B,
            buffers.d_C,
            M,
            K,
            N,
            samples);

    const std::size_t bytes_C =
        static_cast<std::size_t>(M) * N * sizeof(float);

    cudaMemcpy(
        buffers.h_C.data(),
        buffers.d_C,
        bytes_C,
        cudaMemcpyDeviceToHost);

    bool correct = verify_matmul(buffers.h_C);

    std::cout << "CUDA MatMul (naive): "
              << (correct ? "PASS" : "FAIL")
              << '\n';

    cudaFree(buffers.d_A);
    cudaFree(buffers.d_B);
    cudaFree(buffers.d_C);

    return timings;
}

std::vector<double> run_cuda_matmul_tiled()
{
    constexpr int M = 1024;
    constexpr int K = 1024;
    constexpr int N = 1024;

    constexpr int samples = 20;

    MatMulBuffers buffers = setup_matmul(M, K, N);

    std::vector<double> timings =
        time_kernel(
            matmul_tiled_kernel,
            buffers.blocks,
            buffers.threads,
            buffers.d_A,
            buffers.d_B,
            buffers.d_C,
            M,
            K,
            N,
            samples);

    const std::size_t bytes_C =
        static_cast<std::size_t>(M) * N * sizeof(float);

    cudaMemcpy(
        buffers.h_C.data(),
        buffers.d_C,
        bytes_C,
        cudaMemcpyDeviceToHost);

    bool correct = verify_matmul(buffers.h_C);

    std::cout << "CUDA MatMul (tiled): "
              << (correct ? "PASS" : "FAIL")
              << '\n';

    cudaFree(buffers.d_A);
    cudaFree(buffers.d_B);
    cudaFree(buffers.d_C);

    return timings;
}

std::vector<double> run_cuda_matmul_cublas()
{
    constexpr int M = 1024;
    constexpr int K = 1024;
    constexpr int N = 1024;

    constexpr int WARMUP = 5;
    constexpr int SAMPLES = 20;

    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 2.0f);
    std::vector<float> h_C(M * N, 0.0f);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(
        d_A,
        h_A.data(),
        M * K * sizeof(float),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_B,
        h_B.data(),
        K * N * sizeof(float),
        cudaMemcpyHostToDevice
    );

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    /*
     * cuBLAS uses column-major matrices.
     *
     * Our matrices are stored row-major.
     *
     * C = A × B
     *
     * We therefore compute:
     *
     * C^T = B^T × A^T
     *
     * The existing row-major buffers can be passed directly
     * because their memory layout corresponds to the
     * column-major representation of the transposed matrices.
     */
    for (int i = 0; i < WARMUP; ++i)
    {
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            M,
            K,
            &alpha,
            d_B,
            N,
            d_A,
            K,
            &beta,
            d_C,
            N
        );
    }

    cudaDeviceSynchronize();

    std::vector<double> samples;
    samples.reserve(SAMPLES);

    for (int i = 0; i < SAMPLES; ++i)
    {
        cudaEvent_t start;
        cudaEvent_t stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            M,
            K,
            &alpha,
            d_B,
            N,
            d_A,
            K,
            &beta,
            d_C,
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

        samples.push_back(milliseconds);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaMemcpy(
        h_C.data(),
        d_C,
        M * N * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    bool correct = true;

    for (int i = 0; i < M * N; ++i)
    {
        if (std::abs(h_C[i] - 2048.0f) > 1e-3f)
        {
            correct = false;
            break;
        }
    }

    if (!correct)
    {
        std::cerr << "cuBLAS MatMul correctness check FAILED\n";
    }
    else
    {
        std::cout << "cuBLAS MatMul correctness check PASSED\n";
    }

    cublasDestroy(handle);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return samples;
}

} // namespace benchkit