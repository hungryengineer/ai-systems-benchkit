#include "benchkit/cuda_matmul.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

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

std::vector<double> benchkit::run_cuda_matmul()
{
    constexpr int M = 1024;
    constexpr int K = 1024;
    constexpr int N = 1024;

    constexpr int samples = 20;

    const std::size_t bytes_A =
        static_cast<std::size_t>(M) *
        K *
        sizeof(float);

    const std::size_t bytes_B =
        static_cast<std::size_t>(K) *
        N *
        sizeof(float);

    const std::size_t bytes_C =
        static_cast<std::size_t>(M) *
        N *
        sizeof(float);

    //configure host memory
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 2.0f);
    std::vector<float> h_C(M * N, 0.0f);

    //configure device memory
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    //input
    cudaMemcpy(
        d_A,
        h_A.data(),
        bytes_A,
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_B,
        h_B.data(),
        bytes_B,
        cudaMemcpyHostToDevice);

    // grid configuration
    dim3 threads(16, 16);

    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    //warmup
    matmul_kernel<<<blocks, threads>>>(
        d_A,
        d_B,
        d_C,
        M,
        K,
        N);

    cudaDeviceSynchronize();

    //cuda event
    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<double> timings;
    timings.reserve(samples);

    //measure
    for (int i = 0; i < samples; ++i)
    {
        cudaEventRecord(start);

        matmul_kernel<<<blocks, threads>>>(
            d_A,
            d_B,
            d_C,
            M,
            K,
            N);

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

    //copy the result
        cudaMemcpy(
        h_C.data(),
        d_C,
        bytes_C,
        cudaMemcpyDeviceToHost);

    //check correctness
    bool correct = true;

    for (std::size_t i = 0; i < h_C.size(); ++i)
    {
        if (std::fabs(h_C[i] - 2048.0f) > 1e-3f)
        {
            correct = false;
            break;
        }
    }

    std::cout << "CUDA MatMul: "
              << (correct ? "PASS" : "FAIL")
              << '\n';

    //delete memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return timings;
}

//2nd kernal depicting tile shared memory

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
        // block copies exactly one element of each tile, so the block reads
        // a full 16x16 tile per __syncthreads().
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
        // Every thread walks its row of tile_A and column of tile_B.
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += tile_A[threadIdx.y][k]
                 * tile_B[k][threadIdx.x];
        }

        // Wait again before the next iteration overwrites shared memory:
        // no thread may write tile_A/tile_B while others are still reading.
        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}