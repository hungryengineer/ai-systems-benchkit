| Component            | Your machine           | Project target        |
| -------------------- | ---------------------- | --------------------- |
| OS                   | Pop!_OS 24.04 LTS      | Linux                 |
| CPU                  | Ryzen AI 7 350, 8C/16T | Ryzen AI 350, 16-core |
| RAM                  | ~24 GB                 | 24 GB                 |
| GPU                  | RTX 5050, 8 GB         | RTX 5050, 8 GB        |
| NVIDIA driver        | 595.84                 | —                     |
| Driver-reported CUDA | 13.2                   | —                     |
| `nvcc`               | **CUDA 12.0**          | CUDA C++              |
| GCC                  | 13.3                   | C++20 capable         |
| CMake                | 3.28.3                 | CMake                 |
| Git                  | 2.43                   | Git                   |


              CPU
               │
               │
       1. new CPU memory
               │
               ▼
        h_a / h_b / h_c
               │
               │
       2. cudaMalloc
               │
               ▼
        d_a / d_b / d_c
               │
               │
       3. cudaMemcpy H2D
               │
               ▼
              GPU
               │
       4. kernel launch
               │
       ┌───────┴────────┐
       │                │
    Block 0          Block 1 ...
       │
    Threads
       │
       ▼
  i = blockIdx ×
      blockDim +
      threadIdx
       │
       ▼
 C[i] = A[i] + B[i]
       │
       ▼
       d_c
       │
       │ 5. cudaMemcpy D2H
       ▼
      h_c
       │
       ▼
  6. correctness
       │
       ▼
  7. cudaFree/delete