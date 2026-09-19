# AI Systems BenchKit

A small, reproducible C++20 / CUDA benchmarking framework for AI infrastructure.
It measures CPU and GPU kernels with the same methodology, writes machine-readable
JSON results, and can compare two runs to detect regressions.

The project currently ships four benchmarks:

| Benchmark          | What it measures                                             |
| ------------------ | ------------------------------------------------------------ |
| `cpu_vector_add`   | CPU memory-bandwidth bound element-wise add                  |
| `cuda_vector_add`  | GPU element-wise add: kernel, H2D, D2H and end-to-end        |
| `cuda_matmul`      | 1024x1024x1024 SGEMM: naive, shared-memory tiled, and cuBLAS |

---

## Build

Requirements: a C++20 compiler, CMake >= 3.20, and the CUDA toolkit (`nvcc`,
`cudart`, `cublas`).

```bash
cmake -S . -B build
cmake --build build
```

Run the test suite:

```bash
cd build && ctest
```

## Usage

```bash
# Run every benchmark and write JSON results to results/raw/
./build/benchkit run

# Compare two result files of the same benchmark
./build/benchkit compare results/raw/cpu_vector_add_A.json \
                       results/raw/cpu_vector_add_B.json
```

`run` must be executed from the repository root so the relative
`results/raw/` output directory resolves correctly.

---

## Benchmark methodology

All benchmarks follow the same rules so numbers are comparable:

1. **Warm-up.** Kernels are launched at least once (and GPU copies issued)
   before any timing starts, to avoid measuring first-touch / library
   initialization and to let the GPU reach a steady clock state.
2. **Timing.** GPU timings use `cudaEventRecord` / `cudaEventElapsedTime`
   around the work on the stream, with `cudaEventSynchronize` before reading
   the elapsed time. CPU timings use `std::chrono`.
3. **Samples.** Each benchmark records 20 samples. Raw samples are stored in
   the result file; the summary is derived from them.
4. **Statistics.** The reusable `statistics` module
   (`include/benchkit/statistics.hpp`) computes:
   `min`, `median`, `mean`, `p95`, `p99`, `max`, `stddev`.
   Percentiles use linear interpolation between sorted neighbours.
5. **Correctness.** Every benchmark validates its output against a known
   analytic answer. A run that fails correctness is reported as `fail`; it is
   never silently accepted.
6. **Output.** Results are written as JSON (schema in
   `schemas/benchmark_result.schema.json`) containing the benchmark name,
   status, sample count, the captured environment, the raw samples, and the
   summary measurements. A copy of the environment is embedded in every file
   so a result is self-describing.

Every benchmark emits the same shape, so `compare` works across runs and
computes the percentage change per metric.

---

## CPU vs GPU

- On the **CPU**, work runs directly against host memory; a benchmark is just
  a timed loop over `new`/`std::vector` buffers.
- On the **GPU**, data must be moved over the PCIe bus first. A complete GPU
  pipeline is: host `new` -> `cudaMalloc` device buffer -> **H2D** copy ->
  kernel launch -> **D2H** copy -> correctness check -> free.
- Because transfers dominate small problems, BenchKit measures the GPU path in
  four parts rather than as one opaque number:
  - **kernel** — pure device execution time,
  - **H2D** — host-to-device copy latency,
  - **D2H** — device-to-host copy latency,
  - **e2e** — the full H2D + kernel + D2H round trip, i.e. the latency an
    application actually observes per iteration.

The naive/tiled/cuBLAS matmul comparison isolates *compute* performance from
*transfer* performance.

---

## CUDA Vector Add

Element-wise `C[i] = A[i] + B[i]` over `N = 1 << 20` floats (4 MiB per array).
This is memory-bandwidth bound, so it is a useful sanity check that the device
and transfer path work.

```
              CPU
               │
       1. new CPU memory
               │
               ▼
        h_a / h_b / h_c
               │
       2. cudaMalloc
               │
               ▼
        d_a / d_b / d_c
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
  i = blockIdx * blockDim + threadIdx
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
  7. cudaFree / delete
```

The kernel uses a 1-D grid with 256 threads per block and the standard global
index formula. The `if (i < n)` guard handles the case where grid size rounds
up past the array length.

Source: `src/cuda_vector_add.cu`.

---

## Naive MatMul

A textbook SGEMM: one thread computes one `C[row][col]` entry, reading a full
row of `A` and a full column of `B` from global memory for every output element.

- Strengths: simple and obviously correct.
- Weakness: each output element performs `K` global loads of `A` and `K` of
  `B`, so the same data is re-read from global memory `N` / `M` times. It is
  heavily memory-latency bound.

Source: `matmul_kernel` in `src/cuda_matmul.cu`.

---

## Tiled MatMul and shared memory

The tiled kernel improves locality by processing the `K` dimension in
`TILE_SIZE x TILE_SIZE` (16x16) tiles:

1. Each thread of the 16x16 block loads **one element** of the `A` and `B`
   tiles into `__shared__` memory.
2. `__syncthreads()` ensures all writes are visible before reads.
3. Each thread accumulates its partial dot product from the shared-memory
   tiles.
4. A second `__syncthreads()` prevents the next tile load from overwriting
   shared memory that other threads are still reading.

Shared memory is on-chip and dramatically faster than global memory, so each
`A`/`B` element is fetched from DRAM once per tile instead of once per output
element. This is why the tiled kernel is faster than the naive one even though
both do the same `2*M*N*K` floating-point operations.

The block size must equal `TILE_SIZE` (16x16) for this kernel.

Source: `matmul_tiled_kernel` in `src/cuda_matmul.cu`.

---

## cuBLAS

The third MatMul implementation calls NVIDIA's `cublasSgemm`. cuBLAS matrices
are **column-major** while the benchmark's buffers are **row-major**, so the
kernel computes `C^T = B^T * A^T`. The existing row-major buffers can be passed
directly because their memory layout already corresponds to the column-major
representation of the transposed matrices. cuBLAS is the optimized,
vendor-tuned reference point and includes its own warm-up runs.

The cuBLAS library is linked via `CUDA::cublas`
(`find_package(CUDAToolkit)` in `CMakeLists.txt`).

Source: `run_cuda_matmul_cublas` in `src/cuda_matmul.cu`.

---

## GFLOPS

For an `M x K x N` SGEMM there are `2 * M * K * N` floating-point operations
(each multiply-add counts as two). Given a time in milliseconds:

```
GFLOPS = (2 * M * K * N) / (median_ms * 1e6)
```

For `M = K = N = 1024`, that is `2 * 1024^3 = 2.147 GFLOP` per call. GFLOPS is
reported for the naive, tiled, and cuBLAS kernels so their compute throughput
can be compared directly.

---

## Hardware conditions

All timings are sensitive to the machine, driver, clocks, and thermal state.
Results from BenchKit runs on:

| Component            | Value                                  |
| -------------------- | -------------------------------------- |
| OS                   | Pop!_OS 24.04 LTS                      |
| Kernel               | 7.0.11-76070011-generic                |
| CPU                  | AMD Ryzen AI 7 350 w/ Radeon 860M, 8C/16T |
| RAM                  | ~24 GB                                 |
| GPU                  | NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB |
| NVIDIA driver        | 595.84                                 |
| `nvcc`               | CUDA 12.0                              |
| GCC                  | 13.3                                   |
| CMake                | 3.28.3                                 |

Laptop GPUs throttle aggressively. Running several benchmarks back-to-back can
move medians by an order of magnitude versus a cold, single benchmark. Compare
files produced under similar conditions, and prefer the `min` and `median`
columns over `mean` when the machine is noisy.

---

## Example results

Representative output from `benchkit run` on the machine above (numbers vary
with load and thermals):

**CPU vector add** (1M elements, 20 samples, ms):

| min | median | mean | p95 | max | stddev |
| --- | ------ | ---- | --- | --- | ------ |
| 5.92 | 20.93 | 17.25 | 26.10 | 36.75 | 8.76 |

**CUDA vector add** (median, ms):

| kernel | H2D | D2H | e2e |
| ------ | --- | --- | --- |
| 0.56   | 4.81 | 2.71 | 7.44 |

**MatMul 1024x1024x1024** (median):

| Kernel | Median (ms) | GFLOPS |
| ------ | ----------- | ------ |
| Naive  | 52.88       | 40.6   |
| Tiled  | 38.52       | 55.7   |
| cuBLAS | 12.32       | 174.4  |

As expected, `cuBLAS` is fastest, the shared-memory tiled kernel is a clear
improvement over the naive kernel, and the H2D/D2H transfers dominate the
end-to-end vector-add latency.

---

## Repository layout

```
CMakeLists.txt              Build definition (C++20 + CUDA, links CUDA::cublas)
include/benchkit/           Public headers
  cpu_benchmarks.hpp
  cuda_benchmarks.hpp
  cuda_matmul.hpp
  environment.hpp
  json.hpp
  result.hpp                BenchmarkResult / Measurement types
  statistics.hpp            percentile() + calculate_statistics()
include/nlohmann/json.hpp   Vendored JSON library
src/
  main.cpp                  CLI (run / compare), JSON printing & writing
  cpu_benchmarks.cpp
  cuda_vector_add.cu
  cuda_matmul.cu
  environment.cpp           System/GPU info collection
  json.cpp                  Result loading
  statistics.cpp            Reusable statistics module
tests/test_result.cpp       Smoke test
schemas/                    JSON schema for result files
results/raw/                Generated results (git-ignored)
```

## Regenerating results

Generated artifacts are ignored by Git (see `.gitignore`):

- `build/` — CMake build tree,
- `results/raw/*.json` — benchmark output,
- object files, CUDA artifacts, editor/OS files.

Delete `build/` and `results/raw/*.json` freely; both are reproduced by
`cmake -S . -B build && cmake --build build` and `./build/benchkit run`.
