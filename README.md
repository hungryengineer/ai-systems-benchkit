# AI Systems BenchKit

# Week 2 — CUDA Kernel Optimization Laboratory

## Objective

Implement, validate, optimize, and profile CUDA kernels for Softmax, RMSNorm, and fused operations. Compare custom implementations with framework baselines and evaluate CUDA Graph execution.

## Hardware and software

- GPU: NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB
- CPU: AMD Ryzen AI 7 350
- OS: Pop!_OS 24.04 LTS
- NVIDIA driver: 595.84
- System CUDA Toolkit: 12.0
- GCC: 13.3
- CMake: 3.28.3
- PyTorch: 2.14.1+cu130

## Key findings

### Softmax optimization

Vectorized float4 loads and stores improved measured latency over the earlier scalar implementation. Warp-shuffle reductions did not show a clear additional application-level improvement. Register-cached exponentials increased register usage without a consistent latency benefit.

**Decision:** retain the vectorized implementation; do not adopt register caching as the default optimization.

### Softmax + RMSNorm fusion

The cited profiler run measured approximately 190.2 µs for the unfused path and 63.8 µs for the fused path. Fusion reduced two kernel launches to one and eliminated an intermediate global-memory round trip.

**Decision:** retain the fused implementation for the tested workload.

### CUDA Graphs

The comparison used equivalent workloads: 1,000 executions per sample, 20 measured samples, and 10 warm-up executions. The graph contained one kernel node and was replayed 1,000 times per sample.

A representative paired session measured approximately 1.75 ms for normal launches and 2.05 ms for graph replay, excluding the graph's one-time setup cost of approximately 0.56 ms.

**Decision:** the tested one-node graph design did not demonstrate a speedup. This does not establish that CUDA Graphs are generally slower; graphs containing multiple operations may have different submission-overhead characteristics.

### PyTorch comparison

For a 512 × 4096 FP32 Softmax workload, the observed median latencies were:

- Custom vectorized CUDA: 30.592 µs
- PyTorch eager: 52.096 µs
- PyTorch compiled: 36.544 µs

The eager and compiled measurements were collected in an alternating paired run. The custom CUDA result came from a separate run, so the three-way ranking remains preliminary.

The eager and compiled outputs agreed within approximately 4.657 × 10⁻¹⁰ maximum absolute error. The first compiled invocation took approximately 3.336 seconds, including compilation and initialization.

### Matrix multiplication and transfers

The recorded benchmark results showed cuBLAS outperforming the custom tiled and naive matrix-multiplication implementations. The tiled implementation improved on the naive implementation. The vector-add experiment showed that host-device transfers can materially affect end-to-end latency.

## Measurement and correctness

Correctness checks passed for the tested Softmax shapes and the fused/unfused Softmax + RMSNorm paths. Measurements remain sensitive to laptop GPU clocks, thermal state, background load, and timing methodology.

CUDA-event timings, host-observed timings, and profiler timings must not be treated as interchangeable without accounting for their different measurement scopes.

## Limitations

- PyTorch/vendor-library comparisons are not a complete, controlled three-way performance study.
- The PyTorch compiled benchmark's mean latency was higher than its median, indicating variability.
- Some profiler utilization figures do not independently establish a compute-bound or memory-bound classification.
- The one-node CUDA Graph result is specific to the tested workload and implementation.

## Conclusion

Week 2 demonstrated the value of correctness gates, vectorized memory access, kernel fusion, profiling, and disciplined benchmarking. It also showed that plausible optimizations—such as register caching or one-node graph replay—must be evaluated empirically.

The next stage should build on these measurements while preserving workload equivalence, numerical validation, and reproducible timing.


A small, reproducible C++20 / CUDA benchmarking framework for AI infrastructure.
It measures CPU and GPU kernels with the same methodology, writes machine-readable
JSON results, and can compare two runs to detect regressions.

The project currently ships four benchmarks:

| Benchmark          | What it measures                                             |
| ------------------ | ------------------------------------------------------------ |
| `cpu_vector_add`   | CPU memory-bandwidth bound element-wise add                  |
| `cuda_vector_add`  | GPU element-wise add: kernel, H2D, D2H and end-to-end        |
| `cuda_rmsnorm`     | GPU RMSNorm over 512 x 4096 rows                              |
| `cuda_softmax`     | GPU softmax baseline: one block per row, scalar accesses     |
| `cuda_softmax_vectorized` | Same softmax with `float4` access and warp-shuffle reductions |
| `cuda_softmax_rmsnorm_fused` | Day-4 fusion: softmax + RMSNorm in one kernel (`softmax_rmsnorm_kernel`) |
| `cuda_softmax_rmsnorm_unfused` | Same result via 2 launches: V2 softmax then scalar RMSNorm |
| `cuda_graph_normal` | CUDA Graphs baseline: 1000 host-driven launches of `tiny_kernel` |
| `cuda_graph_captured` | One-node `tiny_kernel` graph replayed 1000x per sample, setup timed separately |
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

## Day 3: CUDA Softmax optimization experiment

Day 3 is conceptually complete. It is the worked example of the whole
experiment loop the project is built around:

```
Correct → measure → identify bottleneck → change one thing → profile → keep/reject
```

Softmax was chosen because it is a row-reduction kernel with a well-known set
of possible optimizations. The workload under test is kept fixed across every
version — 512 rows x 4096 floats, 256 threads/block, one block per row — so
each step changes exactly one thing. Every change must first pass the
correctness gate (see `tests/test_softmax_vectorized.cu`, sizes 4/8/16/32/4096)
before it is measured.

### V1 — baseline (kept untouched)

`softmax_kernel` in `src/cuda_softmax.cu`. Scalar row pass with a
shared-memory tree reduction. Initial block size was raised 128 -> 256.

- **Change:** none (this is the reference).
- **Result:** 256 threads/block improved device utilization and latency over
  128; occupancy is limited by launching only a few blocks per row count.

### V2 — memory optimization (kept)

`softmax_kernel_vectorized` in `src/cuda_softmax_vectorized.cu`. The algorithm
is byte-for-byte identical; only the memory-access granularity changes:
scalar `float` loads/stores -> `float4` (4 chunks/thread at hidden=4096).

- **Condition:** `hidden % 4 == 0` (no scalar tail path yet).
- **Profiler result:** substantially higher utilization. Achieved occupancy
  jumped ~52% -> ~90%, DRAM throughput ~16% -> ~36%, memory throughput
  ~17% -> ~42% on the RTX 5050 (see `docs/nsight-profiling/`).

### V3 — reduction optimization (kept, but no clear win)

Replaced the shared-memory tree reduction in the vectorized kernel with a
warp-shuffle reduction:

```
shared-memory tree reduction  →  warp_shuffle_down + block reduce
```

- **Change:** only the two reductions; float4 loads/stores, block size, and
  workload are identical.
- **What improved:** shared-memory footprint and cross-warp synchronization
  (`__syncthreads` per log step -> one per reduction). The correctness gate
  caught the classic shuffling bug on the way — `warp_reduce_*` leaves the
  total only in lane 0, so the final value must be broadcast through shared
  memory (the `total`/`total_max` pattern in `block_reduce_sum`/`_max`).
- **Profiler result:** the Nsight profiler did **not** establish a clear
  additional application-level win over V2. That is a useful negative result:
  the reduction was not the bottleneck, so the speedup is swallowed by memory
  traffic. This is intentionally kept as a documented optimization that did
  not clearly improve the result.

### V4 — register-cached fused exp (kept, also no clear win)

The V4 change makes the intermediate softmax values never leave the SM: the
exp pass writes its results into a per-thread `float4 cached[4]` register
array, and normalization reads straight back from registers. No global
write/read-back of the exp values at all.

- **Change:** only the middle pass of the V3 kernel — discard the `output4`
  store + immediate reload before the normalize pass; every other aspect is
  identical.
- **Profiler result (Nsight Compute on RTX 5050):** DRAM throughput went
  ~36% -> ~39%, achieved occupancy ~90% -> ~91.5%, but duration went up
  slightly (~67 us -> ~73-76 us) and Compute (SM) throughput rose to ~70%.
  ncu's bottleneck report now says *compute is more heavily utilized than
  memory*. **Second documented negative result.** The eliminated read-back was
  already served from L1/L2 for the V3 code path, so DRAM traffic barely
  moved; the kernel is bounded by `expf` SFU work plus the shuffle reductions,
  not by DRAM bandwidth.
- **New, actionable finding exposed by the V4 profile: the partial last wave.**
  With `grid = 512` blocks and a 6-blocks-per-SM occupancy limit (registers are
  the limiter now, 39 regs/thread), only `6 * 20 SMs = 120` blocks are resident,
  so the launch runs **4.27 waves/SM** — four full waves plus a 32-block tail
  that leaves most SMs idle. ncu's `Wave Occupancy` estimate attributes up to
  ~20% of runtime to that tail. Grid sizing turned out to be a bigger lever
  than either optimization above.

### Day 4: Fusion (kept — first clear win)

The remaining Week-2 requirement was **fusion**: merge two ops that touch the
same data into one kernel and measure whether the reduced work actually shows
up. Chosen pair: **softmax followed by RMSNorm** — a realistic transformer
pre-norm + attention-weight ordering.

`softmax_rmsnorm_kernel` in `src/cuda_softmax_rmsnorm.cu` computes
`softmax(x)` then `RMSNorm(softmax, gamma) · gamma` in a single pass per row:

- reuse the V4 register-caching pattern: `exp(x - max)` values live in
  `float4 cached[4]` between the reduction and the normalize pass,
- the RMSNorm sum-of-squares is accumulated in the **same** exp pass
  (`sum2 = sum(e^2)`; `rms = sqrt(sum2 / (hidden * sum^2) + eps)`), so no
  second read of the softmax output is needed,
- normalization and `gamma` scaling are applied straight from registers.

The unfused baseline (`run_cuda_softmax_rmsnorm_unfused`) calls the existing
`softmax_kernel_vectorized` then the scalar `rmsnorm_kernel` — two library
kernels, one full write + two full reads of the 8 MiB intermediate.

Comparison (ncu, clock-locked, RTX 5050; benchkit wall-clock in parens):

| Dimension        | Unfused (2 kernels)          | Fused (`softmax_rmsnorm_kernel`) |
| ---------------- | ---------------------------- | -------------------------------- |
| Launch count     | 2                            | **1**                            |
| Latency (ncu)    | 75.2 + 115.0 = **190.2 us**  | **63.8 us** (3.0x faster)        |
| Latency (benchkit median) | **0.080 ms**       | **0.024 ms** (3.3x faster)       |
| DRAM throughput  | 38.9% / 25.4%                | **45.9%**                        |
| Memory throughput | 110.5 / 72.7 GB/s           | **131.9 GB/s**                   |
| Global traffic (logical) | ~48 MiB (2x reads of x + tmp write + 2x reads of tmp + write) | **~24 MiB** (2x reads of x + write, gamma cached) |
| Numerical error (vs CPU) | <= 2.4e-07             | <= 2.4e-07 (identical)           |

**Verdict: keep.** This is the first change in the whole softmax arc that the
profiler confirms as a clear win — and the reason is instructive: it is *not*
just the halved DRAM traffic (x was already L2-resident in the smaller
single-kernel experiments, but here the 8 MiB `tmp` buffer is a *full extra
round-trip through global memory*). Fusion also replaced the scalar
`rmsnorm_kernel`, which was latency-bound (24 regs, 128 threads, 115 us) with
vectorized, 256-thread work — the fused kernel at 63.8 us is faster than
either unfused kernel alone. The correctness gate still passes with the same
max error, so the reordering cost nothing numerically.

The takeaway for the experiment loop: register shuffles and reduction swaps
(V3/V4) moved nothing because the bottleneck was elsewhere; **fusion removed
an entire global-memory round-trip and an entire scalar pass** — two real
costs that micro-optimizations never touched.

---

## Day 5: CUDA Graphs

Both graph benchmarks live in `src/cuda_graph_benchmark.cu` and run the exact
same workload — 1000 executions per sample of a one-warp `tiny_kernel` (32
floats, `+1.0f` per launch) — so they are directly comparable:

- `cuda_graph_normal`: the host launches the kernel N=1000 times; each of 20
  samples host-times `{1000 launches + device sync}`.
- `cuda_graph_captured`: the graph captures exactly **one** `tiny_kernel`
  launch (`nodes_per_graph = 1`), is instantiated once, and is replayed
  N=1000 times **per sample**. Setup (capture + instantiate) is host-timed
  **once and reported separately** (`setup_ms`); the replay phase times 20
  samples of 1000 replays.

Both paths report the workload equivalence explicitly:
`kernel_executions_per_sample = 1000` and `total_kernel_executions = 20010`
(10 warmup + 20 samples x 1000) — identical for normal and graph paths. They
also share the identical timing structure — a `steady_clock` window around
`{1000 dispatches + device sync}`, 10 warmup executions, and CUDA error checks
at the same points — so each raw sample covers the same timing scope in both
paths. The correctness gate holds for both (`x[i] == i + total_kernel_executions`,
exact in `float`).

Representative numbers from a single paired session (RTX 5050):

| Metric | Normal (1000 launches) | Graph (1-node, 1000 replays) |
| ------ | ---------------------- | ---------------------------- |
| Median sample time | 1.75 ms | 2.05 ms |
| Per kernel / per replay | 1.75 us | 2.05 us |
| One-time setup | — | 0.56 ms (`setup_ms`) |
| Total (20 samples + setup) | 35.0 ms | 41.6 ms |

The honest conclusion: **for a single-node graph replayed N times there is no
speedup.** The value of CUDA Graphs is replacing *many* host-side launch
submissions with a single `cudaGraphLaunch`, but a 1-node graph still costs one
`cudaGraphLaunch` per execution, so it does **not** remove the per-launch host
overhead — replay lands slightly *slower* per execution than the plain launch
path, and the 0.56 ms setup is paid on top. Graphs pay off when a graph holds
many nodes so one replay submits the whole workload, not when it is used as a
1-to-1 substitute for a single kernel launch.

Why not capture on the default stream: this driver returns
`cudaErrorStreamCaptureUnsupported` for `cudaStreamBeginCapture` on the legacy
NULL stream (verified for both ThreadLocal and Global modes), so the graph
path uses one dedicated stream for capture and replay — semantically identical
to the normal path's single default stream.

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
  cuda_rmsnorm.cu
  cuda_softmax.cu           Baseline softmax kernel (V1)
  cuda_softmax_vectorized.cu  float4 + warp-shuffle variant (V2/V3, V4 register cache)
  cuda_softmax_rmsnorm.cu   Day-4 fused softmax+RMSNorm and unfused baseline
  cuda_graph_benchmark.cu   Day-5 CUDA Graphs: 1-node graph replayed 1000x vs 1000 normal launches
  cuda_matmul.cu
  environment.cpp           System/GPU info collection
  json.cpp                  Result loading
  statistics.cpp            Reusable statistics module
tests/test_result.cpp       Smoke test
tests/test_softmax_vectorized.cu  Correctness gate for the V2/V3 kernel
tests/test_softmax_rmsnorm.cu     Correctness gate for the Day-4 fused / unfused paths
docs/nsight-profiling/      Nsight Systems / Compute integration docs
schemas/                    JSON schema for result files
results/raw/                Generated results (git-ignored)
profiles/                   Profiler artifacts (git-ignored)
```

## Regenerating results

Generated artifacts are ignored by Git (see `.gitignore`):

- `build/` — CMake build tree,
- `results/raw/*.json` — benchmark output,
- object files, CUDA artifacts, editor/OS files.

Delete `build/` and `results/raw/*.json` freely; both are reproduced by
`cmake -S . -B build && cmake --build build` and `./build/benchkit run`.
