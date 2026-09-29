# How BenchKit times CUDA today

This describes the *current* (pre-Nsight) measurement path, so the migration
scope is explicit.

## The timing flow

Every CUDA benchmark in `src/` follows the same pattern. Taking
`src/cuda_softmax.cu` as the canonical example:

```cpp
constexpr int rows    = 2;
constexpr int hidden  = 4096;
constexpr int warmup  = 5;
constexpr int samples = 20;
```

1. **Warm-up.** The kernel is launched `warmup` times before timing starts.
   This avoids first-touch / lazy-init effects and lets the GPU reach a
   steady clock state.
2. **Per-sample timing.** For each of `samples` iterations:

   ```cpp
   cudaEventRecord(start);
   softmax_kernel<<<grid, block>>>(d_input, d_output, hidden);
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&elapsed_ms, start, stop);
   samples_ms.push_back(elapsed_ms);
   ```

   Events are recorded on the same stream as the kernel, so the elapsed time
   is the *device execution time* of the kernel only (no H2D/D2H, no launch
   overhead).
3. **Correctness.** Output is validated against a CPU reference
   (`softmax_cpu`, `rmsnorm` reference, etc.) and `max_error` is reported.
4. **Statistics.** `benchkit::calculate_statistics` (`statistics.cpp`) turns
   the raw samples into `min`, `median`, `mean`, `p95`, `p99`, `max`,
   `stddev` (percentiles interpolated).
5. **Output.** A `BenchmarkResult` is written to `results/raw/cuda_<name>_<ts>.json`
   following `schemas/benchmark_result.schema.json`:

   ```json
   {
     "benchmark": "cuda_softmax",
     "status": "pass",
     "sample_count": 20,
     "environment": { "hostname": "pop-os", "...": "..." },
     "raw_samples": [0.024224, 0.020064, "..."],
     "measurements": [
       { "metric": "median", "unit": "ms", "value": 0.0192 },
       { "metric": "max_error", "unit": "", "value": 5.96e-08 }
     ]
   }
   ```

## What the in-code timing measures

- Pure kernel GPU time (device elapsed), `unit = "ms"`.
- One value per launch; 20 samples per benchmark run.

## What the in-code timing cannot tell you

- Memory-bound vs compute-bound (no FLOPS / bandwidth breakdown).
- Occupancy, achieved vs peak, stall reasons.
- H2D/D2H copy overlap and launch gaps on the timeline.
- Memory-op throughput (DRAM, L1/L2, shared) per kernel.
- Whether the 20 samples are distorted by noise, clock boost/drop, or
  contention from concurrent processes.

These are exactly the questions Nsight Systems and Nsight Compute answer.

## Timed kernels (current set)

| Benchmark             | File                | Kernel               |
| --------------------- | ------------------- | -------------------- |
| `cuda_vector_add`     | `src/cuda_vector_add.cu` | `vector_add_kernel` |
| `cuda_rmsnorm`        | `src/cuda_rmsnorm.cu`    | `rmsnorm_kernel`    |
| `cuda_softmax`        | `src/cuda_softmax.cu`    | `softmax_kernel`    |
| `cuda_matmul` suite   | `src/cuda_matmul.cu`     | `matmul_kernel`, `matmul_tiled_kernel`, cuBLAS |

The `run` CLI exposes these as: `vector-add`, `rmsnorm`, `softmax`, `matmul`
(`./build/benchkit run <name>` runs a single benchmark, `./build/benchkit run`
runs all).