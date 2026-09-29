# Migration plan: in-code CUDA timing -> Nsight

## Principle

BenchKit's value is *reproducible, comparable numbers*. The migration does
**not** discard the JSON methodology. It changes the source of the numbers:

- **Phase 1** — Nsight as a companion tool (trace the existing in-code
  timings and cross-check them).
- **Phase 2** — Nsight replaces the in-code timing as the source of truth for
  kernel duration and adds the diagnostic metrics the in-code timers cannot
  produce.
- **Phase 3** — results are re-emitted in the BenchKit JSON schema so
  `compare` still works.

## Phase 1: trace + cross-check

Goal: nothing changes in code; we learn how trustworthy the in-code timings
are and get the first Nsight data.

1. Profile each benchmark once with `nsys profile` (timeline) and
   `ncu -o` (kernel metrics). Commands are in the tool guides.
2. Compare the `nsys` kernel duration for e.g. `softmax_kernel` against the
   `median` in `results/raw/cuda_softmax_*.json`. They should agree within a
   few percent; any large gap signals timing methodology issues (clock
   states, measurement overhead, other work on the device).
3. Record the comparison in `reports/`.

Deliverable: a baseline report per kernel.

## Phase 2: Nsight as the measurement source

Goal: kernel timing moves out of the benchmark binary into the profiler.

1. **Nsight Systems** becomes the *duration* source:
   - `nsys stats` gives per-kernel min/avg/max duration, launch gaps, and
     H2D/D2H copy durations from a single trace.
2. **Nsight Compute** becomes the *perf* source:
   - `--section MemoryWorkloadAnalysis`, `--section Occupancy`,
     `--section SpeedOfLight`, `--section WarpStateStats` answer the
     "memory-bound vs compute-bound / why slow" questions.
3. Benchmarks keep **only** the correctness check and the warm-up loop (the
   warm-up also helps `ncu`, which replays and repeats kernels). The
   `cudaEvent` timing loop can be reduced to a single steady-state run or
   removed.
4. Decide whether to keep the in-code events as a cheap "always-on" smoke
   metric (recommended: keep them; profilers cannot be run in CI on every
   push).

## Phase 3: results reconcile with the BenchKit schema

Goal: Nsight numbers still land in `results/` and `compare` keeps working.

1. Emit per-kernel metrics extracted from `nsys stats --report cuda_gpu_kern_sum`
   and `ncu --csv` into the `BenchmarkResult` shape
   (`schemas/benchmark_result.schema.json`), for example:

   ```json
   {
     "benchmark": "cuda_softmax",
     "status": "pass",
     "environment": { "...": "..." },
     "raw_samples": ["(from nsys, see profiles/)", "..."],
     "measurements": [
       { "metric": "nsys_min_kernel_ms",     "unit": "ms",  "value": 0.0173 },
       { "metric": "nsys_avg_kernel_ms",     "unit": "ms",  "value": 0.0190 },
       { "metric": "ncu_occupancy_pct",      "unit": "%",   "value": 12.5 },
       { "metric": "ncu_dram_utilization",   "unit": "%",   "value": 91.2 },
       { "metric": "ncu_gpu_utilization",    "unit": "%",   "value": 62.0 },
       { "metric": "max_error",              "unit": "",    "value": 5.96e-08 }
     ]
   }
   ```

2. The `raw` artifacts (`.nsys-rep`, `.ncu-rep`, CSV exports) live under
   `profiles/`, git-ignored; the reconciled JSON stays in `results/raw/`.
3. `compare` continues to operate on the reconciled JSON unchanged.

## Migration roadmap

| #    | Change                                              | Status |
| ---- | --------------------------------------------------- | ------ |
| 1    | Add `profiles/` output dir + gitignore               | Todo   |
| 2    | Baseline `nsys` + `ncu` run of all 4 benchmarks      | Todo   |
| 3    | Cross-check in-code median vs nsys duration          | Todo   |
| 4    | Export ncu sections to `reports/`                    | Todo   |
| 5    | Reconciled JSON writer (script or C++)               | Todo   |
| 6    | Optionally drop the in-code `cudaEvent` loop         | Todo   |

## Open questions

- Keep in-code `cudaEvent` timing? (Recommended: yes, as smoke test.)
- `ncu` on this laptop GPU (RTX 5050) requires the app-wide replay flag or
  per-kernel limiting to keep runtime sane. See the Nsight Compute guide.
- Do matmul/cuBLAS benchmarks need their own metric sets? Yes — cuBLAS has no
  source in this repo, so `ncu` speeds/frequencies are the only visibility.