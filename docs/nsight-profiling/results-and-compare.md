# Results and compare

How the profiler data maps back into BenchKit's result files and the
`compare` flow.

## Directory layout

```text
results/
  raw/            # reconciled JSON results (schema-valid)
profiles/         # .nsys-rep / .ncu-rep raw artifacts (git-ignored)
reports/          # baseline analysis derived from the profiler artifacts
```

## What stays

The `BenchmarkResult` JSON in `results/raw/` keeps the same schema
(`schemas/benchmark_result.schema.json`) so:

- `./build/benchkit compare a.json b.json` keeps working;
- correctness (`status`, `max_error`) stays an in-code, trusted check;
- environment metadata stays embedded in each file.

The raw profiler artifacts (`.nsys-rep`, `.ncu-rep`, CSV exports) are the new
`raw_samples`-equivalent trail, stored under `profiles/`.

## What changes

The `measurements` list gains profiler-derived metrics alongside the
in-code ones. Naming convention — no collisions with existing metrics:

| Prefix   | Source | Example metrics |
| -------- | ------ | --------------- |
| (in-code)| existing | `median` (ms), `mean` (ms), `max_error` |
| `nsys_`  | `nsys stats` | `nsys_avg_kernel_ms`, `nsys_min_kernel_ms`, `nsys_max_kernel_ms`, `nsys_h2d_ms`, `nsys_d2h_ms` |
| `ncu_`   | `ncu --csv`   | `ncu_dram_utilization_pct`, `ncu_sm_utilization_pct`, `ncu_occupancy_pct`, `ncu_mem_busy_pct`, `ncu_l2_hit_rate_pct` |

## Compare semantics

`compare` computes the percentage change per metric between two runs. With
profiler metrics added:

- Same benchmark, same machine, two runs → still compares cleanly.
- The `ncu_*` metrics are most valuable for *regression diagnosis*: e.g. a
  `ncu_dram_utilization_pct` drop with an `nsys_avg_kernel_ms` increase points
  at a memory-access-pattern regression, not at noise.
- Keep `sample_count` / `raw_samples` populated even when timings now come
  from `nsys`; if a benchmark drops the in-code event loop, note it in the
  BenchmarkResult via a new optional field (e.g. `"timing_source": "nsys"`).

## Export commands (for the reconciled JSON)

```bash
# nsys -> CSV
nsys stats --report cuda_gpu_kern_sum --format csv \
  profiles/softmax.nsys-rep

# ncu -> CSV (metrics only for the profiled kernel)
ncu --kernel-name regex:softmax_kernel --csv \
  --section SpeedOfLight --section MemoryWorkloadAnalysis \
  ./build/benchkit run softmax
```

The CSV is machine-parseable and is what the reconciler script feeds into the
JSON `measurements`.