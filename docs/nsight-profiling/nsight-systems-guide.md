# Nsight Systems guide (`nsys`)

Nsight Systems (`nsys`) profiles the whole process and produces a timeline:
kernel launches, memory copies, CUDA API calls, and CPU/GPU concurrency.
Use it to answer *"where does the time go / how long does each kernel really
take / how much time is lost to copies and launch gaps?"*.

## Basic timeline trace

BenchKit supports running a single benchmark, which keeps the trace small:

```bash
mkdir -p profiles
# Full timeline trace:
nsys profile \
  --output=profiles/softmax \
  --force-overwrite \
  ./build/benchkit run softmax
```

Output: `profiles/softmax.nsys-rep`. Open it in the Nsight Systems GUI
(`nsys-ui profiles/softmax.nsys-rep`) or inspect it from the CLI:

```bash
# Text timeline summary (kernel + copy + API timings)
nsys stats --report cuda_gpu_trace profiles/softmax.nsys-rep

# Per-kernel aggregated stats (min/avg/max duration, count, inst.)
nsys stats --report cuda_gpu_kern_sum profiles/softmax.nsys-rep

# Copy statistics (H2D/D2H)
nsys stats --report cuda_memcpy_sum profiles/softmax.nsys-rep
```

Because BenchKit's `run` already performs multiple warm-up kernels and 20
timed samples per benchmark, one `nsys` trace captures the whole benchmark
loop — the summary reports aggregate over all those launches.

## What to read per benchmark

| Benchmark         | `nsys profile --output=profiles/<x> ./build/benchkit run <x>` | Key reports |
| ----------------- | ------------------------------------------------------------ | ----------- |
| softmax           | `run softmax`                                                | `cuda_gpu_kern_sum` (kernel time), `cuda_gpu_trace` (launch gaps) |
| rmsnorm           | `run rmsnorm`                                                | same as above |
| vector-add        | `run vector-add`                                             | + `cuda_memcpy_sum` (H2D/D2H dominate) |
| matmul            | `run matmul`                                                 | compare naive vs tiled vs cuBLAS kernel times |

## Keep traces minimal

- Always pass `--force-overwrite` so old traces are replaced.
- Prefer single benchmarks (`run softmax`) over `run` (all) — full runs
  produce large traces and slow down profiling.
- Use `--trace=cuda,osrt` to trim tracing overhead if the timeline gets noisy.

## Notes

- `nsys` measures wall-clock *between* timeline markers; it includes any
  other activity on the device, so run with the GPU otherwise idle.
- Duration from `nsys` (avg over all launches in the run) should match the
  in-code `cudaEvent` median. Disagreement > few % is a red flag for the
  in-code timing (see `migration-plan.md`).