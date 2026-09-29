# Nsight Profiler Integration

## Why

BenchKit currently times CUDA kernels *inside the code* using
`cudaEventRecord` / `cudaEventElapsedTime`. That gives one number per launch
(duration) but nothing about *why* a kernel is slow: it cannot tell whether a
kernel is memory-bound or compute-bound, what its occupancy is, how the
memory hierarchy is used, or where the time actually goes.

The NVIDIA Nsight profilers answer those questions:

- **NVIDIA Nsight Systems** (`nsys`) — a system-wide, low-overhead profiler.
  It produces a timeline of kernel launches, memory copies (H2D/D2H), CUDA
  API calls, and CPU/GPU concurrency. It is the *where-time-goes* tool.
- **NVIDIA Nsight Compute** (`ncu`) — a kernel-level profiler. It gives
  per-kernel hardware counters and metrics: achieved occupancy, DRAM vs
  shared-memory traffic, warp stalls, achieved vs peak FLOPS/bandwidth
  (roofline), and more. It is the *why-it-is-slow* tool.

Goal: keep BenchKit's reproducible JSON methodology, but replace/augment the
hand-rolled kernel timings with Nsight data so results are trustworthy and
diagnosable.

## Environment

```text
GPU:            NVIDIA GeForce RTX 5050 Laptop GPU (8 GiB)
Driver:         595.84
Nsight Systems: nsys 2022.4.2
Nsight Compute: ncu 2026.1.0 (installed to ~/NVIDIA/nsight-compute/2026.1.0)
OS:             Pop!_OS 24.04 (Linux)
```

`ncu`/`ncu-ui` are on `PATH` via `~/.bashrc` (prepended so they shadow the
older CUDA-toolkit copy at `/usr/bin/ncu`).

## Quick start

```bash
# My timeline of one kernel (softmax), trace output in ./profiles/
./build/benchkit run softmax
nsys profile --output=profiles/softmax ./build/benchkit run softmax

# Kernel-level metrics for the same scenario
ncu --kernel-name regex:softmax_kernel -o profiles/softmax ./build/benchkit run softmax
```

Run details and all commands are in the per-tool guides.

## Reading order

1. [How BenchKit times CUDA today](current-benchmarking.md)
2. [The migration plan](migration-plan.md)
3. [Nsight Systems guide](nsight-systems-guide.md)
4. [Nsight Compute guide](nsight-compute-guide.md)
5. [Results and compare](results-and-compare.md)