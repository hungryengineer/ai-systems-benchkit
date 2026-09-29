# Nsight Compute guide (`ncu`)

Nsight Compute (`ncu`) profiles at the **kernel** level with hardware
counters. It replays kernels to collect metrics, so the profiled run is
slower than real-time and results refer to the *profiled* execution.

Use `ncu` to answer *"is this kernel memory-bound or compute-bound? how is
occupancy? where does the time actually go?"*.

## Correct invocation on this machine

BenchKit runs kernels multiple times (warm-up + 20 timed samples). To keep
`ncu` runtime sane and focus on one kernel, always restrict profiling:

```bash
mkdir -p profiles

# Profile only softmax's kernel, one replay per metric set:
ncu \
  --kernel-name regex:softmax_kernel \
  --launch-count 1 \
  --launch-skip 20 \
  -o profiles/softmax \
  ./build/benchkit run softmax
```

- `--kernel-name regex:softmax_kernel` — skip everything else.
- `--launch-count 1` — profile a single launch (after the warm-up loop).
- `--launch-skip 20` — skip the first 20 launches (5 warm-up + samples) so
  the device is warm and clocks are steady.
- `-o profiles/softmax` — writes `profiles/softmax.ncu-rep` (view in the
  `ncu-ui` GUI or export to CSV).

If profiling the *whole suite* at once, restrict to the benchmark of interest
via the CLI (`run rmsnorm`, `run softmax`, ...) the same way as `nsys`.

## Key sections

```bash
ncu ... --set full   # comfort: everything, slow
ncu ... --set basic  # default set
```

Per-benchmark recommended sections:

| Kernel situation           | Sections to collect (`--section`) |
| -------------------------- | --------------------------------- |
| Where is the time?         | `SpeedOfLight`, `Occupancy`, `WarpStateStats` |
| Memory-bound analysis      | `MemoryWorkloadAnalysis`, `MemoryWorkloadAnalysis_Tables` |
| Compute-bound analysis     | `ComputeWorkloadAnalysis` |
| Launch config sanity       | `LaunchStats`, `Occupancy` |

Example for softmax (expected: memory-bound, low occupancy from one-block-per-row):

```bash
ncu \
  --kernel-name regex:softmax_kernel \
  --launch-count 1 --launch-skip 20 \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section Occupancy \
  ./build/benchkit run softmax
```

## Reading the output for our kernels

The current kernels are all **memory-bound**, so the numbers that matter:

- **DRAM Throughput %** (`SpeedOfLight`) — how close to peak memory
  bandwidth. High (> 70%) ⇒ memory-bound, matches the two/three-pass design.
- **Compute (SM) Throughput %** — usually low for these kernels; confirms the
  bottleneck is memory, not math.
- **Achieved Occupancy** — softmax/rmsnorm use one block per row; with
  `hidden` rows small this caps occupancy and shows up as low `Achieved
  Occupancy` + scheduling stalls.
- **Mem Busy / Max Bandwidth** — DRAM utilization per kernel.

Expected findings for the naive kernels:

| Kernel            | Expected result |
| ----------------- | --------------- |
| `vector_add_kernel` | coalesced, near DRAM peak (pure bandwidth) |
| `rmsnorm_kernel`   | two passes over input ⇒ ~2x the traffic of one pass |
| `softmax_kernel`   | three passes over input ⇒ ~3x traffic (max, sum exp, normalize) |
| `matmul_kernel` (naive) | bad reuse, DRAM/latency bound, low FLOPS |
| `matmul_tiled_kernel` | better traffic via shared memory, higher DRAM efficiency |

These directly quantify the "how many times do you touch the data" story from
the kernel design.

## Caveats

- `ncu` **replays** kernels; do not compare its durations to `nsys`/in-code
  timings directly — compare *metrics* (occupancy, utilization, traffic) to
  identify the bottleneck, not ms.
- Installed version: `ncu 2026.1.0` at `~/NVIDIA/nsight-compute/2026.1.0`
  (PATH entry in `~/.bashrc`). Install dir must stay put; uninstall = delete
  that folder.
- On this mobile GPU (RTX 5050) counter collection works out of the box; if
  it is ever gated by `ERR_NVGPUCTRPERM`, run with `sudo ncu ...` or set the
  NVIDIA vendor-controlled CPU performance counter permission.