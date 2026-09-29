# Nsight Profiler Integration

Documentation for moving BenchKit's GPU measurement from the in-code CUDA
timers to the NVIDIA Nsight profilers.

| Doc | Purpose |
| --- | --- |
| [README](README.md) | Overview, motivation, quick start |
| [current-benchmarking](current-benchmarking.md) | How GPU benchmarks are timed today |
| [migration-plan](migration-plan.md) | The changes to adopt Nsight |
| [nsight-systems](nsight-systems-guide.md) | `nsys` usage for this project |
| [nsight-compute](nsight-compute-guide.md) | `ncu` usage for this project |
| [results-and-compare](results-and-compare.md) | How profiler data maps to BenchKit results |