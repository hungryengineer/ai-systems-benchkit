
import statistics
import time

import torch


ROWS = 512
HIDDEN = 4096
WARMUP = 5
ITERATIONS = 20
TOLERANCE = 1e-5


def cpu_reference(values):
    """Match the scalar CPU reference in test_softmax_vectorized.cu."""
    output = []

    for row in values:
        max_val = max(row)
        exps = [
            torch.exp(torch.tensor(v - max_val)).item()
            for v in row
        ]
        total = sum(exps)
        output.append([v / total for v in exps])

    return torch.tensor(output, dtype=torch.float32)


def benchmark_eager(x, warmup, iterations):
    """Measure PyTorch eager steady-state GPU execution."""
    for _ in range(warmup):
        y = torch.softmax(x, dim=-1)

    torch.cuda.synchronize()

    timings_ms = []

    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        y = torch.softmax(x, dim=-1)
        end.record()
        end.synchronize()

        timings_ms.append(start.elapsed_time(end))

    return y, timings_ms


def benchmark_compiled(x, warmup, iterations):
    """Measure compilation separately from steady-state GPU execution."""

    def operation(tensor):
        return torch.softmax(tensor, dim=-1)

    compiled_operation = torch.compile(operation)

    # Includes compilation and the first invocation.
    start_compile = time.perf_counter()
    y = compiled_operation(x)
    torch.cuda.synchronize()
    compile_time_ms = (time.perf_counter() - start_compile) * 1000

    # Warm up the compiled path.
    for _ in range(warmup):
        y = compiled_operation(x)

    torch.cuda.synchronize()

    timings_ms = []

    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        y = compiled_operation(x)
        end.record()
        end.synchronize()

        timings_ms.append(start.elapsed_time(end))

    return y, compile_time_ms, timings_ms


def report_timings(label, timings_ms):
    print(f"{label} median: {statistics.median(timings_ms):.6f} ms")
    print(f"{label} mean:   {statistics.mean(timings_ms):.6f} ms")



def benchmark_once(operation, x, iterations):
    timings_ms = []
    output = None

    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output = operation(x)
        end.record()
        end.synchronize()

        timings_ms.append(start.elapsed_time(end))

    return output, timings_ms


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    count = ROWS * HIDDEN
    values = [((i % 100) - 50) / 10.0 for i in range(count)
    ]
    cpu_input = torch.tensor(
        values, dtype=torch.float32
    ).reshape(ROWS, HIDDEN)
    x = cpu_input.cuda()

    eager = lambda tensor: torch.softmax(tensor, dim=-1)
    compiled = torch.compile(eager)

    # Compile and warm up both implementations before collecting samples.
    eager(x)
    compiled(x)
    torch.cuda.synchronize()

    for _ in range(WARMUP):
        eager(x)
        compiled(x)
    torch.cuda.synchronize()

    eager_samples = []
    compiled_samples = []

    # Alternate which implementation runs first each round.
    for round_index in range(ITERATIONS):
        order = [eager, compiled]
        if round_index % 2:
            order.reverse()

        results = {}
        for operation in order:
            _, timings = benchmark_once(operation, x, 1)
            results[operation] = timings[0]

        eager_samples.append(results[eager])
        compiled_samples.append(results[compiled])

    eager_output = eager(x)
    compiled_output = compiled(x)
    torch.cuda.synchronize()

    max_error = (
        eager_output - compiled_output
    ).abs().max().item()

    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Shape: {ROWS} x {HIDDEN}")
    print(
        f"Eager median: "
        f"{statistics.median(eager_samples):.6f} ms"
    )
    print(
        f"Compiled median: "
        f"{statistics.median(compiled_samples):.6f} ms"
    )
    print(
        f"Eager mean: {statistics.mean(eager_samples):.6f} ms"
    )
    print(
        f"Compiled mean: "
        f"{statistics.mean(compiled_samples):.6f} ms"
    )
    print(f"Max error (eager vs compiled): {max_error:.3e}")
    print(
        "Numerical check:",
        "PASS" if max_error < TOLERANCE else "FAIL",
    )


if __name__ == "__main__":
    main()
