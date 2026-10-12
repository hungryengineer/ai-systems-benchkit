
import statistics
import time

import torch


ROWS = 3
HIDDEN = 4096
WARMUP = 20
ITERATIONS = 100
TOLERANCE = 1e-5


def cpu_reference(values):
    """Match the scalar CPU reference in test_softmax_vectorized.cu."""
    output = []
    for row in values:
        max_val = max(row)
        exps = [torch.exp(torch.tensor(v - max_val)).item() for v in row]
        total = sum(exps)
        output.append([v / total for v in exps])
    return torch.tensor(output, dtype=torch.float32)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    # Reproduce the deterministic input pattern in the CUDA test.
    count = ROWS * HIDDEN
    values = [((i % 100) - 50) / 10.0 for i in range(count)]
    cpu_input = torch.tensor(values, dtype=torch.float32).reshape(ROWS, HIDDEN)
    x = cpu_input.to("cuda")

    # Eager baseline.
    for _ in range(WARMUP):
        y = torch.softmax(x, dim=-1)
    torch.cuda.synchronize()

    timings_ms = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        y = torch.softmax(x, dim=-1)
        torch.cuda.synchronize()
        timings_ms.append((time.perf_counter() - start) * 1000)

    reference = cpu_reference(cpu_input.tolist())
    y_cpu = y.cpu()

    max_error = (y_cpu - reference).abs().max().item()
    max_row_sum_error = (y.sum(dim=-1) - 1.0).abs().max().item()

    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Shape: {ROWS} x {HIDDEN}")
    print(f"Median latency: {statistics.median(timings_ms):.6f} ms")
    print(f"Mean latency: {statistics.mean(timings_ms):.6f} ms")
    print(f"Max absolute error vs CPU reference: {max_error:.3e}")
    print(f"Max row-sum error: {max_row_sum_error:.3e}")
    print(f"Numerical check: {'PASS' if max_error < TOLERANCE else 'FAIL'}")


if __name__ == "__main__":
    main()
