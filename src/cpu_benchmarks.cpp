#include "benchkit/cpu_benchmarks.hpp"

#include <chrono>
#include <cmath>
#include <vector>

namespace benchkit {

CpuBenchmarkResult run_vector_add(const std::size_t elements)
{
    std::vector<double> a(elements, 1.0);
    std::vector<double> b(elements, 2.0);
    std::vector<double> c(elements, 0.0);

    const auto start = std::chrono::high_resolution_clock::now();

    for (std::size_t i = 0; i < elements; ++i) {
        c[i] = a[i] + b[i];
    }

    const auto end = std::chrono::high_resolution_clock::now();

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    // Every element should be exactly 3.0
    bool correct = true;
    for (std::size_t i = 0; i < elements; ++i) {
        if (std::fabs(c[i] - 3.0) > 1e-9) {
            correct = false;
            break;
        }
    }

    return {elapsed_ms, correct};
}

} // namespace benchkit
