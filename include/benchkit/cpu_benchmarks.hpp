#pragma once

#include <cstddef>

namespace benchkit {

struct CpuBenchmarkResult {
    double elapsed_ms;
    bool correct;
};

CpuBenchmarkResult run_vector_add(std::size_t elements);

}