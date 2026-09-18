#pragma once

#include <vector>

namespace benchkit {

struct CudaBenchmarkResult {
    std::vector<double> kernel_timings_ms;
    std::vector<double> h2d_timings_ms;
    std::vector<double> d2h_timings_ms;
    std::vector<double> e2e_timings_ms;
};

CudaBenchmarkResult run_cuda_vector_add();
}