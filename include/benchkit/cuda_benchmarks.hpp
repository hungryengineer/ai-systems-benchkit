#pragma once

#include "benchkit/result.hpp"

#include <string>
#include <vector>

namespace benchkit {

BenchmarkResult run_cuda_vector_add(const EnvironmentInfo& environment);

BenchmarkResult run_cuda_rmsnorm(
    const EnvironmentInfo& environment
);

BenchmarkResult run_cuda_softmax(
    const EnvironmentInfo& environment
);

BenchmarkResult run_cuda_softmax_vectorized(
    const EnvironmentInfo& environment
);

BenchmarkResult run_cuda_softmax_rmsnorm_fused(
    const EnvironmentInfo& environment
);

BenchmarkResult run_cuda_softmax_rmsnorm_unfused(
    const EnvironmentInfo& environment
);

}