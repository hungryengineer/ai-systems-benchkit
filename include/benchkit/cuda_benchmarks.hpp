#pragma once

#include "benchkit/result.hpp"

#include <string>
#include <vector>

namespace benchkit {

BenchmarkResult run_cuda_vector_add(const EnvironmentInfo& environment);

}