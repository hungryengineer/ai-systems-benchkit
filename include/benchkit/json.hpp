#pragma once

#include "benchkit/result.hpp"

#include <string>

namespace benchkit {

BenchmarkResult load_result(const std::string& path);

}