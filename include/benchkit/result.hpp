#pragma once

#include "benchkit/environment.hpp"

#include <string>
#include <vector>

namespace benchkit {

struct Measurement {
    std::string name;
    std::string unit;
    double value;
};

struct BenchmarkResult {
    std::string benchmark;
    std::string status;
    EnvironmentInfo environment;
    int sample_count;

    std::vector<double> raw_samples;

    std::vector<Measurement> measurements;
};

} // namespace benchkit