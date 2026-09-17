#pragma once

#include "benchkit/result.hpp"

#include <string>
#include <vector>

namespace benchkit {

double percentile(const std::vector<double>& values, double percentile_value);

std::vector<Measurement> calculate_statistics(const std::vector<double>& samples, const std::string& unit);

}