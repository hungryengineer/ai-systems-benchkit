#include "benchkit/statistics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace benchkit {

double percentile(
    const std::vector<double>& values,
    double percentile_value)
{
    if (values.empty())
    {
        throw std::runtime_error("Cannot calculate percentile of empty data");
    }

    std::vector<double> sorted = values;

    std::sort(sorted.begin(), sorted.end());

    double position =
        (percentile_value / 100.0) * (sorted.size() - 1);

    std::size_t lower =
        static_cast<std::size_t>(std::floor(position));

    std::size_t upper =
        static_cast<std::size_t>(std::ceil(position));

    if (lower == upper)
    {
        return sorted[lower];
    }

    double fraction = position - lower;

    return sorted[lower] +
           fraction * (sorted[upper] - sorted[lower]);
}

std::vector<Measurement> calculate_statistics(
    const std::vector<double>& samples,
    const std::string& unit)
{
    if (samples.empty())
    {
        throw std::runtime_error("Cannot calculate statistics of empty data");
    }

    double min_time =
        *std::min_element(samples.begin(), samples.end());

    double max_time =
        *std::max_element(samples.begin(), samples.end());

    double sum =
        std::accumulate(
            samples.begin(),
            samples.end(),
            0.0);

    double mean =
        sum / samples.size();

    double squared_diff_sum = 0.0;

    for (double value : samples)
    {
        double difference = value - mean;

        squared_diff_sum +=
            difference * difference;
    }

    double variance =
        squared_diff_sum /
        (samples.size() - 1);

    double standard_deviation =
        std::sqrt(variance);

    double median = percentile(samples, 50.0);
    double p95 = percentile(samples, 95.0);
    double p99 = percentile(samples, 99.0);

    return {
        {"min", unit, min_time},
        {"median", unit, median},
        {"mean", unit, mean},
        {"p95", unit, p95},
        {"p99", unit, p99},
        {"max", unit, max_time},
        {"stddev", unit, standard_deviation}
    };
}

} // namespace benchkit