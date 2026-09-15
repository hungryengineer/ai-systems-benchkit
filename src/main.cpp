#include "benchkit/cpu_benchmarks.hpp"
#include "benchkit/environment.hpp"
#include "benchkit/json.hpp"
#include "benchkit/result.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace {

void print_json_string(const std::string& value)
{
    std::cout << "\"" << value << "\"";
}

void print_environment_json(const benchkit::EnvironmentInfo& env)
{
    std::cout << "{\n";

    std::cout << "    \"hostname\": ";
    print_json_string(env.hostname);
    std::cout << ",\n";

    std::cout << "    \"kernel\": ";
    print_json_string(env.kernel);
    std::cout << ",\n";

    std::cout << "    \"os\": ";
    print_json_string(env.os);
    std::cout << ",\n";

    std::cout << "    \"cpu\": ";
    print_json_string(env.cpu);
    std::cout << ",\n";

    std::cout << "    \"compiler\": ";
    print_json_string(env.compiler);
    std::cout << ",\n";

    std::cout << "    \"cmake\": ";
    print_json_string(env.cmake);
    std::cout << ",\n";

    std::cout << "    \"cuda_toolkit\": ";
    print_json_string(env.cuda_toolkit);
    std::cout << ",\n";

    std::cout << "    \"nvidia_driver\": ";
    print_json_string(env.nvidia_driver);
    std::cout << ",\n";

    std::cout << "    \"gpu\": ";
    print_json_string(env.gpu);

    std::cout << "\n  }";
}

void print_result_json(const benchkit::BenchmarkResult& result)
{
    std::cout << "{\n";

    std::cout << "  \"benchmark\": ";
    print_json_string(result.benchmark);
    std::cout << ",\n";

    std::cout << "  \"status\": ";
    print_json_string(result.status);
    std::cout << ",\n";

    std::cout << "  \"raw_samples\": [";

    for (std::size_t i = 0; i < result.raw_samples.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }

        std::cout << result.raw_samples[i];
    }

    std::cout << "],\n";

    std::cout << "  \"environment\": ";
    print_environment_json(result.environment);
    std::cout << ",\n";

    std::cout << "  \"measurements\": [\n";

    for (std::size_t i = 0; i < result.measurements.size(); ++i)
    {
        const auto& measurement = result.measurements[i];

        std::cout << "    {\n";

        std::cout << "      \"name\": ";
        print_json_string(measurement.name);
        std::cout << ",\n";

        std::cout << "      \"unit\": ";
        print_json_string(measurement.unit);
        std::cout << ",\n";

        std::cout << "      \"value\": "
                  << measurement.value
                  << "\n";

        std::cout << "    }";

        if (i + 1 < result.measurements.size())
        {
            std::cout << ",";
        }

        std::cout << "\n";
    }

    std::cout << "  ]\n";
    std::cout << "}\n";
}

} // namespace

void write_result_json(
    const benchkit::BenchmarkResult& result,
    const std::string& path)
{
    std::ofstream file(path);

    if (!file)
    {
        throw std::runtime_error(
            "Failed to open result file: " + path);
    }

    file << "{\n";

    file << "  \"benchmark\": ";
    file << "\"" << result.benchmark << "\"";
    file << ",\n";

    file << "  \"status\": ";
    file << "\"" << result.status << "\"";
    file << ",\n";

    file << "  \"sample_count\": "
         << result.sample_count
         << ",\n";

    file << "  \"raw_samples\": [";

    for (std::size_t i = 0; i < result.raw_samples.size(); ++i)
    {
        if (i > 0)
        {
            file << ", ";
        }

        file << result.raw_samples[i];
    }

    file << "],\n";

    file << "  \"environment\": ";
    file << "{\n";

    file << "    \"hostname\": \"" << result.environment.hostname << "\",\n";
    file << "    \"kernel\": \"" << result.environment.kernel << "\",\n";
    file << "    \"os\": \"" << result.environment.os << "\",\n";
    file << "    \"cpu\": \"" << result.environment.cpu << "\",\n";
    file << "    \"compiler\": \"" << result.environment.compiler << "\",\n";
    file << "    \"cmake\": \"" << result.environment.cmake << "\",\n";
    file << "    \"cuda_toolkit\": \"" << result.environment.cuda_toolkit << "\",\n";
    file << "    \"nvidia_driver\": \"" << result.environment.nvidia_driver << "\",\n";
    file << "    \"gpu\": \"" << result.environment.gpu << "\"\n";

    file << "  },\n";

    file << "  \"measurements\": [\n";

    for (std::size_t i = 0;
         i < result.measurements.size();
         ++i)
    {
        const auto& measurement =
            result.measurements[i];

        file << "    {\n";

        file << "      \"name\": \""
             << measurement.name
             << "\",\n";

        file << "      \"unit\": \""
             << measurement.unit
             << "\",\n";

        file << "      \"value\": "
             << measurement.value
             << "\n";

        file << "    }";

        if (i + 1 < result.measurements.size())
        {
            file << ",";
        }

        file << "\n";
    }

    file << "  ]\n";
    file << "}\n";
}

std::string make_timestamp()
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);

    std::tm local_time{};

    localtime_r(&time, &local_time);

    std::ostringstream timestamp;

    timestamp << std::put_time(
        &local_time,
        "%Y%m%d_%H%M%S");

    return timestamp.str();
}

double read_metric(
    const std::string& json,
    const std::string& metric_name)
{
    std::string search =
        "\"name\": \"" + metric_name + "\"";

    std::size_t name_position =
        json.find(search);

    if (name_position == std::string::npos)
    {
        throw std::runtime_error(
            "Metric not found: " + metric_name);
    }

    std::size_t value_position =
        json.find("\"value\":", name_position);

    if (value_position == std::string::npos)
    {
        throw std::runtime_error(
            "Value not found for metric: " + metric_name);
    }

    value_position += std::string("\"value\":").size();

    return std::stod(
        json.substr(value_position));
}

double get_measurement(
    const benchkit::BenchmarkResult& result,
    const std::string& name)
{
    for (const auto& measurement : result.measurements)
    {
        if (measurement.name == name)
        {
            return measurement.value;
        }
    }

    throw std::runtime_error(
        "Measurement not found: " + name);
}

double percentage_change(
    double old_value,
    double new_value)
{
    if (old_value == 0.0)
    {
        throw std::runtime_error(
            "Cannot calculate percentage change from zero");
    }

    return ((new_value - old_value) / old_value) * 100.0;
}

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

double tail_ratio(
    const benchkit::BenchmarkResult& result)
{
    double median = get_measurement(result, "median");
    double p95 = get_measurement(result, "p95");

    if (median <= 0.0)
    {
        throw std::runtime_error(
            "Cannot calculate tail ratio with non-positive median");
    }

    return p95 / median;
}


int main(int argc, char* argv[])
{
if (argc < 2)
{
    std::cerr << "Usage: benchkit <command> [arguments]\n";
    std::cerr << "Commands: run, compare\n";
    return 1;
}

std::string command = argv[1];

if (command == "run")
{
    constexpr int samples = 20;
    constexpr std::size_t elements = 1'000'000;

    std::vector<double> timings;
    timings.reserve(samples);

    for (int i = 0; i < samples; ++i)
    {
        auto benchmark =
            benchkit::run_vector_add(elements);

        if (!benchmark.correct)
        {
            std::cerr
                << "Benchmark correctness check FAILED\n";

            return 1;
        }

        timings.push_back(benchmark.elapsed_ms);
    }

    double min_time =
        *std::min_element(timings.begin(), timings.end());

    double max_time =
        *std::max_element(timings.begin(), timings.end());

    double sum =
        std::accumulate(
            timings.begin(),
            timings.end(),
            0.0);

    double mean =
        sum / timings.size();

    double squared_diff_sum = 0.0;

    for (double value : timings)
    {
        double difference = value - mean;

        squared_diff_sum +=
            difference * difference;
    }

    double variance =
        squared_diff_sum /
        (timings.size() - 1);

    double standard_deviation =
        std::sqrt(variance);

    auto environment =
        benchkit::collect_environment();

    double median = percentile(timings, 50.0);
    double p95 = percentile(timings, 95.0);
    double p99 = percentile(timings, 99.0);

    benchkit::BenchmarkResult result{
        .benchmark = "cpu_vector_add",
        .status = "pass",
        .environment = environment,
        .sample_count = samples,
        .raw_samples = timings,
        .measurements = {
        {"min", "ms", min_time},
        {"median", "ms", median},
        {"mean", "ms", mean},
        {"p95", "ms", p95},
        {"p99", "ms", p99},
        {"max", "ms", max_time},
        {"stddev", "ms", standard_deviation}
        }
    };

    print_result_json(result);

    std::string output_path =
        "results/raw/cpu_vector_add_" +
        make_timestamp() +
        ".json";

    write_result_json(result, output_path);

    std::cout << "Raw result: "
              << output_path
              << "\n";
}
else if (command == "compare")
{
    if (argc != 4)
    {
        std::cerr
            << "Usage: benchkit compare <file_a> <file_b>\n";

        return 1;
    }

    std::string file_a = argv[2];
    std::string file_b = argv[3];

    try
    {
        auto result_a =
            benchkit::load_result(file_a);

        auto result_b =
            benchkit::load_result(file_b);

        if (result_a.benchmark != result_b.benchmark)
        {
            std::cerr
                << "Cannot compare different benchmarks:\n"
                << "  A: " << result_a.benchmark << "\n"
                << "  B: " << result_b.benchmark << "\n";

            return 1;
        }

        const std::vector<std::string> metrics = {
            "min",
            "median",
            "mean",
            "p95",
            "p99",
            "max",
            "stddev"
        };

        std::cout << "\nBenchmark comparison\n";
        std::cout << "--------------------\n";
        std::cout << "Benchmark: "
                  << result_a.benchmark
                  << "\n\n";

        std::cout
            << "Metric       Run A       Run B       Change\n";

        std::cout
            << "--------------------------------------------\n";

        for (const auto& metric : metrics)
    {
        double value_a =
            get_measurement(result_a, metric);

        double value_b =
            get_measurement(result_b, metric);

        double change =
            percentage_change(value_a, value_b);

        std::cout
            << std::left
            << std::setw(12)
            << metric
            << std::right
            << std::setw(10)
            << value_a
            << std::setw(12)
            << value_b
            << std::setw(11)
            << change
            << "%\n";
    }

        double tail_ratio_a = tail_ratio(result_a);
        double tail_ratio_b = tail_ratio(result_b);

        double tail_ratio_change =
            percentage_change(tail_ratio_a, tail_ratio_b);

        std::cout << "\nTail latency\n";
        std::cout << "------------\n";

        std::cout
            << "P95 / median   "
            << "Run A: " << tail_ratio_a << "x"
            << "   Run B: " << tail_ratio_b << "x"
            << "   Change: " << tail_ratio_change << "%\n";
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "Compare failed: "
            << error.what()
            << "\n";

        return 1;
    }
}

    return 0;
}