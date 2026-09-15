#include "benchkit/json.hpp"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace benchkit {

BenchmarkResult load_result(const std::string& path)
{
    std::ifstream file(path);

    if (!file)
    {
        throw std::runtime_error(
            "Failed to open result file: " + path);
    }

    nlohmann::json data;

    file >> data;

    BenchmarkResult result;

    result.benchmark =
        data.at("benchmark").get<std::string>();

    result.status =
        data.at("status").get<std::string>();

    result.sample_count =
        data.at("sample_count").get<int>();

    if (data.contains("raw_samples")) {
        for (const auto& sample : data.at("raw_samples")) {
            result.raw_samples.push_back(sample.get<double>());
        }
    }

    const auto& environment =
        data.at("environment");

    result.environment.hostname =
        environment.at("hostname").get<std::string>();

    result.environment.kernel =
        environment.at("kernel").get<std::string>();

    result.environment.os =
        environment.at("os").get<std::string>();

    result.environment.cpu =
        environment.at("cpu").get<std::string>();

    result.environment.compiler =
        environment.at("compiler").get<std::string>();

    result.environment.cmake =
        environment.at("cmake").get<std::string>();

    result.environment.cuda_toolkit =
        environment.at("cuda_toolkit").get<std::string>();

    result.environment.nvidia_driver =
        environment.at("nvidia_driver").get<std::string>();

    result.environment.gpu =
        environment.at("gpu").get<std::string>();

    for (const auto& item :
         data.at("measurements"))
    {
        Measurement measurement;

        measurement.name =
            item.at("name").get<std::string>();

        measurement.unit =
            item.at("unit").get<std::string>();

        measurement.value =
            item.at("value").get<double>();

        result.measurements.push_back(measurement);
    }

    return result;
}

}