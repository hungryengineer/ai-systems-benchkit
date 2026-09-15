#include "benchkit/environment.hpp"

#include <array>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace benchkit {

namespace {

std::string trim(const std::string& value)
{
    const auto start = value.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }

    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

std::string run_command(const std::string& command)
{
    std::array<char, 256> buffer{};
    std::string result;

    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        return "unavailable";
    }

    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        result = result + buffer.data();
        
    }

    pclose(pipe);

    return trim(result);
}

std::string read_first_matching_line(
    const std::string& path,
    const std::string& key)
{
    std::ifstream file(path);

    if (!file.is_open()) {
        return "unavailable";
    }

    std::string line;

    while (std::getline(file, line)) {
        if (line.rfind(key, 0) == 0) {
            const auto separator = line.find(':');

            if (separator != std::string::npos) {
                return trim(line.substr(separator + 1));
            }

            return trim(line);
        }
    }

    return "unavailable";
}

} // namespace

EnvironmentInfo collect_environment()
{
    EnvironmentInfo env;

    env.hostname = run_command("hostname");
    env.kernel = run_command("uname -r");

    env.os = run_command(
        "grep '^PRETTY_NAME=' /etc/os-release | cut -d= -f2- | tr -d '\"'"
    );

    env.cpu = read_first_matching_line(
        "/proc/cpuinfo",
        "model name"
    );

    env.compiler = run_command("g++ --version | head -n 1");
    env.cmake = run_command("cmake --version | head -n 1");

    env.cuda_toolkit = run_command("nvcc --version | tail -n 1");

    env.nvidia_driver = run_command(
        "nvidia-smi --query-gpu=driver_version "
        "--format=csv,noheader | head -n 1"
    );

    env.gpu = run_command(
        "nvidia-smi --query-gpu=name "
        "--format=csv,noheader | head -n 1"
    );

    return env;
}

} // namespace benchkit