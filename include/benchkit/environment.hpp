#pragma once

#include <string>

namespace benchkit {

struct EnvironmentInfo {
    std::string hostname;
    std::string kernel;
    std::string os;
    std::string cpu;
    std::string compiler;
    std::string cmake;
    std::string cuda_toolkit;
    std::string nvidia_driver;
    std::string gpu;
};

EnvironmentInfo collect_environment();

} // namespace benchkit