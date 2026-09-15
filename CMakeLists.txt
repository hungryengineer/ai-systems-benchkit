cmake_minimum_required(VERSION 3.20)

project(ai_systems_benchkit
    VERSION 0.1.0
    DESCRIPTION "Reproducible C++/CUDA benchmarking framework for AI infrastructure"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

add_executable(
    benchkit
    src/main.cpp
    src/environment.cpp
    src/json.cpp
    src/cpu_benchmarks.cpp
)

target_include_directories(benchkit
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(benchkit
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

enable_testing()

add_executable(benchkit_result_test
    tests/test_result.cpp
    src/environment.cpp
)

target_include_directories(benchkit_result_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(benchkit_result_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_test(
    NAME result_test
    COMMAND benchkit_result_test
)