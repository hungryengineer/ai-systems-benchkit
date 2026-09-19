#pragma once

#include <vector>

namespace benchkit {

std::vector<double> run_cuda_matmul();
std::vector<double> run_cuda_matmul_tiled();
std::vector<double> run_cuda_matmul_cublas();

}