#pragma once
#include <vector>
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_gemm(const float* a, const float* b, float* c, long long n);
