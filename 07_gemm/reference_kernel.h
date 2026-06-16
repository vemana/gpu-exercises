#pragma once
#include <vector>
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_reference_gemm(const float* a, const float* b, float* c, int n);
