#pragma once

#include <vector>

#include "../utils/framework.h"

std::vector<LaunchConfig> launch_parallel_associative_scan(
    const float* A, const float* B, const float* x, float* h, int N);
