#pragma once

#include <vector>

#include "../utils/framework.h"

std::vector<LaunchConfig> launch_monte_carlo(
    float* d_sum, int N, float S0, float K, float r, float sigma, float T);
