#pragma once
#include "framework.h"

std::vector<LaunchConfig> launch_monte_carlo_reference(
    float* d_sum, int N, float S0, float K, float r, float sigma, float T);
