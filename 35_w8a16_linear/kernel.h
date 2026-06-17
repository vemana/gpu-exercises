#pragma once

#include <cstdint>
#include <vector>

#include "../utils/framework.h"

std::vector<LaunchConfig> launch_w8a16_linear(
    const float* d_X, const int8_t* d_W_int8, const float* d_scales, float* d_Y,
    int M, int N, int K, int group_size);
