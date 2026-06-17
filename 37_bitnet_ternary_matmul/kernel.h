#pragma once

#include <cstdint>
#include <vector>

#include "../utils/framework.h"

std::vector<LaunchConfig> launch_bitnet_ternary_matmul(
    const float* A, const uint8_t* W_packed, float* C,
    int M, int N, int K);
