#pragma once

#include <vector>

#include "../utils/framework.h"

std::vector<LaunchConfig> launch_split_k_flash_decoding(
    const float* q, const float* k, const float* v, float* o,
    float* partial_o, float* partial_m, float* partial_l,
    int S, int D, int split_size);
