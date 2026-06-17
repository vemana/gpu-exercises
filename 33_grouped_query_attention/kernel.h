#pragma once

#include <vector>

#include "../utils/framework.h"

std::vector<LaunchConfig> launch_gqa(
    const float* d_Q, const float* d_K, const float* d_V, float* d_Out,
    int seq_len, int num_q_heads, int num_kv_heads, int head_dim);
