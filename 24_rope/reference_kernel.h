#pragma once

#include <vector>

#include "../utils/utils.h"

std::vector<LaunchConfig> launch_reference_rope(const float* input, float* output, long long batch, long long seq_len, long long hidden_dim);
