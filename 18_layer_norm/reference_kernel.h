#pragma once

#include <vector>

#include "../utils/utils.h"

std::vector<LaunchConfig> launch_reference_layer_norm(const float* input, float* output, long long batch_seq, long long hidden_dim);
