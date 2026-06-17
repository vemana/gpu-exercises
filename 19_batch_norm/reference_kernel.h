#pragma once

#include <vector>

#include "../utils/utils.h"

std::vector<LaunchConfig> launch_reference_batch_norm(const float* input, float* output, long long N, long long C, long long H, long long W);
