#pragma once
#include <vector>
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_reference_im2col(const float* input, float* output, long long C, long long H, long long W);
