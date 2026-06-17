#pragma once
#include <vector>
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_depthwise_conv2d(const float* input, const float* filter, float* output, long long C, long long H, long long W);
