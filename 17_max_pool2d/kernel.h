#pragma once
#include <vector>
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_max_pool2d(const float* input, float* output, long long channels, long long height, long long width);
