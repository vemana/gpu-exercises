#pragma once

#include <vector>

#include "../utils/utils.h"

std::vector<LaunchConfig> launch_reference_conv2d(const float* a, const float* filter, float* c, long long width, long long height);
