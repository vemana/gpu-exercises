#pragma once
#include <vector>
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_conv2d(const float* a, const float* filter, float* c, int width, int height);
