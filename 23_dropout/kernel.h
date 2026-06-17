#pragma once

#include <vector>

#include "../utils/utils.h"

std::vector<LaunchConfig> launch_dropout(const float* input, float* output, long long size, float drop_prob);
