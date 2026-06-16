#pragma once
#include <vector>
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_gather(const float* source, float* dest, const int* indices, long long size);
