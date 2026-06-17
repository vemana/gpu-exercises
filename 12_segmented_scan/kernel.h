#pragma once

#include <vector>

#include "../utils/utils.h"

std::vector<LaunchConfig> launch_segmented_scan(const float* a, const int* flags, float* c, long long size);
