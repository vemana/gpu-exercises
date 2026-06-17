#pragma once
#include <vector>
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_flash_attention(const float* q, const float* k, const float* v, float* o, long long B, long long N, long long S, long long D);
