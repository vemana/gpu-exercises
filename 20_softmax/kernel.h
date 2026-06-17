#pragma once
#include <vector>
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_softmax(const float* input, float* output, long long batch_seq, long long vocab_size);
