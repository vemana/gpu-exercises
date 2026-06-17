#pragma once

#include <vector>

#include "../utils/framework.h"

std::vector<LaunchConfig> launch_sliding_window_attention(
    const float* Q, const float* K, const float* V, float* O,
    int S, int D, int W);
