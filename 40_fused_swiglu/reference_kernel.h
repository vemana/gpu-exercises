#pragma once

#include <vector>

#include "../utils/framework.h"

std::vector<LaunchConfig> launch_reference_fused_swiglu(
    const float* X, const float* gate, float* O, int total_elements);
