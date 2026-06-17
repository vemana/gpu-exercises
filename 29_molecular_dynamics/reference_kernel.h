#pragma once

#include <vector>

#include "../utils/framework.h"

std::vector<LaunchConfig> launch_md_reference(
    const float2* d_pos, float2* d_forces, 
    int N, float domain_size, float cutoff_radius);
