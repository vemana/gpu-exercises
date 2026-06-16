#pragma once
#include <vector>
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_reference_stencil(const float* a, float* c, int size, int radius);
