#pragma once
#include <vector>
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_stencil(const float* a, float* c, int size, int radius);
