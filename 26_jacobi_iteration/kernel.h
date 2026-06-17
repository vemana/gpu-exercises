#pragma once

#include <vector>

#include "../utils/framework.h"

std::vector<LaunchConfig> launch_jacobi(float* d_u, float* d_u_tmp, int H, int W, int num_iters);
