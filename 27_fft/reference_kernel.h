#pragma once

#include <cuComplex.h>
#include <vector>

#include "../utils/framework.h"

std::vector<LaunchConfig> launch_fft_reference(cuFloatComplex* d_x, int N);
