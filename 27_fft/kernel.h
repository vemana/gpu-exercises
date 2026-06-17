#pragma once

#include <cuComplex.h>
#include <vector>

#include "../utils/framework.h"

std::vector<LaunchConfig> launch_fft(cuFloatComplex* d_x, int N);
