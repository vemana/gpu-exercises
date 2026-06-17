#pragma once
#include "framework.h"
#include <cuComplex.h>

std::vector<LaunchConfig> launch_fft_reference(cuFloatComplex* d_x, int N);
