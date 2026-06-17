#pragma once
#include "framework.h"
#include <cuComplex.h>

std::vector<LaunchConfig> launch_fft(cuFloatComplex* d_x, int N);
