

#include <vector>

#include "../utils/utils.h"

#ifndef KERNEL_H
#define KERNEL_H

std::vector<LaunchConfig> launch_reduce(const float* a, float* c, long long size);

#endif
