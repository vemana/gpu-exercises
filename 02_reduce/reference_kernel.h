

#include <vector>

#include "../utils/utils.h"

#ifndef REFERENCE_KERNEL_H
#define REFERENCE_KERNEL_H

std::vector<LaunchConfig> launch_reference_reduce(const float* a, float* c, long long size);

#endif
