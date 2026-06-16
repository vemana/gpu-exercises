#ifndef REFERENCE_KERNEL_H
#define REFERENCE_KERNEL_H

#include <vector>
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_reference_reduce(const float* a, float* c, long long size);

#endif
