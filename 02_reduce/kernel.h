#ifndef KERNEL_H
#define KERNEL_H

#include <vector>
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_reduce(const float* a, float* c, int size);

#endif
