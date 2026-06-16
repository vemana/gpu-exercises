#pragma once
#include <vector>
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_spmv(const float* values, const int* col_indices, const int* row_offsets, const float* x, float* y, int num_rows);
