#pragma once
#include <vector>
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_reference_spmv(const float* values, const int* col_indices, const int* row_offsets, const float* x, float* y, long long num_rows);
