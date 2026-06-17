#pragma once
#include "framework.h"

std::vector<LaunchConfig> launch_cg_reference(
    const int* d_row_ptr, const int* d_col_ind, const float* d_values, 
    const float* d_b, float* d_x, int N, int nnz, int num_iters);
