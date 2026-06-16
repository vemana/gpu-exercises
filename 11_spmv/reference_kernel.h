#pragma once
#include "../utils/utils.h"

LaunchMetrics launch_reference_spmv(const float* values, const int* col_indices, const int* row_offsets, const float* x, float* y, int num_rows);
