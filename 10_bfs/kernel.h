#pragma once
#include "../utils/utils.h"

LaunchMetrics launch_bfs(const int* row_offsets, const int* col_indices, int* distances, int num_nodes, int num_edges, int source_node);
