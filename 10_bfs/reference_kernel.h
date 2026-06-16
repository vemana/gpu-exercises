#pragma once
#include <vector>
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_reference_bfs(const int* row_offsets, const int* col_indices, int* distances, int num_nodes, int num_edges, int source_node);
