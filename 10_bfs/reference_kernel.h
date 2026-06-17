#pragma once

#include <vector>

#include "../utils/utils.h"

std::vector<LaunchConfig> launch_reference_bfs(const int* row_offsets, const int* col_indices, int* distances, long long num_nodes, long long num_edges, int source_node);
