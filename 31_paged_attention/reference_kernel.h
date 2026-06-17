#pragma once

#include <vector>

#include "../utils/framework.h"

std::vector<LaunchConfig> launch_paged_attention_reference(
    const float* d_Q, const float* d_K_cache, const float* d_V_cache,
    const int* d_block_tables, const int* d_context_lens, float* d_Out,
    int num_seqs, int head_dim, int block_size, int max_blocks_per_seq);
