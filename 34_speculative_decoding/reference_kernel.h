#pragma once

#include <vector>

#include "../utils/framework.h"

std::vector<LaunchConfig> launch_speculative_decoding_reference(
    const float* d_target_probs, const float* d_draft_probs, const float* d_random_vals,
    int* d_accept_lengths, int num_batches, int draft_len);
