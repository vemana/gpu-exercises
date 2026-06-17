#pragma once

#include <vector>

#include "../utils/framework.h"

std::vector<LaunchConfig> launch_moe_routing(
    const float* d_logits, int* d_top_experts, float* d_routing_weights, 
    int num_tokens, int num_experts, int K);
