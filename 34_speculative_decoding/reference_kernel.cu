

#include "reference_kernel.h"

#include <vector>

#include "../utils/framework.h"

__global__ void ref_speculative_decoding_kernel(
    const float* target_probs, const float* draft_probs, const float* random_vals,
    int* accept_lengths, int num_batches, int draft_len) {
    
    // Each warp processes exactly 1 batch
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    
    if (warp_id >= num_batches) return;
    
    bool is_active = (lane_id < draft_len);
    
    bool reject = false;
    if (is_active) {
        int idx = warp_id * draft_len + lane_id;
        float p = target_probs[idx];
        float q = draft_probs[idx];
        float r = random_vals[idx];
        
        // Token is accepted if r < p / q
        // Therefore it is REJECTED if r >= p / q
        if (r >= p / q) {
            reject = true;
        }
    }
    
    // __ballot_sync gives a 32-bit integer where the n-th bit is 1 
    // if the n-th thread evaluated `reject` to true.
    // We only care about active lanes (mask = (1 << draft_len) - 1).
    unsigned int active_mask = (1u << draft_len) - 1;
    unsigned int reject_mask = __ballot_sync(active_mask, reject);
    
    if (lane_id == 0) {
        if (reject_mask == 0) {
            // All active tokens accepted
            accept_lengths[warp_id] = draft_len;
        } else {
            // __ffs(x) returns the index of the first (least significant) bit set to 1.
            // Indexing is 1-based, so subtract 1 to get the 0-based lane ID.
            int first_reject_lane = __ffs(reject_mask) - 1;
            accept_lengths[warp_id] = first_reject_lane;
        }
    }
}

std::vector<LaunchConfig> launch_speculative_decoding_reference(
    const float* d_target_probs, const float* d_draft_probs, const float* d_random_vals,
    int* d_accept_lengths, int num_batches, int draft_len) {
    
    int threads = 256;
    int warps_per_block = threads / 32;
    dim3 block(threads);
    dim3 grid((num_batches + warps_per_block - 1) / warps_per_block);
    
    ref_speculative_decoding_kernel<<<grid, block>>>(d_target_probs, d_draft_probs, d_random_vals, d_accept_lengths, num_batches, draft_len);

    return {{"ref_speculative_decoding_kernel", (const void*)ref_speculative_decoding_kernel, (long long)grid.x, (int)block.x, 0}};
}
