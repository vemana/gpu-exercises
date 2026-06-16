#include "reference_kernel.h"
#include <cuda_runtime.h>
#include <math.h>
#include <vector>
#include "../utils/utils.h"
#include "../utils/tracer.h"

// Highly optimized N-body reference kernel using shared memory tiling and loop unrolling
__global__ void reference_nbody_kernel(const float* pos_x, const float* pos_y, const float* pos_z, const float* mass, 
                                       const float* vel_x_in, const float* vel_y_in, const float* vel_z_in,
                                       float* vel_x_out, float* vel_y_out, float* vel_z_out, long long num_bodies, float dt) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use shared memory for tiling
    extern __shared__ float4 sh_pos_mass[]; 
    
    float my_x, my_y, my_z, my_mass;
    if (i < num_bodies) {
        my_x = pos_x[i];
        my_y = pos_y[i];
        my_z = pos_z[i];
        my_mass = mass[i];
    }
    
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    const float G = 6.67430e-11f;
    
    for (long long tile = 0; tile < (num_bodies + blockDim.x - 1) / blockDim.x; ++tile) {
        long long tile_idx = tile * blockDim.x + threadIdx.x;
        if (tile_idx < num_bodies) {
            sh_pos_mass[threadIdx.x] = make_float4(pos_x[tile_idx], pos_y[tile_idx], pos_z[tile_idx], mass[tile_idx]);
        } else {
            sh_pos_mass[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        __syncthreads();
        
        #pragma unroll 16
        for (int j = 0; j < blockDim.x; ++j) {
            float4 other = sh_pos_mass[j];
            float dx = other.x - my_x;
            float dy = other.y - my_y;
            float dz = other.z - my_z;
            float dist_sq = dx*dx + dy*dy + dz*dz + 1e-9f;
            float inv_dist = rsqrtf(dist_sq);
            float inv_dist3 = inv_dist * inv_dist * inv_dist;
            float f = G * my_mass * other.w * inv_dist3;
            if (tile * blockDim.x + j != i) {
                fx += f * dx;
                fy += f * dy;
                fz += f * dz;
            }
        }
        __syncthreads();
    }
    
    if (i < num_bodies) {
        vel_x_out[i] = vel_x_in[i] + (fx / my_mass) * dt;
        vel_y_out[i] = vel_y_in[i] + (fy / my_mass) * dt;
        vel_z_out[i] = vel_z_in[i] + (fz / my_mass) * dt;
    }
}

std::vector<LaunchConfig> launch_reference_nbody(const float* pos_x, const float* pos_y, const float* pos_z, const float* mass,
                                     const float* vel_x_in, const float* vel_y_in, const float* vel_z_in,
                                     float* vel_x_out, float* vel_y_out, float* vel_z_out, long long num_bodies, float dt) {
    global_tracer.trace("Entering launch_reference_nbody (Tiled Version)");

    int threadsPerBlock = 256;
    long long blocksPerGrid = (num_bodies + threadsPerBlock - 1) / threadsPerBlock;

    size_t smemSize = threadsPerBlock * sizeof(float4);

    global_tracer.trace("Launching reference_nbody_kernel");
    reference_nbody_kernel<<<blocksPerGrid, threadsPerBlock, smemSize>>>(
        pos_x, pos_y, pos_z, mass,
        vel_x_in, vel_y_in, vel_z_in,
        vel_x_out, vel_y_out, vel_z_out,
        num_bodies, dt
    );

    global_tracer.trace("Exiting launch_reference_nbody");
    return {{"reference_nbody_kernel", (const void*)reference_nbody_kernel, (long long)blocksPerGrid, threadsPerBlock, smemSize}};
}
