

#include "kernel.h"

#include <vector>

#include "../utils/tracer.h"
#include "../utils/utils.h"

__global__ void nbody_kernel(const float* pos_x, const float* pos_y, const float* pos_z, const float* mass, 
                             const float* vel_x_in, const float* vel_y_in, const float* vel_z_in,
                             float* vel_x_out, float* vel_y_out, float* vel_z_out, long long num_bodies, float dt) {
    // TODO: Implement N-Body simulation
}

std::vector<LaunchConfig> launch_nbody(const float* pos_x, const float* pos_y, const float* pos_z, const float* mass,
                           const float* vel_x_in, const float* vel_y_in, const float* vel_z_in,
                           float* vel_x_out, float* vel_y_out, float* vel_z_out, long long num_bodies, float dt) {
    global_tracer.trace("Entering launch_nbody (Student)");

    // TODO: Define grid and block dimensions
    int threadsPerBlock = 256;
    long long blocksPerGrid = (num_bodies + threadsPerBlock - 1) / threadsPerBlock;

    global_tracer.trace("Launching nbody_kernel (Student)");
    // nbody_kernel<<<blocksPerGrid, threadsPerBlock>>>(pos_x, pos_y, pos_z, mass, vel_x_in, vel_y_in, vel_z_in, vel_x_out, vel_y_out, vel_z_out, num_bodies, dt);

    global_tracer.trace("Exiting launch_nbody (Student)");
    return {{"nbody_kernel", (const void*)nbody_kernel, (long long)blocksPerGrid, threadsPerBlock, 0}};
}
