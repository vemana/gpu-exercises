#pragma once
#include <vector>
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_nbody(const float* pos_x, const float* pos_y, const float* pos_z, const float* mass, 
                           const float* vel_x_in, const float* vel_y_in, const float* vel_z_in,
                           float* vel_x_out, float* vel_y_out, float* vel_z_out, int num_bodies, float dt);
