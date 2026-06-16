#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include "../utils/framework.h"
#include "../utils/argparse.h"
#include "../utils/tracer.h"
#include "kernel.h"
#include "reference_kernel.h"

void cpu_baseline(const float* pos_x, const float* pos_y, const float* pos_z, const float* mass, 
                  const float* vel_x_in, const float* vel_y_in, const float* vel_z_in,
                  float* vel_x_out, float* vel_y_out, float* vel_z_out, int num_bodies, float dt) {
    const float G = 6.67430e-11f;
    for (int i = 0; i < num_bodies; ++i) {
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;
        for (int j = 0; j < num_bodies; ++j) {
            if (i != j) {
                float dx = pos_x[j] - pos_x[i];
                float dy = pos_y[j] - pos_y[i];
                float dz = pos_z[j] - pos_z[i];
                float dist_sq = dx*dx + dy*dy + dz*dz + 1e-9f; // Softening
                float inv_dist = 1.0f / std::sqrt(dist_sq);
                float inv_dist3 = inv_dist * inv_dist * inv_dist;
                float f = G * mass[i] * mass[j] * inv_dist3;
                fx += f * dx;
                fy += f * dy;
                fz += f * dz;
            }
        }
        vel_x_out[i] = vel_x_in[i] + (fx / mass[i]) * dt;
        vel_y_out[i] = vel_y_in[i] + (fy / mass[i]) * dt;
        vel_z_out[i] = vel_z_in[i] + (fz / mass[i]) * dt;
    }
}

struct NBodyTest : public ProblemTest<1> {
    std::vector<float> h_pos_x, h_pos_y, h_pos_z, h_mass;
    std::vector<float> h_vel_x_in, h_vel_y_in, h_vel_z_in;
    std::vector<float> h_vel_x_out, h_vel_y_out, h_vel_z_out;
    std::vector<float> h_vel_x_ref, h_vel_y_ref, h_vel_z_ref;

    float *d_pos_x = nullptr, *d_pos_y = nullptr, *d_pos_z = nullptr, *d_mass = nullptr;
    float *d_vel_x_in = nullptr, *d_vel_y_in = nullptr, *d_vel_z_in = nullptr;
    float *d_vel_x_out = nullptr, *d_vel_y_out = nullptr, *d_vel_z_out = nullptr;
    float dt = 0.01f;

    NBodyTest(const TestSize<1>& size) : ProblemTest<1>(size) {}

    void generate_test_data(bool check) override {
        int n = size.dims[0];
        h_pos_x.resize(n); h_pos_y.resize(n); h_pos_z.resize(n); h_mass.resize(n);
        h_vel_x_in.resize(n); h_vel_y_in.resize(n); h_vel_z_in.resize(n);
        h_vel_x_out.assign(n, 0); h_vel_y_out.assign(n, 0); h_vel_z_out.assign(n, 0);
        h_vel_x_ref.assign(n, 0); h_vel_y_ref.assign(n, 0); h_vel_z_ref.assign(n, 0);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> pos_dist(-1e3f, 1e3f);
        std::uniform_real_distribution<float> vel_dist(-10.0f, 10.0f);
        std::uniform_real_distribution<float> mass_dist(1e12f, 1e15f);
        
        for (int i = 0; i < n; ++i) {
            h_pos_x[i] = pos_dist(gen);
            h_pos_y[i] = pos_dist(gen);
            h_pos_z[i] = pos_dist(gen);
            h_mass[i] = mass_dist(gen);
            h_vel_x_in[i] = vel_dist(gen);
            h_vel_y_in[i] = vel_dist(gen);
            h_vel_z_in[i] = vel_dist(gen);
        }
        
        if (check) {
            cpu_baseline(h_pos_x.data(), h_pos_y.data(), h_pos_z.data(), h_mass.data(),
                         h_vel_x_in.data(), h_vel_y_in.data(), h_vel_z_in.data(),
                         h_vel_x_ref.data(), h_vel_y_ref.data(), h_vel_z_ref.data(), n, dt);
        }
    }

    void setup_reference() override {
        int n = size.dims[0];
        cudaMalloc(&d_pos_x, n * sizeof(float)); cudaMalloc(&d_pos_y, n * sizeof(float)); cudaMalloc(&d_pos_z, n * sizeof(float)); cudaMalloc(&d_mass, n * sizeof(float));
        cudaMalloc(&d_vel_x_in, n * sizeof(float)); cudaMalloc(&d_vel_y_in, n * sizeof(float)); cudaMalloc(&d_vel_z_in, n * sizeof(float));
        cudaMalloc(&d_vel_x_out, n * sizeof(float)); cudaMalloc(&d_vel_y_out, n * sizeof(float)); cudaMalloc(&d_vel_z_out, n * sizeof(float));
        
        cudaMemcpy(d_pos_x, h_pos_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos_y, h_pos_y.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos_z, h_pos_z.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mass, h_mass.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel_x_in, h_vel_x_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel_y_in, h_vel_y_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel_z_in, h_vel_z_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        
        cudaMemset(d_vel_x_out, 0, n * sizeof(float));
        cudaMemset(d_vel_y_out, 0, n * sizeof(float));
        cudaMemset(d_vel_z_out, 0, n * sizeof(float));
    }

    LaunchMetrics launch_reference() override {
        return launch_reference_nbody(d_pos_x, d_pos_y, d_pos_z, d_mass, d_vel_x_in, d_vel_y_in, d_vel_z_in, d_vel_x_out, d_vel_y_out, d_vel_z_out, size.dims[0], dt);
    }

    void setup_student() override {
        int n = size.dims[0];
        cudaMalloc(&d_pos_x, n * sizeof(float)); cudaMalloc(&d_pos_y, n * sizeof(float)); cudaMalloc(&d_pos_z, n * sizeof(float)); cudaMalloc(&d_mass, n * sizeof(float));
        cudaMalloc(&d_vel_x_in, n * sizeof(float)); cudaMalloc(&d_vel_y_in, n * sizeof(float)); cudaMalloc(&d_vel_z_in, n * sizeof(float));
        cudaMalloc(&d_vel_x_out, n * sizeof(float)); cudaMalloc(&d_vel_y_out, n * sizeof(float)); cudaMalloc(&d_vel_z_out, n * sizeof(float));
        
        cudaMemcpy(d_pos_x, h_pos_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos_y, h_pos_y.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos_z, h_pos_z.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mass, h_mass.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel_x_in, h_vel_x_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel_y_in, h_vel_y_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel_z_in, h_vel_z_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        
        cudaMemset(d_vel_x_out, 0, n * sizeof(float));
        cudaMemset(d_vel_y_out, 0, n * sizeof(float));
        cudaMemset(d_vel_z_out, 0, n * sizeof(float));
    }

    LaunchMetrics launch_student() override {
        return launch_nbody(d_pos_x, d_pos_y, d_pos_z, d_mass, d_vel_x_in, d_vel_y_in, d_vel_z_in, d_vel_x_out, d_vel_y_out, d_vel_z_out, size.dims[0], dt);
    }

    CorrectnessResult verify() override {
        int n = size.dims[0];
        cudaMemcpy(h_vel_x_out.data(), d_vel_x_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vel_y_out.data(), d_vel_y_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vel_z_out.data(), d_vel_z_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        
        CorrectnessResult res_x = check_correctness(h_vel_x_ref.data(), h_vel_x_out.data(), n, 1e-2f);
        CorrectnessResult res_y = check_correctness(h_vel_y_ref.data(), h_vel_y_out.data(), n, 1e-2f);
        CorrectnessResult res_z = check_correctness(h_vel_z_ref.data(), h_vel_z_out.data(), n, 1e-2f);
        
        if (!res_x.passed) return res_x;
        if (!res_y.passed) return res_y;
        return res_z;
    }

    void print_mismatch() override {
        int n = size.dims[0];
        std::cout << "\n--- Expected Output (First 10) [X, Y, Z] ---\n";
        for (int i = 0; i < std::min(n, 10); ++i) {
            std::cout << "[" << std::fixed << std::setprecision(4) << h_vel_x_ref[i] << ", "
                      << h_vel_y_ref[i] << ", " << h_vel_z_ref[i] << "]\n";
        }
        std::cout << "\n--- Actual Output (First 10) [X, Y, Z] ---\n";
        for (int i = 0; i < std::min(n, 10); ++i) {
            std::cout << "[" << std::fixed << std::setprecision(4) << h_vel_x_out[i] << ", "
                      << h_vel_y_out[i] << ", " << h_vel_z_out[i] << "]\n";
        }
        std::cout << "\n";
    }

    double get_bandwidth_bytes() override {
        // N-body bandwidth isn't easily defined due to O(N^2) data reuse
        // But useful memory moved per iteration is reading state, writing state:
        int n = size.dims[0];
        return n * 7 * sizeof(float) + n * 3 * sizeof(float); // Read pos(3), mass(1), vel(3); Write vel(3)
    }

    void teardown() override {
        if (d_pos_x) cudaFree(d_pos_x);
        if (d_pos_y) cudaFree(d_pos_y);
        if (d_pos_z) cudaFree(d_pos_z);
        if (d_mass) cudaFree(d_mass);
        if (d_vel_x_in) cudaFree(d_vel_x_in);
        if (d_vel_y_in) cudaFree(d_vel_y_in);
        if (d_vel_z_in) cudaFree(d_vel_z_in);
        if (d_vel_x_out) cudaFree(d_vel_x_out);
        if (d_vel_y_out) cudaFree(d_vel_y_out);
        if (d_vel_z_out) cudaFree(d_vel_z_out);
    }
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);

    // O(N^2) complexity, keep correctness sizes relatively small
    std::vector<TestSize<1>> correctness_sizes = {
        {1}, {31}, {32}, {33}, {63}, {64}, {65}, {255}, {256}, {257}, {1024}
    };
    
    std::vector<TestSize<1>> perf_sizes = {
        {16384}, {32768}, {65536}
    };

    run_test_suite<1, NBodyTest>("Exercise 14: N-Body Simulation", config, correctness_sizes, perf_sizes);
    return 0;
}
