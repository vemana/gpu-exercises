#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cuda_runtime.h>
#include "../utils/framework.h"
#include "../utils/argparse.h"
#include "kernel.h"
#include "reference_kernel.h"

// Note: HPC applications typically use double precision.
// We use float here for consistency across the repository exercises.

void cpu_md(const std::vector<float2>& pos, std::vector<float2>& forces, int N, float cutoff_radius) {
    float cutoff_sq = cutoff_radius * cutoff_radius;
    for (int i = 0; i < N; ++i) {
        float2 f_i = make_float2(0.0f, 0.0f);
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                float dx = pos[i].x - pos[j].x;
                float dy = pos[i].y - pos[j].y;
                float r2 = dx*dx + dy*dy;
                
                if (r2 < cutoff_sq && r2 > 0.001f) {
                    double r2_d = r2;
                    double r2_inv = 1.0 / r2_d;
                    double r6_inv = r2_inv * r2_inv * r2_inv;
                    double f_mag = 24.0 * r6_inv * (2.0 * r6_inv - 1.0) * r2_inv;
                    
                    f_i.x += (float)(f_mag * dx);
                    f_i.y += (float)(f_mag * dy);
                }
            }
        }
        forces[i] = f_i;
    }
}

const float DOMAIN_SIZE = 100.0f;
const float CUTOFF_RADIUS = 2.5f;

struct MDTest : public ProblemTest<1> {
    std::vector<float2> h_pos;
    std::vector<float2> h_forces;
    std::vector<float2> h_forces_ref;

    float2 *d_pos = nullptr;
    float2 *d_forces = nullptr;

    MDTest(const TestSize<1>& size) : ProblemTest<1>(size) {}

    void generate_test_data(bool check) override {
        int N = size.dims[0];
        h_pos.resize(N);
        h_forces.resize(N, make_float2(0.0f, 0.0f));
        h_forces_ref.resize(N, make_float2(0.0f, 0.0f));
        
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(0.0f, DOMAIN_SIZE);
        
        // Try to place particles not too close to avoid extreme forces blowing up float precision
        for (int i = 0; i < N; ++i) {
            h_pos[i] = make_float2(dist(gen), dist(gen));
        }

        if (check) {
            cpu_md(h_pos, h_forces_ref, N, CUTOFF_RADIUS);
        }
    }

    void setup_reference() override {
        int N = size.dims[0];
        cudaMalloc(&d_pos, N * sizeof(float2));
        cudaMalloc(&d_forces, N * sizeof(float2));
        cudaMemcpy(d_pos, h_pos.data(), N * sizeof(float2), cudaMemcpyHostToDevice);
        cudaMemset(d_forces, 0, N * sizeof(float2));
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_md_reference(d_pos, d_forces, size.dims[0], DOMAIN_SIZE, CUTOFF_RADIUS);
    }

    void teardown_reference() override {
        cudaFree(d_pos);
        cudaFree(d_forces);
    }

    void setup_student() override {
        int N = size.dims[0];
        cudaMalloc(&d_pos, N * sizeof(float2));
        cudaMalloc(&d_forces, N * sizeof(float2));
        cudaMemcpy(d_pos, h_pos.data(), N * sizeof(float2), cudaMemcpyHostToDevice);
        cudaMemset(d_forces, 0, N * sizeof(float2));
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_md(d_pos, d_forces, size.dims[0], DOMAIN_SIZE, CUTOFF_RADIUS);
    }

    void teardown_student() override {
        cudaFree(d_pos);
        cudaFree(d_forces);
    }

    CorrectnessResult verify() override {
        int N = size.dims[0];
        cudaMemcpy(h_forces.data(), d_forces, N * sizeof(float2), cudaMemcpyDeviceToHost);
        
        // Floating point non-associativity of addition across different atomic traversal orders
        // means we need a generous relative tolerance.
        for (int i = 0; i < N; ++i) {
            float diff_x = std::abs(h_forces_ref[i].x - h_forces[i].x);
            float diff_y = std::abs(h_forces_ref[i].y - h_forces[i].y);
            float mag_x = std::abs(h_forces_ref[i].x) + 1e-3f;
            float mag_y = std::abs(h_forces_ref[i].y) + 1e-3f;
            
            if ((diff_x > 1e-1f && diff_x / mag_x > 1e-2f) || 
                (diff_y > 1e-1f && diff_y / mag_y > 1e-2f)) {
                return {false, i, h_forces_ref[i].x, h_forces[i].x};
            }
        }
        return {true, -1, 0.0f, 0.0f};
    }

    void print_mismatch() override {
        int N = size.dims[0];
        std::cout << "--- Expected vs Actual ---" << std::endl;
        for (int i = 0; i < std::min(N, 20); ++i) {
            std::cout << "Idx " << i << ": Expected (" << h_forces_ref[i].x << ", " << h_forces_ref[i].y 
                      << ") | Actual (" << h_forces[i].x << ", " << h_forces[i].y << ")" << std::endl;
        }
    }

    double get_bandwidth_bytes() override {
        int N = size.dims[0];
        // Memory accesses:
        // Building cells: N reads (pos), N atomics (cell_starts), N writes (particle_next)
        // Computing forces: N reads (pos), N writes (forces), plus irregular lookups depending on density.
        // We'll estimate the baseline deterministic reads/writes.
        return (double)N * (sizeof(float2) * 2 + sizeof(int) * 3);
    }

    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);

    std::vector<TestSize<1>> correctness_sizes = {
        {{100}}, {{1000}}, {{5000}}
    };
    
    std::vector<TestSize<1>> perf_sizes = {
        {{10000}}, {{50000}}, {{100000}}
    };

    run_test_suite<1, MDTest>("Exercise 29: Molecular Dynamics", config, correctness_sizes, perf_sizes);
    return 0;
}
