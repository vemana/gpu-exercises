#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "../utils/framework.h"
#include "../utils/argparse.h"
#include "kernel.h"
#include "reference_kernel.h"

// Note: HPC applications typically use double precision.
// We use float here for consistency across the repository exercises.

void cpu_baseline(const float* u_initial, float* u_final, int H, int W, int num_iters) {
    std::vector<float> current(u_initial, u_initial + H * W);
    std::vector<float> next = current;

    float* curr_ptr = current.data();
    float* next_ptr = next.data();

    for (int iter = 0; iter < num_iters; ++iter) {
        for (int y = 1; y < H - 1; ++y) {
            for (int x = 1; x < W - 1; ++x) {
                next_ptr[y * W + x] = 0.25f * (curr_ptr[(y - 1) * W + x] + curr_ptr[(y + 1) * W + x] +
                                               curr_ptr[y * W + x - 1] + curr_ptr[y * W + x + 1]);
            }
        }
        std::swap(curr_ptr, next_ptr);
    }
    
    std::copy(curr_ptr, curr_ptr + H * W, u_final);
}

const int NUM_ITERS = 100;

struct JacobiTest : public ProblemTest<2> {
    std::vector<float> h_u_initial;
    std::vector<float> h_u;
    std::vector<float> h_u_ref;

    float *d_u = nullptr;
    float *d_u_tmp = nullptr;

    JacobiTest(const TestSize<2>& size) : ProblemTest<2>(size) {}

    void generate_test_data(bool check) override {
        int H = size.dims[0];
        int W = size.dims[1];
        int N = H * W;
        h_u_initial.resize(N, 0.0f);
        h_u.assign(N, 0.0f);
        h_u_ref.assign(N, 0.0f);

        for (int i = 0; i < H; ++i) {
            h_u_initial[i * W] = 100.0f;
        }
        for (int j = 0; j < W; ++j) {
            h_u_initial[j] = 100.0f;
        }

        if (check) cpu_baseline(h_u_initial.data(), h_u_ref.data(), H, W, NUM_ITERS);
    }

    void setup_reference() override {
        int N = size.dims[0] * size.dims[1];
        cudaMalloc(&d_u, N * sizeof(float));
        cudaMalloc(&d_u_tmp, N * sizeof(float));
        cudaMemcpy(d_u, h_u_initial.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_u_tmp, h_u_initial.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_jacobi_reference(d_u, d_u_tmp, size.dims[0], size.dims[1], NUM_ITERS);
    }

    void teardown_reference() override {
        cudaFree(d_u);
        cudaFree(d_u_tmp);
    }

    void setup_student() override {
        int N = size.dims[0] * size.dims[1];
        cudaMalloc(&d_u, N * sizeof(float));
        cudaMalloc(&d_u_tmp, N * sizeof(float));
        cudaMemcpy(d_u, h_u_initial.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_u_tmp, h_u_initial.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_jacobi(d_u, d_u_tmp, size.dims[0], size.dims[1], NUM_ITERS);
    }

    void teardown_student() override {
        cudaFree(d_u);
        cudaFree(d_u_tmp);
    }

    CorrectnessResult verify() override {
        int N = size.dims[0] * size.dims[1];
        cudaMemcpy(h_u.data(), d_u, N * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_u_ref.data(), h_u.data(), N, 1e-3f);
    }

    void print_mismatch() override {
        int N = size.dims[0] * size.dims[1];
        print_array("Initial Grid", h_u_initial.data(), N);
        print_array("Expected Grid", h_u_ref.data(), N);
        print_array("Actual Grid", h_u.data(), N);
    }

    double get_bandwidth_bytes() override {
        // Each iteration reads H*W elements and writes H*W elements, doing it NUM_ITERS times
        return 2.0 * NUM_ITERS * size.dims[0] * size.dims[1] * sizeof(float);
    }

    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<2> config = parse_args<2>(argc, argv);

    std::vector<TestSize<2>> correctness_sizes = {
        {{16, 16}}, {{17, 17}}, {{64, 64}}, {{65, 63}}
    };
    
    std::vector<TestSize<2>> perf_sizes = {
        {{256, 256}}, {{1024, 1024}}, {{2048, 2048}}
    };

    run_test_suite<2, JacobiTest>("Exercise 26: Jacobi Iteration", config, correctness_sizes, perf_sizes);
    return 0;
}
