

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/argparse.h"
#include "../utils/framework.h"
#include "../utils/tracer.h"
#include "../utils/utils.h"
#include "kernel.h"
#include "reference_kernel.h"

void cpu_baseline(const float* a, const float* b, float* c, long long size) {
    for (long long i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

struct MapTest : public ProblemTest<1> {
    std::vector<float> h_a;
    std::vector<float> h_b;
    std::vector<float> h_c;
    std::vector<float> h_c_ref;

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    MapTest(const TestSize<1>& size) : ProblemTest<1>(size) {}

    void generate_test_data(bool check) override {
        long long n = size.dims[0];
        h_a.resize(n);
        h_b.resize(n);
        h_c.assign(n, 0.0f);
        h_c_ref.assign(n, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(0.0f, 100.0f);
        for (long long i = 0; i < n; ++i) {
            h_a[i] = dist(gen);
            h_b[i] = dist(gen);
        }
        if (check) cpu_baseline(h_a.data(), h_b.data(), h_c_ref.data(), n);
    }

    void setup_reference() override {
        long long n = size.dims[0];
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_b, n * sizeof(float));
        cudaMalloc(&d_c, n * sizeof(float));
        cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_c, 0, n * sizeof(float));
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_map(d_a, d_b, d_c, size.dims[0]);
    }

    void teardown_reference() override {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    void setup_student() override {
        long long n = size.dims[0];
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_b, n * sizeof(float));
        cudaMalloc(&d_c, n * sizeof(float));
        cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_c, 0, n * sizeof(float));
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_map(d_a, d_b, d_c, size.dims[0]);
    }

    void teardown_student() override {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    CorrectnessResult verify() override {
        cudaMemcpy(h_c.data(), d_c, size.dims[0] * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_c_ref.data(), h_c.data(), size.dims[0], 1e-5f);
    }

    void print_mismatch() override {
        print_array("Input (a)", h_a.data(), size.dims[0]);
        print_array("Input (b)", h_b.data(), size.dims[0]);
        print_array("Expected Output", h_c_ref.data(), size.dims[0]);
        print_array("Actual Output", h_c.data(), size.dims[0]);
    }

    double get_bandwidth_bytes() override {
        return 3.0 * size.dims[0] * sizeof(float);
    }

    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);

    std::vector<TestSize<1>> correctness_sizes = {
        {1}, {31}, {32}, {33}, {63}, {64}, {65}, {1023}, {1024}, {1025}
    };
    
    std::vector<TestSize<1>> perf_sizes = {
        {{1 << 20}}, {{1 << 24}}, {{1 << 26}}, {{1 << 30}}
    };

    run_test_suite<1, MapTest>("Exercise 01: Vector Addition", config, correctness_sizes, perf_sizes);
    return 0;
}
