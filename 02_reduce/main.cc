

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

Tracer global_tracer(true);

void cpu_baseline(const float* a, float* c, long long size) {
    float sum = 0.0f;
    for (long long i = 0; i < size; ++i) {
        sum += a[i];
    }
    c[0] = sum;
}

struct ReduceTest : public ProblemTest<1> {
    float *d_a = nullptr;
    float *d_c = nullptr;
    std::vector<float> h_a;
    std::vector<float> h_c;
    std::vector<float> h_c_ref_cpu;

    ReduceTest(const TestSize<1>& size) : ProblemTest<1>(size) {}

    void generate_test_data(bool check) override {
        long long n = size.dims[0];
        h_a.resize(n);
        h_c.resize(1, 0.0f);
        h_c_ref_cpu.resize(1, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (long long i = 0; i < n; ++i) {
            h_a[i] = dist(gen);
        }

        if (check) {
            cpu_baseline(h_a.data(), h_c_ref_cpu.data(), n);
        }

        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_c, sizeof(float));
        cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    }

    void setup_reference() override {
        cudaMemset(d_c, 0, sizeof(float));
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_reduce(d_a, d_c, size.dims[0]);
    }

    void setup_student() override {
        cudaMemset(d_c, 0, sizeof(float));
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_reduce(d_a, d_c, size.dims[0]);
    }

    CorrectnessResult verify() override {
        cudaMemcpy(h_c.data(), d_c, sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_c_ref_cpu.data(), h_c.data(), 1, 1e-1);
    }

    void print_mismatch() override {
        print_array("Input (a)", h_a.data(), size.dims[0]);
        print_array("Expected Output", h_c_ref_cpu.data(), 1);
        print_array("Actual Output", h_c.data(), 1);
    }

    double get_bandwidth_bytes() override {
        return size.dims[0] * sizeof(float);
    }

    void teardown() override {
        cudaFree(d_a);
        cudaFree(d_c);
    }
};

int main(int argc, char** argv) {
    auto config = parse_args<1>(argc, argv);
    
    std::vector<TestSize<1>> correctness_sizes = {
        {{1}}, {{31}}, {{32}}, {{33}}, {{63}}, {{64}}, {{65}},
        {{1023}}, {{1024}}, {{1025}}
    };

    std::vector<TestSize<1>> perf_sizes = {
        {{1 << 20}}, {{1 << 24}}, {{1 << 26}}, {{1 << 30}}, {{1LL << 32}}
    };

    run_test_suite<1, ReduceTest>("Sum reduction", config, correctness_sizes, perf_sizes);

    return 0;
}
