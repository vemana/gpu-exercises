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

void cpu_baseline(const float* a, float* c, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        c[i] = sum;
        sum += a[i];
    }
}

struct ScanTest : public ProblemTest<1> {
    std::vector<float> h_a;
    std::vector<float> h_c;
    std::vector<float> h_c_ref;

    float *d_a = nullptr;
    float *d_c = nullptr;

    ScanTest(const TestSize<1>& size) : ProblemTest<1>(size) {}

    void generate_test_data(bool check) override {
        int n = size.dims[0];
        h_a.resize(n);
        h_c.assign(n, 0.0f);
        h_c_ref.assign(n, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(1.0f, 5.0f);
        for (int i = 0; i < n; ++i) {
            h_a[i] = dist(gen);
        }
        if (check) cpu_baseline(h_a.data(), h_c_ref.data(), n);
    }

    void setup_reference() override {
        int n = size.dims[0];
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_c, n * sizeof(float));
        cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_c, 0, n * sizeof(float));
    }

    LaunchMetrics launch_reference() override {
        return launch_reference_scan(d_a, d_c, size.dims[0]);
    }

    void setup_student() override {
        int n = size.dims[0];
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_c, n * sizeof(float));
        cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_c, 0, n * sizeof(float));
    }

    LaunchMetrics launch_student() override {
        return launch_scan(d_a, d_c, size.dims[0]);
    }

    CorrectnessResult verify() override {
        cudaMemcpy(h_c.data(), d_c, size.dims[0] * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_c_ref.data(), h_c.data(), size.dims[0], 1e-1f);
    }

    void print_mismatch() override {
        std::cout << "\n--- Expected Output ---\n";
        for (int i = 0; i < std::min(size.dims[0], 10LL); ++i) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(8) << h_c_ref[i];
            if ((i + 1) % 10 == 0) std::cout << "\n";
        }
        std::cout << "\n--- Actual Output ---\n";
        for (int i = 0; i < std::min(size.dims[0], 10LL); ++i) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(8) << h_c[i];
            if ((i + 1) % 10 == 0) std::cout << "\n";
        }
        std::cout << "\n";
    }

    double get_bandwidth_bytes() override {
        return 2.0 * size.dims[0] * sizeof(float);
    }

    void teardown() override {
        if (d_a) cudaFree(d_a);
        if (d_c) cudaFree(d_c);
    }
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);

    std::vector<TestSize<1>> correctness_sizes = {
        {1}, {31}, {32}, {33}, {63}, {64}, {65}, {1023}, {1024}, {1025}
    };
    
    std::vector<TestSize<1>> perf_sizes = {
        {1048576}, {16777216}, {67108864}
    };

    run_test_suite<1, ScanTest>("Exercise 03: Exclusive Prefix Sum", config, correctness_sizes, perf_sizes);
    return 0;
}
