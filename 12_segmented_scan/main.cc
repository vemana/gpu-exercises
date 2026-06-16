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

void cpu_baseline(const float* a, const int* flags, float* c, int size) {
    float current_sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        if (flags[i] == 1) {
            current_sum = 0.0f;
        }
        current_sum += a[i];
        c[i] = current_sum;
    }
}

struct SegmentedScanTest : public ProblemTest<1> {
    std::vector<float> h_a;
    std::vector<int> h_flags;
    std::vector<float> h_c;
    std::vector<float> h_c_ref;

    float *d_a = nullptr;
    int *d_flags = nullptr;
    float *d_c = nullptr;

    SegmentedScanTest(const TestSize<1>& size) : ProblemTest<1>(size) {}

    void generate_test_data(bool check) override {
        int n = size.dims[0];
        h_a.resize(n);
        h_flags.resize(n);
        h_c.assign(n, 0.0f);
        h_c_ref.assign(n, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(0.0f, 10.0f);
        std::uniform_real_distribution<float> flag_prob(0.0f, 1.0f);
        
        for (int i = 0; i < n; ++i) {
            h_a[i] = dist(gen);
            // 10% chance to start a new segment, but first element is always start
            if (i == 0 || flag_prob(gen) < 0.1f) {
                h_flags[i] = 1;
            } else {
                h_flags[i] = 0;
            }
        }
        if (check) cpu_baseline(h_a.data(), h_flags.data(), h_c_ref.data(), n);
    }

    void setup_reference() override {
        int n = size.dims[0];
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_flags, n * sizeof(int));
        cudaMalloc(&d_c, n * sizeof(float));
        
        cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_flags, h_flags.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_c, 0, n * sizeof(float));
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_segmented_scan(d_a, d_flags, d_c, size.dims[0]);
    }

    void setup_student() override {
        int n = size.dims[0];
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_flags, n * sizeof(int));
        cudaMalloc(&d_c, n * sizeof(float));
        
        cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_flags, h_flags.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_c, 0, n * sizeof(float));
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_segmented_scan(d_a, d_flags, d_c, size.dims[0]);
    }

    CorrectnessResult verify() override {
        int n = size.dims[0];
        cudaMemcpy(h_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_c_ref.data(), h_c.data(), n, 1e-2f);
    }

    void print_mismatch() override {
        int n = size.dims[0];
        std::cout << "\n--- Expected Output (First 10) ---\n";
        for (int i = 0; i < std::min(n, 10); ++i) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(8) << h_c_ref[i];
            if ((i + 1) % 10 == 0) std::cout << "\n";
        }
        std::cout << "\n--- Actual Output (First 10) ---\n";
        for (int i = 0; i < std::min(n, 10); ++i) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(8) << h_c[i];
            if ((i + 1) % 10 == 0) std::cout << "\n";
        }
        std::cout << "\n";
    }

    double get_bandwidth_bytes() override {
        // Read A, Read Flags, Write C
        return size.dims[0] * sizeof(float) + size.dims[0] * sizeof(int) + size.dims[0] * sizeof(float);
    }

    void teardown() override {
        if (d_a) cudaFree(d_a);
        if (d_flags) cudaFree(d_flags);
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

    run_test_suite<1, SegmentedScanTest>("Exercise 12: Segmented Scan", config, correctness_sizes, perf_sizes);
    return 0;
}
