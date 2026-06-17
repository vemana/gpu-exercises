

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/argparse.h"
#include "../utils/framework.h"
#include "../utils/tracer.h"
#include "kernel.h"
#include "reference_kernel.h"

void cpu_baseline(const int* a, int* c, long long size) {
    for (long long i = 0; i < size; ++i) {
        c[i] = a[i];
    }
    std::sort(c, c + size);
}

struct RadixSortTest : public ProblemTest<1> {
    std::vector<int> h_a;
    std::vector<int> h_c;
    std::vector<int> h_c_ref;

    int *d_a = nullptr;
    int *d_c = nullptr;

    RadixSortTest(const TestSize<1>& size) : ProblemTest<1>(size) {}

    void generate_test_data(bool check) override {
        long long n = size.dims[0];
        h_a.resize(n);
        h_c.assign(n, 0);
        h_c_ref.assign(n, 0);

        std::mt19937 gen(42);
        // Using positive integers for simpler radix sort implementation
        std::uniform_int_distribution<int> dist(0, 1000000);
        for (long long i = 0; i < n; ++i) {
            h_a[i] = dist(gen);
        }
        if (check) cpu_baseline(h_a.data(), h_c_ref.data(), n);
    }

    void setup_reference() override {
        long long n = size.dims[0];
        cudaMalloc(&d_a, n * sizeof(int));
        cudaMalloc(&d_c, n * sizeof(int));
        
        cudaMemcpy(d_a, h_a.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_c, 0, n * sizeof(int));
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_radix_sort(d_a, d_c, size.dims[0]);
    }

    void setup_student() override {
        long long n = size.dims[0];
        cudaMalloc(&d_a, n * sizeof(int));
        cudaMalloc(&d_c, n * sizeof(int));
        
        cudaMemcpy(d_a, h_a.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_c, 0, n * sizeof(int));
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_radix_sort(d_a, d_c, size.dims[0]);
    }

    CorrectnessResult verify() override {
        long long n = size.dims[0];
        cudaMemcpy(h_c.data(), d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        return check_correctness(h_c_ref.data(), h_c.data(), n, 0);
    }

    void print_mismatch() override {
        long long n = size.dims[0];
        print_array("Input (a)", h_a.data(), n);
        print_array("Expected Output", h_c_ref.data(), n);
        print_array("Actual Output", h_c.data(), n);
    }

    double get_bandwidth_bytes() override {
        // Useful bandwidth: reading the array once, writing it once.
        return 2.0 * size.dims[0] * sizeof(int);
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

    run_test_suite<1, RadixSortTest>("Exercise 09: Radix Sort", config, correctness_sizes, perf_sizes);
    return 0;
}
