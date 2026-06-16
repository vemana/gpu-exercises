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

void cpu_baseline(const int* a, int* bins, int size, int num_bins) {
    for (int i = 0; i < size; ++i) {
        if (a[i] >= 0 && a[i] < num_bins) {
            bins[a[i]]++;
        }
    }
}

struct HistogramTest : public ProblemTest<1> {
    std::vector<int> h_a;
    std::vector<int> h_bins;
    std::vector<int> h_bins_ref;

    int *d_a = nullptr;
    int *d_bins = nullptr;
    int num_bins = 256;

    HistogramTest(const TestSize<1>& size) : ProblemTest<1>(size) {}

    void generate_test_data(bool check) override {
        int n = size.dims[0];
        h_a.resize(n);
        h_bins.assign(num_bins, 0);
        h_bins_ref.assign(num_bins, 0);

        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(0, num_bins - 1);
        for (int i = 0; i < n; ++i) {
            h_a[i] = dist(gen);
        }
        if (check) cpu_baseline(h_a.data(), h_bins_ref.data(), n, num_bins);
    }

    void setup_reference() override {
        int n = size.dims[0];
        cudaMalloc(&d_a, n * sizeof(int));
        cudaMalloc(&d_bins, num_bins * sizeof(int));
        cudaMemcpy(d_a, h_a.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_bins, 0, num_bins * sizeof(int));
    }

    LaunchMetrics launch_reference() override {
        return launch_reference_histogram(d_a, d_bins, size.dims[0], num_bins);
    }

    void setup_student() override {
        int n = size.dims[0];
        cudaMalloc(&d_a, n * sizeof(int));
        cudaMalloc(&d_bins, num_bins * sizeof(int));
        cudaMemcpy(d_a, h_a.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_bins, 0, num_bins * sizeof(int));
    }

    LaunchMetrics launch_student() override {
        return launch_histogram(d_a, d_bins, size.dims[0], num_bins);
    }

    CorrectnessResult verify() override {
        cudaMemcpy(h_bins.data(), d_bins, num_bins * sizeof(int), cudaMemcpyDeviceToHost);
        return check_correctness(h_bins_ref.data(), h_bins.data(), num_bins, 0);
    }

    void print_mismatch() override {
        std::cout << "\n--- Expected Output ---\n";
        for (int i = 0; i < std::min(num_bins, 10); ++i) {
            std::cout << std::setw(8) << h_bins_ref[i];
            if ((i + 1) % 10 == 0) std::cout << "\n";
        }
        std::cout << "\n--- Actual Output ---\n";
        for (int i = 0; i < std::min(num_bins, 10); ++i) {
            std::cout << std::setw(8) << h_bins[i];
            if ((i + 1) % 10 == 0) std::cout << "\n";
        }
        std::cout << "\n";
    }

    double get_bandwidth_bytes() override {
        return size.dims[0] * sizeof(int) + num_bins * sizeof(int);
    }

    void teardown() override {
        if (d_a) cudaFree(d_a);
        if (d_bins) cudaFree(d_bins);
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

    run_test_suite<1, HistogramTest>("Exercise 04: Histogram", config, correctness_sizes, perf_sizes);
    return 0;
}
