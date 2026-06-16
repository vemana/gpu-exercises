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

void cpu_baseline(const int* a, int* c, int* count, int size) {
    int valid_count = 0;
    for (int i = 0; i < size; ++i) {
        if (a[i] > 0) {
            c[valid_count++] = a[i];
        }
    }
    *count = valid_count;
}

struct StreamCompactionTest : public ProblemTest<1> {
    std::vector<int> h_a;
    std::vector<int> h_c;
    std::vector<int> h_c_ref;
    int h_count = 0;
    int h_count_ref = 0;

    int *d_a = nullptr;
    int *d_c = nullptr;
    int *d_count = nullptr;

    StreamCompactionTest(const TestSize<1>& size) : ProblemTest<1>(size) {}

    void generate_test_data(bool check) override {
        int n = size.dims[0];
        h_a.resize(n);
        h_c.assign(n, 0);
        h_c_ref.assign(n, 0);

        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(-50, 50);
        for (int i = 0; i < n; ++i) {
            h_a[i] = dist(gen);
        }
        if (check) cpu_baseline(h_a.data(), h_c_ref.data(), &h_count_ref, n);
    }

    void setup_reference() override {
        int n = size.dims[0];
        cudaMalloc(&d_a, n * sizeof(int));
        cudaMalloc(&d_c, n * sizeof(int));
        cudaMalloc(&d_count, sizeof(int));
        
        cudaMemcpy(d_a, h_a.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_c, 0, n * sizeof(int));
        cudaMemset(d_count, 0, sizeof(int));
    }

    LaunchMetrics launch_reference() override {
        return launch_reference_compaction(d_a, d_c, d_count, size.dims[0]);
    }

    void setup_student() override {
        int n = size.dims[0];
        cudaMalloc(&d_a, n * sizeof(int));
        cudaMalloc(&d_c, n * sizeof(int));
        cudaMalloc(&d_count, sizeof(int));
        
        cudaMemcpy(d_a, h_a.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_c, 0, n * sizeof(int));
        cudaMemset(d_count, 0, sizeof(int));
    }

    LaunchMetrics launch_student() override {
        return launch_compaction(d_a, d_c, d_count, size.dims[0]);
    }

    CorrectnessResult verify() override {
        cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_c.data(), d_c, h_count * sizeof(int), cudaMemcpyDeviceToHost);
        
        if (h_count != h_count_ref) {
            return {false, 0, (float)h_count_ref, (float)h_count};
        }
        
        return check_correctness(h_c_ref.data(), h_c.data(), h_count_ref, 0);
    }

    void print_mismatch() override {
        std::cout << "\nExpected Count: " << h_count_ref << "\n";
        std::cout << "Actual Count:   " << h_count << "\n";
        
        std::cout << "\n--- Expected Output (First 10) ---\n";
        for (int i = 0; i < std::min(h_count_ref, 10); ++i) {
            std::cout << std::setw(8) << h_c_ref[i];
            if ((i + 1) % 10 == 0) std::cout << "\n";
        }
        std::cout << "\n--- Actual Output (First 10) ---\n";
        for (int i = 0; i < std::min(h_count, 10); ++i) {
            std::cout << std::setw(8) << h_c[i];
            if ((i + 1) % 10 == 0) std::cout << "\n";
        }
        std::cout << "\n";
    }

    double get_bandwidth_bytes() override {
        // Read entire array A, write to array C (only positive elements)
        return size.dims[0] * sizeof(int) + h_count_ref * sizeof(int);
    }

    void teardown() override {
        if (d_a) cudaFree(d_a);
        if (d_c) cudaFree(d_c);
        if (d_count) cudaFree(d_count);
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

    run_test_suite<1, StreamCompactionTest>("Exercise 08: Stream Compaction", config, correctness_sizes, perf_sizes);
    return 0;
}
