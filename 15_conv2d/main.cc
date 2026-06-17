

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

void cpu_baseline(const float* a, const float* filter, float* c, long long width, long long height) {
    int filter_size = 3;
    int offset = filter_size / 2;
    for (long long y = 0; y < height; ++y) {
        for (long long x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int fy = -offset; fy <= offset; ++fy) {
                for (int fx = -offset; fx <= offset; ++fx) {
                    int iy = y + fy;
                    int ix = x + fx;
                    if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                        float pixel = a[iy * width + ix];
                        float weight = filter[(fy + offset) * filter_size + (fx + offset)];
                        sum += pixel * weight;
                    }
                }
            }
            c[y * width + x] = sum;
        }
    }
}

struct Conv2DTest : public ProblemTest<2> {
    std::vector<float> h_a;
    std::vector<float> h_filter;
    std::vector<float> h_c;
    std::vector<float> h_c_ref;

    float *d_a = nullptr;
    float *d_filter = nullptr;
    float *d_c = nullptr;

    Conv2DTest(const TestSize<2>& size) : ProblemTest<2>(size) {}

    void generate_test_data(bool check) override {
        long long width = size.dims[0];
        long long height = size.dims[1];
        long long n = width * height;
        h_a.resize(n);
        h_filter.resize(9);
        h_c.assign(n, 0.0f);
        h_c_ref.assign(n, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        
        for (long long i = 0; i < n; ++i) {
            h_a[i] = dist(gen);
        }
        for (long long i = 0; i < 9; ++i) {
            h_filter[i] = dist(gen);
        }
        
        if (check) cpu_baseline(h_a.data(), h_filter.data(), h_c_ref.data(), width, height);
    }

    void setup_reference() override {
        long long width = size.dims[0];
        long long height = size.dims[1];
        long long n = width * height;
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_filter, 9 * sizeof(float));
        cudaMalloc(&d_c, n * sizeof(float));
        
        cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_filter, h_filter.data(), 9 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_c, 0, n * sizeof(float));
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_conv2d(d_a, d_filter, d_c, size.dims[0], size.dims[1]);
    }

    void setup_student() override {
        long long width = size.dims[0];
        long long height = size.dims[1];
        long long n = width * height;
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_filter, 9 * sizeof(float));
        cudaMalloc(&d_c, n * sizeof(float));
        
        cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_filter, h_filter.data(), 9 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_c, 0, n * sizeof(float));
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_conv2d(d_a, d_filter, d_c, size.dims[0], size.dims[1]);
    }

    CorrectnessResult verify() override {
        long long width = size.dims[0];
        long long height = size.dims[1];
        long long n = width * height;
        cudaMemcpy(h_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_c_ref.data(), h_c.data(), n, 1e-2f);
    }

    void print_mismatch() override {
        long long width = size.dims[0];
        long long height = size.dims[1];
        long long n = width * height;
        print_array("Input (a)", h_a.data(), n);
        print_array("Filter", h_filter.data(), 9);
        print_array("Expected Output", h_c_ref.data(), n);
        print_array("Actual Output", h_c.data(), n);
    }

    double get_bandwidth_bytes() override {
        long long width = size.dims[0];
        long long height = size.dims[1];
        long long n = width * height;
        // Ideally reads a pixel once, writes a pixel once. Filter read is tiny.
        return 2.0 * n * sizeof(float);
    }

    void teardown() override {
        if (d_a) cudaFree(d_a);
        if (d_filter) cudaFree(d_filter);
        if (d_c) cudaFree(d_c);
    }
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<2> config = parse_args<2>(argc, argv);

    std::vector<TestSize<2>> correctness_sizes = {
        {1, 1}, {15, 15}, {16, 16}, {17, 17}, {31, 31}, {32, 32}, {33, 33}, {127, 127}, {128, 128}, {129, 129}
    };
    
    std::vector<TestSize<2>> perf_sizes = {
        {1024, 1024}, {2048, 2048}, {4096, 4096}
    };

    run_test_suite<2, Conv2DTest>("Exercise 15: 2D Convolution", config, correctness_sizes, perf_sizes);
    return 0;
}
