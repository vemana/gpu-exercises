#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "../utils/framework.h"
#include "../utils/argparse.h"
#include "kernel.h"
#include "reference_kernel.h"

#define PI 3.14159265358979323846f

// Note: HPC applications typically use double precision.
// We use float here for consistency across the repository exercises.

void cpu_dft(const cuFloatComplex* x_in, cuFloatComplex* x_out, int N) {
    for (int k = 0; k < N; ++k) {
        double sum_re = 0.0;
        double sum_im = 0.0;
        for (int n = 0; n < N; ++n) {
            double theta = -2.0 * 3.14159265358979323846 * k * n / N;
            double cos_theta = std::cos(theta);
            double sin_theta = std::sin(theta);
            
            sum_re += (double)x_in[n].x * cos_theta - (double)x_in[n].y * sin_theta;
            sum_im += (double)x_in[n].x * sin_theta + (double)x_in[n].y * cos_theta;
        }
        x_out[k] = make_cuFloatComplex((float)sum_re, (float)sum_im);
    }
}

struct FFTTest : public ProblemTest<1> {
    std::vector<cuFloatComplex> h_x_initial;
    std::vector<cuFloatComplex> h_x;
    std::vector<cuFloatComplex> h_x_ref;

    cuFloatComplex *d_x = nullptr;

    FFTTest(const TestSize<1>& size) : ProblemTest<1>(size) {}

    void generate_test_data(bool check) override {
        int N = size.dims[0];
        // Ensure N is a power of 2
        if ((N & (N - 1)) != 0) {
            std::cerr << "Error: N must be a power of 2 for Radix-2 FFT." << std::endl;
            exit(1);
        }

        h_x_initial.resize(N);
        h_x.resize(N);
        h_x_ref.resize(N);

        for (int i = 0; i < N; ++i) {
            // Signal: combination of sine waves
            float val = std::sin(2.0f * PI * i * 4.0f / N) + 0.5f * std::sin(2.0f * PI * i * 8.0f / N);
            h_x_initial[i] = make_cuFloatComplex(val, 0.0f);
        }

        if (check) {
            // Compute slow CPU DFT for correctness
            cpu_dft(h_x_initial.data(), h_x_ref.data(), N);
        }
    }

    void setup_reference() override {
        int N = size.dims[0];
        cudaMalloc(&d_x, N * sizeof(cuFloatComplex));
        cudaMemcpy(d_x, h_x_initial.data(), N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_fft_reference(d_x, size.dims[0]);
    }

    void teardown_reference() override {
        cudaFree(d_x);
    }

    void setup_student() override {
        int N = size.dims[0];
        cudaMalloc(&d_x, N * sizeof(cuFloatComplex));
        cudaMemcpy(d_x, h_x_initial.data(), N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_fft(d_x, size.dims[0]);
    }

    void teardown_student() override {
        cudaFree(d_x);
    }

    CorrectnessResult verify() override {
        int N = size.dims[0];
        cudaMemcpy(h_x.data(), d_x, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < N; ++i) {
            float diff_re = std::abs(h_x_ref[i].x - h_x[i].x);
            float diff_im = std::abs(h_x_ref[i].y - h_x[i].y);
            float magnitude = std::sqrt(h_x_ref[i].x * h_x_ref[i].x + h_x_ref[i].y * h_x_ref[i].y) + 1e-5f;
            // DFT accumulates floating point errors heavily. Use relative error or larger absolute.
            if (diff_re > 1.0f || diff_im > 1.0f || (diff_re / magnitude > 1e-3f && diff_re > 0.1f) || (diff_im / magnitude > 1e-3f && diff_im > 0.1f)) {
                return {false, i, diff_re > diff_im ? h_x_ref[i].x : h_x_ref[i].y, diff_re > diff_im ? h_x[i].x : h_x[i].y};
            }
        }
        return {true, -1, 0.0f, 0.0f};
    }

    void print_mismatch() override {
        int N = size.dims[0];
        std::cout << "--- Expected vs Actual (Real | Imaginary) ---" << std::endl;
        for (int i = 0; i < std::min(N, 20); ++i) {
            std::cout << "Idx " << i << ": Expected (" << h_x_ref[i].x << ", " << h_x_ref[i].y 
                      << ") | Actual (" << h_x[i].x << ", " << h_x[i].y << ")" << std::endl;
        }
    }

    double get_bandwidth_bytes() override {
        int N = size.dims[0];
        int log2N = 0;
        while ((1 << log2N) < N) log2N++;
        
        // Bit reversal: 1 read, 1 write per element
        // log2N stages: 1 read, 1 write per element per stage
        return (double)N * sizeof(cuFloatComplex) * 2 * (1 + log2N);
    }

    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);

    // Only powers of 2 for Radix-2 FFT
    std::vector<TestSize<1>> correctness_sizes = {
        {{16}}, {{64}}, {{1024}}, {{4096}}
    };
    
    std::vector<TestSize<1>> perf_sizes = {
        {{1 << 16}}, {{1 << 20}}, {{1 << 24}}
    };

    run_test_suite<1, FFTTest>("Exercise 27: Fast Fourier Transform", config, correctness_sizes, perf_sizes);
    return 0;
}
