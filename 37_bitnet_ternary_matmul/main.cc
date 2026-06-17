#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <cstdint>

#include <cuda_runtime.h>

#include "../utils/argparse.h"
#include "../utils/framework.h"
#include "../utils/tracer.h"
#include "../utils/utils.h"
#include "kernel.h"
#include "reference_kernel.h"

// CPU Baseline: Unpacks 2-bit values from W and performs addition/subtraction instead of multiplication.
void cpu_bitnet_matmul(const float* A, const uint8_t* W_packed, float* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            int k_packed_limit = K / 4;
            for (int k_idx = 0; k_idx < k_packed_limit; ++k_idx) {
                uint8_t packed_val = W_packed[k_idx * N + n];
                for (int i = 0; i < 4; ++i) {
                    int k = k_idx * 4 + i;
                    float a_val = A[m * K + k];
                    uint8_t bits = (packed_val >> (i * 2)) & 0x3;
                    if (bits == 1) sum += a_val;
                    else if (bits == 2) sum -= a_val;
                }
            }
            C[m * N + n] = sum;
        }
    }
}

struct BitNetTest : public ProblemTest<1> {
    int M = 64;
    int K = 1024;
    
    std::vector<float> h_A;
    std::vector<uint8_t> h_W_packed;
    std::vector<float> h_C;
    std::vector<float> h_C_ref;

    float *d_A = nullptr;
    uint8_t *d_W_packed = nullptr;
    float *d_C = nullptr;

    BitNetTest(const TestSize<1>& size) : ProblemTest<1>(size) {}

    void generate_test_data(bool check) override {
        int N = size.dims[0];
        h_A.resize(M * K);
        int packed_K = K / 4;
        h_W_packed.resize(packed_K * N);
        h_C.assign(M * N, 0.0f);
        h_C_ref.assign(M * N, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::uniform_int_distribution<int> weight_dist(0, 2);

        for (int i = 0; i < M * K; ++i) h_A[i] = dist(gen);

        for (int k_idx = 0; k_idx < packed_K; ++k_idx) {
            for (int n = 0; n < N; ++n) {
                uint8_t packed_byte = 0;
                for (int i = 0; i < 4; ++i) {
                    int w_val = weight_dist(gen); 
                    packed_byte |= (w_val << (i * 2));
                }
                h_W_packed[k_idx * N + n] = packed_byte;
            }
        }
        if (check) cpu_bitnet_matmul(h_A.data(), h_W_packed.data(), h_C_ref.data(), M, N, K);
    }

    void setup_memory() {
        int N = size.dims[0];
        int packed_K = K / 4;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_W_packed, packed_K * N * sizeof(uint8_t));
        cudaMalloc(&d_C, M * N * sizeof(float));
        cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_W_packed, h_W_packed.data(), packed_K * N * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemset(d_C, 0, M * N * sizeof(float));
    }

    void free_memory() {
        cudaFree(d_A);
        cudaFree(d_W_packed);
        cudaFree(d_C);
    }

    void setup_reference() override { setup_memory(); }
    void teardown_reference() override { free_memory(); }
    void setup_student() override { setup_memory(); }
    void teardown_student() override { free_memory(); }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_bitnet_ternary_matmul(d_A, d_W_packed, d_C, M, size.dims[0], K);
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_bitnet_ternary_matmul(d_A, d_W_packed, d_C, M, size.dims[0], K);
    }

    CorrectnessResult verify() override {
        cudaMemcpy(h_C.data(), d_C, M * size.dims[0] * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_C_ref.data(), h_C.data(), M * size.dims[0], 1e-3f);
    }

    void print_mismatch() override {
        print_array("Expected Output", h_C_ref.data(), M * size.dims[0]);
        print_array("Actual Output", h_C.data(), M * size.dims[0]);
    }

    double get_bandwidth_bytes() override {
        int N = size.dims[0];
        return (M * K * sizeof(float)) + ((K / 4) * N * sizeof(uint8_t)) + (M * N * sizeof(float));
    }
    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);
    std::vector<TestSize<1>> correctness_sizes = {{64}, {128}, {256}};
    std::vector<TestSize<1>> perf_sizes = {{1024}, {2048}, {4096}};
    run_test_suite<1, BitNetTest>("Exercise 37: BitNet Ternary MatMul", config, correctness_sizes, perf_sizes);
    return 0;
}
