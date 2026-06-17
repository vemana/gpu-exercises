

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/argparse.h"
#include "../utils/framework.h"
#include "../utils/tracer.h"
#include "kernel.h"
#include "reference_kernel.h"

#define PI 3.14159265358979323846f

// Note: HPC applications typically use double precision.
// We use float here for consistency across the repository exercises.

uint32_t cpu_pcg_hash(uint32_t input) {
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float cpu_rand_uniform(uint32_t& state) {
    state = cpu_pcg_hash(state);
    return (float)(state + 1) / 4294967297.0f;
}

float cpu_monte_carlo(int N, float S0, float K, float r, float sigma, float T) {
    double total_sum = 0.0;
    float drift = (r - 0.5f * sigma * sigma) * T;
    float vol = sigma * std::sqrt(T);

    for (int i = 0; i < N; ++i) {
        uint32_t state = i + 12345;
        
        float u1 = cpu_rand_uniform(state);
        float u2 = cpu_rand_uniform(state);
        
        float z = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * PI * u2);
        float ST = S0 * std::exp(drift + vol * z);
        float payoff = std::max(ST - K, 0.0f);
        
        total_sum += payoff;
    }
    
    // Average and discount
    return (float)(std::exp(-r * T) * (total_sum / N));
}

struct MonteCarloTest : public ProblemTest<1> {
    float S0 = 100.0f;
    float K = 100.0f;
    float r = 0.05f;
    float sigma = 0.2f;
    float T = 1.0f;
    
    float h_sum_ref = 0.0f;
    float h_sum = 0.0f;
    
    float* d_sum = nullptr;

    MonteCarloTest(const TestSize<1>& size) : ProblemTest<1>(size) {}

    void generate_test_data(bool check) override {
        if (check) {
            h_sum_ref = cpu_monte_carlo(size.dims[0], S0, K, r, sigma, T);
        }
    }

    void setup_reference() override {
        cudaMalloc(&d_sum, sizeof(float));
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_monte_carlo_reference(d_sum, size.dims[0], S0, K, r, sigma, T);
    }

    void teardown_reference() override {
        cudaFree(d_sum);
    }

    void setup_student() override {
        cudaMalloc(&d_sum, sizeof(float));
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_monte_carlo(d_sum, size.dims[0], S0, K, r, sigma, T);
    }

    void teardown_student() override {
        cudaFree(d_sum);
    }

    CorrectnessResult verify() override {
        cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
        
        // The kernel returns the SUM of payoffs, not the discounted average.
        // We must compute the discounted average to compare with CPU.
        h_sum = std::exp(-r * T) * (h_sum / size.dims[0]);
        
        float diff = std::abs(h_sum_ref - h_sum);
        // Float accumulation differs between CPU sequential and GPU parallel tree-reduction.
        // For large N, we accept a 0.5% relative error.
        if (diff > 1e-2f && (diff / h_sum_ref) > 0.005f) {
            return {false, 0, h_sum_ref, h_sum};
        }
        return {true, -1, 0.0f, 0.0f};
    }

    void print_mismatch() override {
        std::cout << "--- Expected Option Price vs Actual ---" << std::endl;
        std::cout << "Expected: " << h_sum_ref << " | Actual: " << h_sum << std::endl;
    }

    double get_bandwidth_bytes() override {
        // Monte Carlo is totally compute bound. 
        // We read parameters (broadcast) and write out 1 float (reduction).
        // Returning a tiny nominal value so BW calculation doesn't divide by zero.
        return sizeof(float);
    }

    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);

    std::vector<TestSize<1>> correctness_sizes = {
        {{1000}}, {{100000}}, {{1000000}}
    };
    
    std::vector<TestSize<1>> perf_sizes = {
        {{10000000}}, {{50000000}}
    };

    run_test_suite<1, MonteCarloTest>("Exercise 30: Monte Carlo Options Pricing", config, correctness_sizes, perf_sizes);
    return 0;
}
