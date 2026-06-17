

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

void cpu_moe_routing(
    const std::vector<float>& logits, std::vector<int>& top_experts, std::vector<float>& routing_weights, 
    int num_tokens, int num_experts, int K) {
    
    for (int t = 0; t < num_tokens; ++t) {
        const float* my_logits = &logits[t * num_experts];
        
        int best_e = -1, second_e = -1;
        float best_val = -1e20f, second_val = -1e20f;
        
        for (int e = 0; e < num_experts; ++e) {
            float val = my_logits[e];
            if (val > best_val) {
                second_val = best_val;
                second_e = best_e;
                best_val = val;
                best_e = e;
            } else if (val > second_val) {
                second_val = val;
                second_e = e;
            }
        }
        
        float max_val = best_val;
        float exp1 = std::exp(best_val - max_val);
        float exp2 = std::exp(second_val - max_val);
        float sum_exp = exp1 + exp2;
        
        top_experts[t * K + 0] = best_e;
        top_experts[t * K + 1] = second_e;
        routing_weights[t * K + 0] = exp1 / sum_exp;
        routing_weights[t * K + 1] = exp2 / sum_exp;
    }
}

struct MoERoutingTest : public ProblemTest<1> {
    int num_tokens;
    int num_experts = 8;
    int K = 2;
    
    std::vector<float> h_logits;
    std::vector<int> h_top_experts_ref, h_top_experts;
    std::vector<float> h_routing_weights_ref, h_routing_weights;

    float *d_logits = nullptr, *d_routing_weights = nullptr;
    int *d_top_experts = nullptr;

    MoERoutingTest(const TestSize<1>& size) : ProblemTest<1>(size) {
        num_tokens = size.dims[0];
    }

    void generate_test_data(bool check) override {
        h_logits.resize(num_tokens * num_experts);
        h_top_experts.resize(num_tokens * K, 0);
        h_top_experts_ref.resize(num_tokens * K, 0);
        h_routing_weights.resize(num_tokens * K, 0.0f);
        h_routing_weights_ref.resize(num_tokens * K, 0.0f);
        
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
        
        for (float& val : h_logits) val = dist(gen);

        if (check) {
            cpu_moe_routing(h_logits, h_top_experts_ref, h_routing_weights_ref, num_tokens, num_experts, K);
        }
    }

    void setup_reference() override {
        cudaMalloc(&d_logits, num_tokens * num_experts * sizeof(float));
        cudaMalloc(&d_top_experts, num_tokens * K * sizeof(int));
        cudaMalloc(&d_routing_weights, num_tokens * K * sizeof(float));
        
        cudaMemcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_moe_routing_reference(d_logits, d_top_experts, d_routing_weights, num_tokens, num_experts, K);
    }

    void teardown_reference() override {
        cudaFree(d_logits); cudaFree(d_top_experts); cudaFree(d_routing_weights);
    }

    void setup_student() override {
        cudaMalloc(&d_logits, num_tokens * num_experts * sizeof(float));
        cudaMalloc(&d_top_experts, num_tokens * K * sizeof(int));
        cudaMalloc(&d_routing_weights, num_tokens * K * sizeof(float));
        
        cudaMemcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_moe_routing(d_logits, d_top_experts, d_routing_weights, num_tokens, num_experts, K);
    }

    void teardown_student() override {
        cudaFree(d_logits); cudaFree(d_top_experts); cudaFree(d_routing_weights);
    }

    CorrectnessResult verify() override {
        cudaMemcpy(h_top_experts.data(), d_top_experts, num_tokens * K * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_routing_weights.data(), d_routing_weights, num_tokens * K * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < num_tokens * K; ++i) {
            if (h_top_experts_ref[i] != h_top_experts[i]) {
                return {false, i, (float)h_top_experts_ref[i], (float)h_top_experts[i]};
            }
            float diff = std::abs(h_routing_weights_ref[i] - h_routing_weights[i]);
            if (diff > 1e-4f) {
                return {false, i, h_routing_weights_ref[i], h_routing_weights[i]};
            }
        }
        return {true, -1, 0.0f, 0.0f};
    }

    void print_mismatch() override {
        std::cout << "--- Expected vs Actual ---" << std::endl;
        for (int i = 0; i < std::min((int)num_tokens * K, 20); ++i) {
            std::cout << "Idx " << i << ": Expected E=" << h_top_experts_ref[i] << " (W=" << h_routing_weights_ref[i] << ")"
                      << " | Actual E=" << h_top_experts[i] << " (W=" << h_routing_weights[i] << ")" << std::endl;
        }
    }

    double get_bandwidth_bytes() override {
        return num_tokens * num_experts * sizeof(float) + num_tokens * K * sizeof(int) + num_tokens * K * sizeof(float);
    }

    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);

    // Number of tokens
    std::vector<TestSize<1>> correctness_sizes = {
        {{1}}, {{1000}}, {{65536}}
    };
    
    std::vector<TestSize<1>> perf_sizes = {
        {{1048576}}, {{16777216}}
    };

    run_test_suite<1, MoERoutingTest>("Exercise 32: MoE Top-K Routing", config, correctness_sizes, perf_sizes);
    return 0;
}
