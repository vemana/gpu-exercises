
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/argparse.h"
#include "../utils/framework.h"
#include "../utils/tracer.h"
#include "kernel.h"
#include "reference_kernel.h"

void cpu_speculative_decoding(
    const std::vector<float>& target_probs, const std::vector<float>& draft_probs, const std::vector<float>& random_vals,
    std::vector<int>& accept_lengths, int num_batches, int draft_len) {
    
    for (int b = 0; b < num_batches; ++b) {
        int accept_len = draft_len;
        for (int i = 0; i < draft_len; ++i) {
            int idx = b * draft_len + i;
            float p = target_probs[idx];
            float q = draft_probs[idx];
            float r = random_vals[idx];
            
            if (r >= p / q) {
                accept_len = i;
                break;
            }
        }
        accept_lengths[b] = accept_len;
    }
}

struct SpecDecTest : public ProblemTest<1> {
    int num_batches;
    int draft_len = 8;
    
    std::vector<float> h_target_probs, h_draft_probs, h_random_vals;
    std::vector<int> h_accept_lengths, h_accept_lengths_ref;

    float *d_target_probs = nullptr, *d_draft_probs = nullptr, *d_random_vals = nullptr;
    int *d_accept_lengths = nullptr;

    SpecDecTest(const TestSize<1>& size) : ProblemTest<1>(size) {
        num_batches = size.dims[0];
    }

    void generate_test_data(bool check) override {
        h_target_probs.resize(num_batches * draft_len);
        h_draft_probs.resize(num_batches * draft_len);
        h_random_vals.resize(num_batches * draft_len);
        h_accept_lengths.resize(num_batches, 0);
        h_accept_lengths_ref.resize(num_batches, 0);
        
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> prob_dist(0.01f, 0.99f);
        std::uniform_real_distribution<float> r_dist(0.0f, 1.0f);
        
        for (float& val : h_target_probs) val = prob_dist(gen);
        for (float& val : h_draft_probs) val = prob_dist(gen);
        for (float& val : h_random_vals) val = r_dist(gen);

        if (check) {
            cpu_speculative_decoding(h_target_probs, h_draft_probs, h_random_vals, h_accept_lengths_ref, num_batches, draft_len);
        }
    }

    void setup_reference() override {
        cudaMalloc(&d_target_probs, h_target_probs.size() * sizeof(float));
        cudaMalloc(&d_draft_probs, h_draft_probs.size() * sizeof(float));
        cudaMalloc(&d_random_vals, h_random_vals.size() * sizeof(float));
        cudaMalloc(&d_accept_lengths, num_batches * sizeof(int));
        
        cudaMemcpy(d_target_probs, h_target_probs.data(), h_target_probs.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_draft_probs, h_draft_probs.data(), h_draft_probs.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_random_vals, h_random_vals.data(), h_random_vals.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_speculative_decoding_reference(
            d_target_probs, d_draft_probs, d_random_vals, d_accept_lengths, num_batches, draft_len);
    }

    void teardown_reference() override {
        cudaFree(d_target_probs); cudaFree(d_draft_probs);
        cudaFree(d_random_vals); cudaFree(d_accept_lengths);
    }

    void setup_student() override {
        cudaMalloc(&d_target_probs, h_target_probs.size() * sizeof(float));
        cudaMalloc(&d_draft_probs, h_draft_probs.size() * sizeof(float));
        cudaMalloc(&d_random_vals, h_random_vals.size() * sizeof(float));
        cudaMalloc(&d_accept_lengths, num_batches * sizeof(int));
        
        cudaMemcpy(d_target_probs, h_target_probs.data(), h_target_probs.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_draft_probs, h_draft_probs.data(), h_draft_probs.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_random_vals, h_random_vals.data(), h_random_vals.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_speculative_decoding(
            d_target_probs, d_draft_probs, d_random_vals, d_accept_lengths, num_batches, draft_len);
    }

    void teardown_student() override {
        cudaFree(d_target_probs); cudaFree(d_draft_probs);
        cudaFree(d_random_vals); cudaFree(d_accept_lengths);
    }

    CorrectnessResult verify() override {
        cudaMemcpy(h_accept_lengths.data(), d_accept_lengths, num_batches * sizeof(int), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < num_batches; ++i) {
            if (h_accept_lengths_ref[i] != h_accept_lengths[i]) {
                return {false, i, (float)h_accept_lengths_ref[i], (float)h_accept_lengths[i]};
            }
        }
        return {true, -1, 0.0f, 0.0f};
    }

    void print_mismatch() override {
        std::cout << "--- Expected vs Actual ---" << std::endl;
        for (int i = 0; i < std::min(num_batches, 20); ++i) {
            std::cout << "Batch " << i << ": Expected Length " << h_accept_lengths_ref[i] << " | Actual " << h_accept_lengths[i] << std::endl;
        }
    }

    double get_bandwidth_bytes() override {
        return num_batches * draft_len * sizeof(float) * 3 + num_batches * sizeof(int);
    }

    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);

    // Number of batches (sequences evaluating drafts)
    std::vector<TestSize<1>> correctness_sizes = {
        {{1}}, {{1000}}, {{65536}}
    };
    
    std::vector<TestSize<1>> perf_sizes = {
        {{1048576}}, {{16777216}}
    };

    run_test_suite<1, SpecDecTest>("Exercise 34: Speculative Decoding (Parallel Rejection Sampling)", config, correctness_sizes, perf_sizes);
    return 0;
}
