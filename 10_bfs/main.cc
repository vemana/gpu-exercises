#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <queue>
#include <cuda_runtime.h>
#include "../utils/framework.h"
#include "../utils/argparse.h"
#include "../utils/tracer.h"
#include "kernel.h"
#include "reference_kernel.h"

void cpu_baseline(const int* row_offsets, const int* col_indices, int* distances, int num_nodes, int source_node) {
    for (int i = 0; i < num_nodes; ++i) {
        distances[i] = -1;
    }
    distances[source_node] = 0;
    
    std::vector<int> q;
    q.push_back(source_node);
    int head = 0;
    
    while (head < q.size()) {
        int u = q[head++];
        int start_edge = row_offsets[u];
        int end_edge = row_offsets[u + 1];
        
        for (int i = start_edge; i < end_edge; ++i) {
            int v = col_indices[i];
            if (distances[v] == -1) {
                distances[v] = distances[u] + 1;
                q.push_back(v);
            }
        }
    }
}

struct BFSTest : public ProblemTest<1> {
    std::vector<int> h_row_offsets;
    std::vector<int> h_col_indices;
    std::vector<int> h_distances;
    std::vector<int> h_distances_ref;
    int num_edges = 0;

    int *d_row_offsets = nullptr;
    int *d_col_indices = nullptr;
    int *d_distances = nullptr;

    BFSTest(const TestSize<1>& size) : ProblemTest<1>(size) {}

    void generate_test_data(bool check) override {
        int n = size.dims[0]; // number of nodes
        h_row_offsets.resize(n + 1, 0);
        h_distances.assign(n, -1);
        h_distances_ref.assign(n, -1);

        // Generate a random graph with avg degree 4
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);
        
        std::vector<std::vector<int>> adj(n);
        for (int i = 0; i < n; ++i) {
            // connect to i+1 to ensure connectivity
            if (i < n - 1) adj[i].push_back(i + 1);
            if (i > 0) adj[i].push_back(i - 1);
            
            // random edges
            for (int j = 0; j < 2; ++j) {
                int target = gen() % n;
                adj[i].push_back(target);
            }
        }

        h_row_offsets[0] = 0;
        for (int i = 0; i < n; ++i) {
            for (int v : adj[i]) {
                h_col_indices.push_back(v);
            }
            h_row_offsets[i + 1] = h_col_indices.size();
        }
        num_edges = h_col_indices.size();

        if (check) cpu_baseline(h_row_offsets.data(), h_col_indices.data(), h_distances_ref.data(), n, 0);
    }

    void setup_reference() override {
        int n = size.dims[0];
        cudaMalloc(&d_row_offsets, (n + 1) * sizeof(int));
        cudaMalloc(&d_col_indices, num_edges * sizeof(int));
        cudaMalloc(&d_distances, n * sizeof(int));
        
        cudaMemcpy(d_row_offsets, h_row_offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_indices, h_col_indices.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_distances, 0xFF, n * sizeof(int)); // -1
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_bfs(d_row_offsets, d_col_indices, d_distances, size.dims[0], num_edges, 0);
    }

    void setup_student() override {
        int n = size.dims[0];
        cudaMalloc(&d_row_offsets, (n + 1) * sizeof(int));
        cudaMalloc(&d_col_indices, num_edges * sizeof(int));
        cudaMalloc(&d_distances, n * sizeof(int));
        
        cudaMemcpy(d_row_offsets, h_row_offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_indices, h_col_indices.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_distances, 0xFF, n * sizeof(int)); // -1
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_bfs(d_row_offsets, d_col_indices, d_distances, size.dims[0], num_edges, 0);
    }

    CorrectnessResult verify() override {
        int n = size.dims[0];
        cudaMemcpy(h_distances.data(), d_distances, n * sizeof(int), cudaMemcpyDeviceToHost);
        return check_correctness(h_distances_ref.data(), h_distances.data(), n, 0);
    }

    void print_mismatch() override {
        int n = size.dims[0];
        std::cout << "\n--- Expected Output (First 10) ---\n";
        for (int i = 0; i < std::min(n, 10); ++i) {
            std::cout << std::setw(8) << h_distances_ref[i];
            if ((i + 1) % 10 == 0) std::cout << "\n";
        }
        std::cout << "\n--- Actual Output (First 10) ---\n";
        for (int i = 0; i < std::min(n, 10); ++i) {
            std::cout << std::setw(8) << h_distances[i];
            if ((i + 1) % 10 == 0) std::cout << "\n";
        }
        std::cout << "\n";
    }

    double get_bandwidth_bytes() override {
        // Varies per algorithm, but approx O(V + E) memory accesses
        return size.dims[0] * sizeof(int) + num_edges * sizeof(int);
    }

    void teardown() override {
        if (d_row_offsets) cudaFree(d_row_offsets);
        if (d_col_indices) cudaFree(d_col_indices);
        if (d_distances) cudaFree(d_distances);
    }
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);

    std::vector<TestSize<1>> correctness_sizes = {
        {1}, {31}, {32}, {33}, {63}, {64}, {65}, {1023}, {1024}, {1025}
    };
    
    std::vector<TestSize<1>> perf_sizes = {
        {1048576}, {4194304}, {16777216}
    };

    run_test_suite<1, BFSTest>("Exercise 10: Breadth-First Search (BFS)", config, correctness_sizes, perf_sizes);
    return 0;
}
