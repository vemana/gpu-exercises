#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include "tracer.h"

struct CorrectnessResult {
    bool passed;
    int error_index;
    float expected_val;
    float actual_val;
};

inline CorrectnessResult check_correctness(const float* expected, const float* actual, int size, float tolerance = 1e-1) {
    for (int i = 0; i < size; ++i) {
        float diff = std::abs((float)expected[i] - (float)actual[i]);
        if (diff > tolerance) {
            return {false, i, (float)expected[i], (float)actual[i]};
        }
    }
    return {true, -1, 0.0f, 0.0f};
}

inline CorrectnessResult check_correctness(const int* expected, const int* actual, int size, int tolerance = 0) {
    for (int i = 0; i < size; ++i) {
        int diff = std::abs(expected[i] - actual[i]);
        if (diff > tolerance) {
            return {false, i, (float)expected[i], (float)actual[i]};
        }
    }
    return {true, -1, 0.0f, 0.0f};
}

inline void print_correctness_result(const CorrectnessResult& result) {
    if (!result.passed) {
        std::cerr << "Mismatch at index " << result.error_index 
                  << ": expected " << result.expected_val 
                  << ", got " << result.actual_val << std::endl;
        std::cout << "\n\n"
                  << "\x1b[1;31m" // Bold Red
                  << "================================================================================\n"
                  << "                         🚨 CORRECTNESS TEST FAILED! 🚨                         \n"
                  << "================================================================================\n"
                  << "\x1b[0m" // Reset
                  << "\n";
    } else {
        std::cout << "Correctness test passed!" << std::endl;
    }
}

inline std::string format_with_commas(long long value) {
    std::string s = std::to_string(value);
    int n = s.length() - 3;
    int start = (value < 0) ? 1 : 0;
    while (n > start) {
        s.insert(n, ",");
        n -= 3;
    }
    return s;
}

template <typename T>
inline void print_array(const std::string& name, const T* arr, int size, int cols_per_row = 20) {
    std::cout << "--- " << name << " ---" << std::endl;
    for (int i = 0; i < size; ++i) {
        std::cout << std::setw(8) << arr[i] << " ";
        if ((i + 1) % cols_per_row == 0) {
            std::cout << std::endl;
        }
    }
    if (size % cols_per_row != 0) {
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

struct OccupancyMetrics {
    bool is_dummy = false;
    int block_size;
    size_t dynamic_smem;
    
    int active_blocks;
    int max_blocks_by_blocks;
    int max_blocks_by_threads;
    int max_blocks_by_smem;
    int max_blocks_by_regs;
    
    int prop_max_blocks;
    int prop_max_threads;
    size_t prop_max_smem;
    int prop_max_regs;
    
    size_t smem_per_block;
    int regs_per_block;
    
    float block_util;
    float thread_util;
    float smem_util;
    float reg_util;
};

struct LaunchConfig {
    std::string kernel_name;
    const void* kernel_func;
    int blocksPerGrid;
    int threadsPerBlock;
    size_t dynamicSmemBytes;
};

inline OccupancyMetrics calculate_occupancy(const void* kernel_func, int block_size, size_t dynamic_smem) {
    if (kernel_func == nullptr) {
        OccupancyMetrics m;
        m.is_dummy = true;
        return m;
    }
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    
    cudaFuncAttributes attr;
    if (err != cudaSuccess || prop.warpSize == 0) {
        prop.warpSize = 32;
        prop.sharedMemPerMultiprocessor = 102400; // 100 KB
        prop.maxBlocksPerMultiProcessor = 16; 
        prop.maxThreadsPerMultiProcessor = 1536;
        prop.regsPerMultiprocessor = 65536;
        
        attr.sharedSizeBytes = 0; 
        attr.numRegs = 32;        
    } else {
        cudaFuncGetAttributes(&attr, kernel_func);
    }

    OccupancyMetrics m;
    m.block_size = block_size;
    m.dynamic_smem = dynamic_smem;
    
    m.prop_max_blocks = prop.maxBlocksPerMultiProcessor;
    m.prop_max_threads = prop.maxThreadsPerMultiProcessor;
    m.prop_max_smem = prop.sharedMemPerMultiprocessor;
    m.prop_max_regs = prop.regsPerMultiprocessor;

    int warps_per_block = (block_size + prop.warpSize - 1) / prop.warpSize;
    m.smem_per_block = attr.sharedSizeBytes + dynamic_smem;
    m.regs_per_block = attr.numRegs * prop.warpSize * warps_per_block; 
    
    m.max_blocks_by_blocks = m.prop_max_blocks;
    m.max_blocks_by_threads = m.prop_max_threads / block_size;
    m.max_blocks_by_smem = m.smem_per_block > 0 ? m.prop_max_smem / m.smem_per_block : m.max_blocks_by_blocks;
    m.max_blocks_by_regs = m.regs_per_block > 0 ? m.prop_max_regs / m.regs_per_block : m.max_blocks_by_blocks;
    
    m.active_blocks = std::min({m.max_blocks_by_blocks, m.max_blocks_by_threads, m.max_blocks_by_smem, m.max_blocks_by_regs});

    m.block_util = (float)m.active_blocks / m.max_blocks_by_blocks * 100.0f;
    m.thread_util = (float)(m.active_blocks * block_size) / m.prop_max_threads * 100.0f;
    m.smem_util = m.smem_per_block > 0 ? (float)(m.active_blocks * m.smem_per_block) / m.prop_max_smem * 100.0f : 0.0f;
    m.reg_util = m.regs_per_block > 0 ? (float)(m.active_blocks * m.regs_per_block) / m.prop_max_regs * 100.0f : 0.0f;

    return m;
}

inline void print_occupancy(const char* label, const OccupancyMetrics& m, const std::string& problem_size_str, int grid_size) {
    if (m.is_dummy) return;
    
    global_tracer.trace(std::string("Querying occupancy for ") + label);
    std::cout << "\n--- Occupancy & Utilization: " << label << " ---" << std::endl;
    std::cout << "Context: Problem Size = " << problem_size_str 
              << ", Grid Size = " << format_with_commas(grid_size) 
              << ", Block Size = " << m.block_size 
              << ", Dynamic Shared Mem = " << format_with_commas(m.dynamic_smem) << " bytes" << std::endl;
    std::cout << std::left << std::setw(25) << "Metric" 
              << std::setw(20) << "Kernel Usage/Block" 
              << std::setw(20) << "Hardware Limit/SM" 
              << std::setw(20) << "Bottleneck Blocks" 
              << std::setw(15) << "Utilization" << std::endl;
    std::cout << std::string(100, '-') << std::endl;

    std::cout << std::left << std::setw(25) << "Blocks"
              << std::setw(20) << "1"
              << std::setw(20) << m.prop_max_blocks
              << std::setw(20) << m.max_blocks_by_blocks
              << std::fixed << std::setprecision(2) << m.block_util << "%" << std::endl;

    std::cout << std::left << std::setw(25) << "Threads"
              << std::setw(20) << m.block_size
              << std::setw(20) << m.prop_max_threads
              << std::setw(20) << m.max_blocks_by_threads
              << std::fixed << std::setprecision(2) << m.thread_util << "%" << std::endl;

    std::cout << std::left << std::setw(25) << "Shared Memory (Bytes)"
              << std::setw(20) << m.smem_per_block
              << std::setw(20) << m.prop_max_smem
              << std::setw(20) << (m.smem_per_block > 0 ? std::to_string(m.max_blocks_by_smem) : "N/A")
              << std::fixed << std::setprecision(2) << m.smem_util << "%" << std::endl;

    std::cout << std::left << std::setw(25) << "Registers"
              << std::setw(20) << m.regs_per_block
              << std::setw(20) << m.prop_max_regs
              << std::setw(20) << (m.regs_per_block > 0 ? std::to_string(m.max_blocks_by_regs) : "N/A")
              << std::fixed << std::setprecision(2) << m.reg_util << "%" << std::endl;
              
    std::cout << "\nOverall Active Blocks per SM: " << m.active_blocks << " (Limited by: ";
    if (m.active_blocks == m.max_blocks_by_blocks) std::cout << "Blocks/SM";
    else if (m.active_blocks == m.max_blocks_by_threads) std::cout << "Threads/SM";
    else if (m.active_blocks == m.max_blocks_by_smem) std::cout << "Shared Memory";
    else std::cout << "Registers";
    std::cout << ")\n" << std::string(100, '-') << "\n" << std::endl;
}

#endif
