# GPU Basics: Foundational CUDA Algorithms

Welcome! This repository contains a structured series of exercises designed to teach you foundational algorithms used in GPU kernel programming, focusing on **CUDA** and **NVIDIA GPUs**. 

Instead of just learning syntax, these exercises walk you through the core parallel patterns (like Map, Reduce, Scan, Sort, Stencil, and more) that power the world's most advanced GPU applications—from real-time graphics to massive deep learning models.

## Prerequisites

To build and run these exercises, you will need the following installed on your system:
- **NVIDIA GPU** with compatible drivers installed.
- **CUDA Toolkit** (which provides `nvcc`, the NVIDIA CUDA Compiler).
- **NVIDIA NSight Compute** (NCU) which provides `ncu` for profiling kernels
- **A System C++ Compiler** (like `gcc` or `clang`) that is compatible with your installed CUDA version.
- **Make** (for building the executables using the provided `Makefile`s).

## How the Exercises Are Structured

Every exercise in this repository represents a distinct foundational algorithmic pattern. They are designed so you can focus entirely on writing high-performance kernels without worrying about the boilerplate of setting up test harnesses, data generation, or CPU baselines.

Each exercise directory contains:
1. **Your Workspace (`kernel.cu`)**: You are tasked with implementing a specific kernel (or sequence of kernels) to solve the problem.
2. **The Scaffold / Test Harness**: A pre-built `main.cc` that handles compiling, running, and verifying your kernel. 
3. **The Reference Kernel (`reference_kernel.cu`)**: A highly optimized reference implementation. The scaffold automatically compares your implementation's correctness and performance against this gold standard.
4. **Occupancy & Utilization Metrics**: As you test your kernel, the test scaffold will output detailed hardware utilization metrics (like Shared Memory limits, Register bottlenecks, and Block allocations per Streaming Multiprocessor). This immediate feedback loop is designed to help you analyze *why* your kernel is slow and how to tweak your grid sizing or memory accesses to improve it.

## Getting Started

To get a feel for how everything works, start with the simplest parallel pattern: **Vector Addition (Map)**.

👉 **[Click here to open the Map Exercise and begin!](01_map/README.md)**

### Available Exercises

1. [Map](01_map/README.md)
2. [Reduce](02_reduce/README.md)
3. [Scan](03_scan/README.md)
4. [Histogram](04_histogram/README.md)
5. [Stencil](05_stencil/README.md)
6. [Scatter Gather](06_scatter_gather/README.md)
7. [GEMM](07_gemm/README.md)
8. [Stream Compaction](08_stream_compaction/README.md)
9. [Radix Sort](09_radix_sort/README.md)
10. [BFS](10_bfs/README.md)
11. [SpMV](11_spmv/README.md)
12. [Segmented Scan](12_segmented_scan/README.md)
13. [Merge Sort](13_merge_sort/README.md)
14. [N-Body](14_nbody/README.md)
15. [Conv2D](15_conv2d/README.md)
16. [Activation](16_activation/README.md)
17. [Max Pool 2D](17_max_pool2d/README.md)
18. [Layer Norm](18_layer_norm/README.md)
19. [Batch Norm](19_batch_norm/README.md)
20. [Softmax](20_softmax/README.md)
21. [Im2Col](21_im2col/README.md)
22. [Depthwise Conv2D](22_depthwise_conv2d/README.md)
23. [Dropout](23_dropout/README.md)
24. [RoPE](24_rope/README.md)
25. [Flash Attention](25_flash_attention/README.md)
26. [Jacobi Iteration](26_jacobi_iteration/README.md)
27. [Fast Fourier Transform](27_fft/README.md)
28. [Conjugate Gradient](28_conjugate_gradient/README.md)
29. [Molecular Dynamics](29_molecular_dynamics/README.md)
30. [Monte Carlo Integration](30_monte_carlo/README.md)
31. [PagedAttention](31_paged_attention/README.md)
32. [MoE Routing](32_moe_routing/README.md)
33. [Grouped-Query Attention](33_grouped_query_attention/README.md)
34. [Speculative Decoding](34_speculative_decoding/README.md)
35. [W8A16 Linear](35_w8a16_linear/README.md)

Sample output from implementing the addition map kernel in a straightforward manner. Run as `make && bin/run_test.sh`
```
================================================================================
                              TEST SIZE: 1,048,576
================================================================================

--- Occupancy & Utilization: Student Kernel: map_kernel ---
Context: Problem Size = 1,048,576, Grid Size = 4,096, Block Size = 256, Dynamic Shared Mem = 0 bytes
Metric                   Kernel Usage/Block  Hardware Limit/SM   Bottleneck Blocks   Utilization    
----------------------------------------------------------------------------------------------------
Blocks                   1                   16                  16                  37.50%
Threads                  256                 1536                6                   100.00%
Shared Memory (Bytes)    0                   102400              N/A                 0.00%
Registers                6656                65536               9                   60.94%

Overall Active Blocks per SM: 6 (Limited by: Threads/SM)
----------------------------------------------------------------------------------------------------

[Size       1,048,576] Performance:
  Reference Kernel : 0.03 ms (409.60 GB/s)
  Student Kernel   : 0.03 ms (491.52 GB/s)
  Speedup (Student/Teacher) : 1.20x


================================================================================
                             TEST SIZE: 16,777,216
================================================================================

--- Occupancy & Utilization: Student Kernel: map_kernel ---
Context: Problem Size = 16,777,216, Grid Size = 65,536, Block Size = 256, Dynamic Shared Mem = 0 bytes
Metric                   Kernel Usage/Block  Hardware Limit/SM   Bottleneck Blocks   Utilization    
----------------------------------------------------------------------------------------------------
Blocks                   1                   16                  16                  37.50%
Threads                  256                 1536                6                   100.00%
Shared Memory (Bytes)    0                   102400              N/A                 0.00%
Registers                6656                65536               9                   60.94%

Overall Active Blocks per SM: 6 (Limited by: Threads/SM)
----------------------------------------------------------------------------------------------------

[Size      16,777,216] Performance:
  Reference Kernel : 0.26 ms (774.05 GB/s)
  Student Kernel   : 0.24 ms (826.08 GB/s)
  Speedup (Student/Teacher) : 1.07x


================================================================================
                             TEST SIZE: 67,108,864
================================================================================

--- Occupancy & Utilization: Student Kernel: map_kernel ---
Context: Problem Size = 67,108,864, Grid Size = 262,144, Block Size = 256, Dynamic Shared Mem = 0 bytes
Metric                   Kernel Usage/Block  Hardware Limit/SM   Bottleneck Blocks   Utilization    
----------------------------------------------------------------------------------------------------
Blocks                   1                   16                  16                  37.50%
Threads                  256                 1536                6                   100.00%
Shared Memory (Bytes)    0                   102400              N/A                 0.00%
Registers                6656                65536               9                   60.94%

Overall Active Blocks per SM: 6 (Limited by: Threads/SM)
----------------------------------------------------------------------------------------------------

[Size      67,108,864] Performance:
  Reference Kernel : 0.97 ms (829.57 GB/s)
  Student Kernel   : 0.95 ms (852.04 GB/s)
  Speedup (Student/Teacher) : 1.03x


================================================================================
                               STUDENT KERNELS
================================================================================

--- map_kernel ---
Problem Size             Block Util    Thread Util      SMem Util       Reg Util
--------------------------------------------------------------------------------
1,048,576                    37.50%        100.00%          0.00%         60.94%
16,777,216                   37.50%        100.00%          0.00%         60.94%
67,108,864                   37.50%        100.00%          0.00%         60.94%

================================================================================
                               SPEEDUP SUMMARY
================================================================================
Problem Size                Speedup        Student Time      Reference Time
--------------------------------------------------------------------------------
1,048,576                     1.20x             0.03 ms             0.03 ms
16,777,216                    1.07x             0.24 ms             0.26 ms
67,108,864                    1.03x             0.95 ms             0.97 ms

```

If you run with `make && bin/run_profiler.sh`, you get more detailed, lower level stats via Nsight NCU.
```

  void cub::static_kernel<cub::policy_350_t, long, thrust::binary_transform_f<thrust::device_ptr<const float>, thrust::device_ptr<const float>, thrust::device_ptr<float>, thrust::no_stencil_tag, thrust::plus<float>, thrust::always_true_predicate>>(T2, T3) (131072, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: Command line profiler metrics
    ------------------------------------------------------------ ----------- ------------
    Metric Name                                                  Metric Unit Metric Value
    ------------------------------------------------------------ ----------- ------------
    dram__bytes.sum                                                    Mbyte       804.48
    dram__throughput.avg.pct_of_peak_sustained_elapsed                     %        93.20
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                    sector   16,777,216
    l2__throughput.avg.pct_of_peak_sustained_elapsed                              (!) n/a
    sm__cycles_active.avg.pct_of_peak_sustained_elapsed                    %        99.67
    sm__inst_executed.avg.per_cycle_active                        inst/cycle         0.28
    sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed           %         1.69
    smsp__inst_executed.avg.per_cycle_active                      inst/cycle         0.07
    ------------------------------------------------------------ ----------- ------------

  map_kernel(const float *, const float *, float *, int) (262144, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: Command line profiler metrics
    ------------------------------------------------------------ ----------- ------------
    Metric Name                                                  Metric Unit Metric Value
    ------------------------------------------------------------ ----------- ------------
    dram__bytes.sum                                                    Mbyte       804.44
    dram__throughput.avg.pct_of_peak_sustained_elapsed                     %        93.82
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                    sector   16,777,216
    l2__throughput.avg.pct_of_peak_sustained_elapsed                              (!) n/a
    sm__cycles_active.avg.pct_of_peak_sustained_elapsed                    %        99.71
    sm__inst_executed.avg.per_cycle_active                        inst/cycle         1.04
    sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed           %        10.24
    smsp__inst_executed.avg.per_cycle_active                      inst/cycle         0.26
    ------------------------------------------------------------ ----------- ------------


================================================================================
                            Nsight Compute Metric Guide                         
================================================================================
dram__bytes.sum                                                            
  # of bytes accessed in DRAM

dram__throughput.avg.pct_of_peak_sustained_elapsed                         
  DRAM throughput as a percentage of peak sustained elapsed cycles

l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                             
  # of L1 global load sectors

l2__throughput.avg.pct_of_peak_sustained_elapsed                           
  L2 cache throughput as a percentage of peak sustained elapsed cycles

sm__cycles_active.avg.pct_of_peak_sustained_elapsed                        
  SM active cycles as a percentage of peak sustained elapsed cycles

sm__inst_executed.avg.per_cycle_active                                     
  Average number of instructions executed per SM active cycle

smsp__inst_executed.avg.per_cycle_active                                   
  Average number of instructions executed per SMSP active cycle

sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed               
  FMA pipe active cycles as a percentage of peak sustained elapsed cycles
```

## Acknowledgements

All the code in this repository has been generated and refined using **Antigravity** and **Gemini 3.1 Pro**.

- [Exercise 36: Split-K Flash Decoding](./36_split_k_flash_decoding) - Optimizing LLM generation via Split-K.
- [Exercise 37: BitNet Ternary MatMul](./37_bitnet_ternary_matmul) - Extreme 1.58-bit ternary quantized networks.
- [Exercise 38: Parallel Associative Scan](./38_parallel_associative_scan) - Mamba / State Space Models recurrence.
- [Exercise 39: Sliding Window Attention](./39_sliding_window_attention) - Banded block-sparse attention (Mistral).
- [Exercise 40: Fused SwiGLU MLP](./40_fused_swiglu) - Memory-bandwidth optimized activation functions.
