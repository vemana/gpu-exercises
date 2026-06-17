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

1. **Your Workspace (`kernel.cu`)**: You are tasked with implementing a specific kernel (or a
   collection of kernels) to solve the problem.
2. **The Scaffold / Test Harness**: `main.cc` handles compiling, running, verifying and timing your
   kernel.
3. **The CPU Baseline (`main.cc`)**: CPU implementation of the algorithm. The scaffold checks the
   reference CUDA kernel (see below) against this CPU baseline for correctness.
4. **The Reference Kernel (`reference_kernel.cu`)**: An optimized reference implementation. The
   scaffold compares your kernel's correctness and performance against it. Note that the CPU kernel
   would be too slow on larger problem sizes; so, your kernel is evaluated against the reference
   kernel, which in turn is verified against the cpu implementation.
5. **Occupancy & Utilization Metrics**: If you ask for it, the test scaffold outputs detailed
   hardware utilization metrics (like Shared Memory limits, Register bottlenecks, and Block
   allocations per Streaming Multiprocessor). This immediate feedback loop is designed to help you
   analyze *why* your kernel is slow and how to tweak your grid sizing or memory accesses to improve
   it. See the sample commands in the exercises for various ways to run your kernels.

## Getting Started

To get a feel for how everything works, start with the simplest parallel pattern: **Vector Addition (Map)**.

👉 **[Click here to open the Map Exercise and begin!](01_map/README.md)**

### Available Exercises

| Exercise | Problem Description | Primary Skills Learnt |
| --- | --- | --- |
| 1. [Map](01_map/README.md) | Point-wise vector addition | Launching kernels, thread IDs, memory coalescing |
| 2. [Reduce](02_reduce/README.md) | Parallel sum reduction | Shared memory, tree reduction, warp-level primitives |
| 3. [Scan](03_scan/README.md) | Inclusive/exclusive prefix sum | Work-efficient algorithms, Kogge-Stone/Blelloch, double buffering |
| 4. [Histogram](04_histogram/README.md) | Frequency counting of elements | Atomic operations, privatization in shared memory |
| 5. [Stencil](05_stencil/README.md) | 1D/2D grid-based neighborhood operations | Ghost cells, 2D blocks, shared memory caching |
| 6. [Scatter Gather](06_scatter_gather/README.md) | Indirect memory access patterns | Dealing with uncoalesced memory, sparse data access |
| 7. [GEMM](07_gemm/README.md) | General Matrix Multiplication | Tiling, register blocking, memory bus saturation |
| 8. [Stream Compaction](08_stream_compaction/README.md) | Filtering elements matching a condition | Scan + Scatter, predication |
| 9. [Radix Sort](09_radix_sort/README.md) | Sorting elements by bit-level keys | Scan, scatter, bitwise operations |
| 10. [BFS](10_bfs/README.md) | Breadth-First Search on a graph | Frontier queues, atomic queues, dynamic parallelism |
| 11. [SpMV](11_spmv/README.md) | Sparse Matrix-Vector Multiplication | CSR/COO formats, warp divergence handling |
| 12. [Segmented Scan](12_segmented_scan/README.md) | Independent prefix sums across subarrays | Head flags, warp shuffling, block-level synchronization |
| 13. [Merge Sort](13_merge_sort/README.md) | Parallel sorting algorithm | Merge path algorithm, binary search on GPU |
| 14. [N-Body](14_nbody/README.md) | All-pairs gravitational interactions | Tiling to reduce memory bandwidth, math pipelining |
| 15. [Conv2D](15_conv2d/README.md) | 2D Image Convolution | 2D tiling, constant memory for filters |
| 16. [Activation](16_activation/README.md) | Neural network non-linearities (e.g., ReLU) | Element-wise operations, vectorized loads (float4) |
| 17. [Max Pool 2D](17_max_pool2d/README.md) | Spatial downsampling for CNNs | Windowed reduction, boundary/edge handling |
| 18. [Layer Norm](18_layer_norm/README.md) | Normalization across feature dimension | Block-wide reductions, numerical stability (Welford's) |
| 19. [Batch Norm](19_batch_norm/README.md) | Normalization across batch dimension | Global reduction, atomic adds vs two-pass algorithms |
| 20. [Softmax](20_softmax/README.md) | Probability distribution scaling | Max reduction, sum reduction, fast math intrinsics |
| 21. [Im2Col](21_im2col/README.md) | Image to column transformation for convolutions | Tensor reshaping, memory coalescing for GEMM |
| 22. [Depthwise Conv2D](22_depthwise_conv2d/README.md) | Independent channel convolution | Specialized 2D kernels, avoiding memory bounds |
| 23. [Dropout](23_dropout/README.md) | Randomized neural network regularization | Philox random number generation, bit packing |
| 24. [RoPE](24_rope/README.md) | Rotary Positional Embeddings for Transformers | Complex numbers, sinusoidal embedding, token-wise parallelism |
| 25. [Flash Attention](25_flash_attention/README.md) | Memory-efficient exact attention | Tiling the $O(N^2)$ matrix, online softmax, SRAM management |
| 26. [Jacobi Iteration](26_jacobi_iteration/README.md) | Iterative PDE solver | CUDA Graphs, multi-step synchronization |
| 27. [Fast Fourier Transform](27_fft/README.md) | Signal processing frequency domain | Cooley-Tukey, bit-reversal, shared memory bank conflicts |
| 28. [Conjugate Gradient](28_conjugate_gradient/README.md) | Sparse linear solver | SpMV + Dot product pipelines, cuBLAS integration |
| 29. [Molecular Dynamics](29_molecular_dynamics/README.md) | Particle simulation with cutoff radii | Cell-linked lists, spatial hashing, neighbor lists |
| 30. [Monte Carlo Integration](30_monte_carlo/README.md) | Stochastic area simulation | cuRAND, massive parallel reduction |
| 31. [PagedAttention](31_paged_attention/README.md) | LLM KV Cache memory management | Block tables, non-contiguous memory management |
| 32. [MoE Routing](32_moe_routing/README.md) | Mixture of Experts token routing | Top-K algorithms, sorting, dynamic dispatch |
| 33. [Grouped-Query Attention](33_grouped_query_attention/README.md) | LLM inference attention mechanism | Broadcast semantics, shared KV cache, memory bandwidth optimization |
| 34. [Speculative Decoding](34_speculative_decoding/README.md) | Fast LLM multi-token decoding | Prefix matching, parallel verification |
| 35. [W8A16 Linear](35_w8a16_linear/README.md) | Quantized matrix math | Mixed precision, `dp4a` dot product, dequantization |
| 36. [Split-K Flash Decoding](36_split_k_flash_decoding/README.md) | Long-context LLM inference | Sequence dimension splitting, parallel global reductions |
| 37. [BitNet Ternary MatMul](37_bitnet_ternary_matmul/README.md) | Extreme 1.58-bit ternary quantized networks | Bit packing, sub-byte extraction, avoiding FMA units |
| 38. [Parallel Associative Scan](38_parallel_associative_scan/README.md) | Mamba / State Space Models recurrence | Custom associative operators, prefix scan for sequential dependencies |
| 39. [Sliding Window Attention](39_sliding_window_attention/README.md) | Banded block-sparse attention (Mistral) | Banded diagonal masking, loop boundary optimization |
| 40. [Fused SwiGLU MLP](40_fused_swiglu/README.md) | Memory-bandwidth optimized activation functions | Kernel fusion, bandwidth bound optimization |

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


## Is this AI slop?

Since most of this code is generated via **Antigravity** it's natural to ask what of this repo 
is actually believable and what is not. By comparing the reference kernel impl against the 
cpu implementation, there's a good chance that the reference kernel is working as expected. The 
two models of computation are so widely far apart that a coincidence is unlikely. Besides, as 
the exercises are attempted, any bugs will get ironed out.
