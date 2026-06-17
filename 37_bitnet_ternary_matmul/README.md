# Exercise 37: BitNet Ternary MatMul

## Problem Description
**The Problem:** Implement a ternary matrix multiplication $C = A \times W$.
- $A$ is an $M \times K$ float matrix.
- $W$ is a $K \times N$ ternary matrix, heavily packed.
Since the ternary weights require only 2 bits (we encode `00`=0, `01`=+1, `10`=-1), we pack 4 weights into a single `uint8_t` along the $K$ dimension. 
Your kernel must:
1. Load a single packed byte of $W$.
2. Extract the 4 weights using bitshifts and masks.
3. For each weight, conditionally add or subtract the corresponding element from $A$ to the accumulator.
4. Write the final sum to $C$.

**Sample Input/Output:** 
For $M=1, N=1, K=4$:

**Inputs:**
A = [1.5, 2.0, 3.5, 4.0]
W_packed = [0b10000110] (Binary representation: 10_00_01_10)

Let's unpack `0b10000110` from right to left (bits 0-1, 2-3, 4-5, 6-7):
- Bits 0-1: `10` $
ightarrow$ Weight = -1
- Bits 2-3: `01` $
ightarrow$ Weight = +1
- Bits 4-5: `00` $
ightarrow$ Weight = 0
- Bits 6-7: `10` $
ightarrow$ Weight = -1

Unpacked weights: [-1, 1, 0, -1]

**Computation:**
$C[0, 0] = (1.5 \times -1) + (2.0 \times 1) + (3.5 \times 0) + (4.0 \times -1)$
$C[0, 0] = -1.5 + 2.0 + 0.0 - 4.0 = -3.5$

**Output:**
C = [-3.5]

**Practical Importance:** By reducing weights to 1.58 bits, memory footprint is drastically slashed (an 8x reduction compared to 16-bit floats). More importantly, the power consumption and silicon area required for ALUs are heavily reduced since complex floating-point multipliers can be swapped for simple adders. If hardware fully adapts to ternary operations, future GPUs could run BitNet models 10x faster and cheaper.

**Historical Anecdotes:** The quest to remove multiplication from neural networks has been a holy grail for decades (e.g., BinaryConnect in 2015). However, earlier attempts severely degraded the intelligence of large models. In early 2024, Microsoft Research released the "1.58-bit" paper, proving that by adding a `0` value to standard binary weights (creating a ternary $\{-1, 0, 1\}$ system), models could match the perplexity and performance of full-precision Llama models. It sent shockwaves through the community, promising a new era of "Multiplication-Free" LLMs.

**References:** 
1. *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits* by Shuming Ma et al. (2024) - The seminal paper introducing ternary weights that eliminate multiplication.
2. *BitNet: Scaling 1-bit Transformers for Large Language Models* - The predecessor 1-bit paper.

> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_bitnet_matmul`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
Multiplication is one of the most expensive operations on modern hardware. In large language models (LLMs), matrix multiplication (MatMul) dominates the computation. What if we could completely eliminate multiplications? 
BitNet (specifically 1.58-bit BitNet) achieves this by constraining the model's weights to only three possible values: $\{-1, 0, 1\}$. When you multiply a number by 1, it's just an addition. When you multiply by -1, it's a subtraction. And multiplying by 0 does nothing!
In this exercise, you will implement a GPU kernel that reads heavily packed weights (4 weights crammed into a single byte). Instead of using expensive FMA (Fused Multiply-Add) instructions, your kernel will unpack these weights using bitwise operations and perform simple additions or subtractions. This is a glimpse into the future of ultra-efficient AI hardware!

## Objective
Implement a kernel for BitNet Ternary MatMul.
You will learn about:
- Advanced kernel design and warp-level primitives.
- Hardware constraints of the GPU.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized reference implementation of the kernel. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `bitnet_ternary_matmul_kernel`.
3. Implement `launch_bitnet_ternary_matmul` to configure the grid and block dimensions, and launch your kernel.
4. Make sure `launch_bitnet_ternary_matmul` returns a `std::vector<LaunchConfig>` containing the configurations for all kernels launched so the test framework can automatically display your kernel's hardware utilization.
5. Compile using `make` and run `./bin/run_test.sh` to evaluate your correctness and performance, and `./bin/run_profiler.sh` to identify bottlenecks.

## Typical Commands
The test suite executables in `./bin/` support various command line arguments to help you analyze and debug your kernel. You can use `./bin/run_test.sh` to run the kernel normally, or `./bin/run_profiler.sh` to run it under Nsight Compute for detailed profiling.

- **Help Menus**:
  ```bash
  make && ./bin/run_test.sh -h
  make && ./bin/run_profiler.sh -h
  ```
- **Test a Specific Size** (tests the predefined size closest to the given value, preferring the higher value in a tie):
  ```bash
  make && ./bin/run_test.sh --size 16777216
  ```
- **Profile Memory Metrics** (runs Nsight Compute with predefined metrics for memory-bound kernels):
  ```bash
  make && ./bin/run_profiler.sh --ncu_argset=memory --size 16777216
  ```
- **Test Sizes Above a Threshold**:
  ```bash
  make && ./bin/run_test.sh --above 1048576
  ```
- **Enable Verbose Tracing** (prints detailed setup and launch trace logs):
  ```bash
  make && ./bin/run_test.sh --verbose
  ```
- **Run the Reference Kernel Only** (verifies correctness of the reference kernel without testing or benchmarking your implementation):
  ```bash
  make && ./bin/run_test.sh --test_ref_kernel_only
  ```
