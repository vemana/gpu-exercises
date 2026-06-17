# Exercise 40: Fused SwiGLU MLP

## Problem Description
**The Problem:** Implement a fused SwiGLU kernel.
- Inputs: `X` and `gate`, both arrays of floats of size `total_elements`.
- Output: `O`, an array of floats of the same size.
- The mathematical operation for every element $i$ is:
  $$O[i] = X[i] \times \text{SiLU}(X[i]) \times gate[i]$$
  where $\text{SiLU}(x) = rac{x}{1 + \exp(-x)}$.
  
Your kernel should read $X[i]$ and $gate[i]$ from global memory exactly once, compute the result in registers, and write it to $O[i]$ exactly once.

**Sample Input/Output:** 
Let $total\_elements = 2$.
**Inputs:**
$X = [1.0, -1.0]$
$gate = [2.0, 0.5]$

**Computation for index 0:**
$X[0] = 1.0$
$\text{SiLU}(1.0) = rac{1.0}{1.0 + \exp(-1.0)} = rac{1.0}{1.0 + 0.367879} = rac{1.0}{1.367879} = 0.731058$
$O[0] = \text{SiLU}(1.0) \times gate[0] = 0.731058 \times 2.0 = 1.462116$

**Computation for index 1:**
$X[1] = -1.0$
$\text{SiLU}(-1.0) = rac{-1.0}{1.0 + \exp(1.0)} = rac{-1.0}{1.0 + 2.71828} = rac{-1.0}{3.71828} = -0.26894$
$O[1] = \text{SiLU}(-1.0) \times gate[1] = -0.26894 \times 0.5 = -0.13447$

**Output:**
$O = [1.4621, -0.1345]$

**Practical Importance:** SwiGLU is universally used in almost every state-of-the-art open-weights model today (Llama-1, 2, 3; Mistral, Mixtral, Qwen, etc.). Because it involves very little math but massive amounts of data moving around, it is "Memory Bandwidth Bound". Fusing this kernel eliminates roundtrips to HBM (High Bandwidth Memory), offering an immediate and significant speedup for the entire LLM. A senior ML Systems engineer will optimize this further using `float4` vectorized loads/stores to saturate the memory bus perfectly.

**Historical Anecdotes:** Noam Shazeer is a legendary figure in AI, being a core author of the original *Attention Is All You Need* paper. Years later, he decided to test a seemingly arbitrary set of gating functions on Transformer MLPs. He empirically found that SwiGLU was consistently slightly better than the rest. The improvement wasn't purely theoretical—it just worked better in practice. From that point on, nearly the entire industry quietly swapped out their standard ReLU/GELU layers for SwiGLU, and it became a defining signature of the modern LLM architecture.

**References:** 
1. *GLU Variants Improve Transformer* by Noam Shazeer (2020) - The paper that systematically proved that GLU variants (like SwiGLU) yield superior language models.
2. *Gaussian Error Linear Units (GELUs)* - A related foundational activation function paper.

> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_swiglu`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
In older neural networks, a Multi-Layer Perceptron (MLP) consisted of two Linear layers with a ReLU activation in between. In modern LLMs (like Llama-3 and Mistral), this has been entirely replaced by the SwiGLU block.
SwiGLU involves expanding the input into *two* separate high-dimensional vectors. One vector goes through the Swish (or SiLU) activation function, and is then multiplied element-wise by the other "Gate" vector. 
If we write this in PyTorch without a fused kernel, the GPU must write the huge vector to its memory, run the SiLU kernel, write it to memory again, load the gate vector from memory, multiply them, and write the final result back to memory. Memory transfers are incredibly slow!
Kernel "fusion" fixes this by doing all of those steps in a single pass entirely within the ultrafast registers of the Streaming Multiprocessors.

## Objective
Implement a kernel for Fused SwiGLU MLP.
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
2. Implement the `fused_swiglu_kernel`.
3. Implement `launch_fused_swiglu` to configure the grid and block dimensions, and launch your kernel.
4. Make sure `launch_fused_swiglu` returns a `std::vector<LaunchConfig>` containing the configurations for all kernels launched so the test framework can automatically display your kernel's hardware utilization.
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
