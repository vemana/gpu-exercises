# Exercise 35: W8A16 Linear (Weight-Only Quantization)

## Problem Description
**The Problem:** Serving an LLM requires holding the entire model in VRAM. For a 70B parameter model, using `float16` requires ~140GB of memory, which costs thousands of dollars a month to host. However, researchers discovered that you can compress the model weights into 8-bit integers (`int8`) with minimal quality loss, halving the memory requirement and doubling the memory bandwidth speed!

**Sample Input/Output:** 
- Input: We have an `int8` weight chunk `[-50, 127]`. The scaling factor for this group is `0.01`. The input activation vector $X$ is `[2.0, 1.0]`.
- Operation: Dequantize the weights on the fly during the dot product: 
  - $W_0 = -50 \times 0.01 = -0.5$
  - $W_1 = 127 \times 0.01 = 1.27$
  - Dot Product = $(2.0 \times -0.5) + (1.0 \times 1.27) = 0.27$.
- Output: The output feature value $Y = 0.27$.

**Practical Importance:** 
Weight-only quantization (like INT8, AWQ, GPTQ) has completely democratized AI. It allows massive, frontier LLMs to run on consumer hardware (like a single RTX 4090 or a Macbook) instead of massive server clusters. The decoding phase of LLMs is famously memory-bandwidth bound, meaning the GPU spends more time waiting for weights to arrive from VRAM than it does calculating. By sending 1 byte instead of 2 bytes per weight, we instantly double the generation speed!

**Historical Anecdotes:** 
The boom of local AI was sparked by projects like `llama.cpp` and `bitsandbytes`. Tim Dettmers (creator of bitsandbytes) pushed the limits of `int8` quantization for transformers, proving that with group-wise scaling factors, you can retain almost perfect quality while drastically reducing hardware limits. 

> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_baseline`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
**Typical CUDA Techniques:** 
- **On-The-Fly Dequantization:** Notice that we do *not* dequantize the weights and store them in memory. We load them as `int8`, cast them to `float` directly inside a thread's register, apply the scale factor, and immediately do the multiplication. This is crucial because if we dequantized them in memory, we would still have to transfer `float`s across the memory bus, destroying the bandwidth savings!
- **Data Types:** *Note: In real systems, the activations ($X$) and outputs ($Y$) are `half` (FP16). We use `float` in this exercise for consistency.*

## Objective
Implement a dense linear layer ($Y = XW^T$) where the weights are stored in `int8` format and dynamically dequantized.
You will learn about:
- Mixed precision arithmetic in CUDA.
- Dealing with scale factors and group sizes in quantized tensors.
- Overcoming memory bandwidth bottlenecks in LLMs.

## References
- *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* by Dettmers et al.
- *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* by Lin et al.

## Files Description
- **main.cc**: The test bench. Generates random float inputs, int8 weights, and scaling factors, then checks your dequantized dot-product against a sequential CPU baseline. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement the W8A16 dot product. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized reference implementation.

## What You Should Do
1. Open `kernel.cu`.
2. Notice that the grid is set up so each block computes exactly 1 element of the output matrix $Y[m, n]$.
3. Loop over the $K$ dimension. For each step $k$:
4. Calculate `group_id = k / group_size`.
5. Fetch the `scale` factor.
6. Fetch the `int8` weight.
7. Dequantize it: cast the weight to `float` and multiply by the `scale`.
8. Multiply it with the input $X$ and accumulate in a local register.
9. Perform a block-wide reduction to sum up all threads' results.
10. Have thread 0 write the final sum to $Y[m, n]$.

## Typical Commands
- **Help Menus**:
  ```bash
  make && ./bin/run_test.sh -h
  ```
- **Test a Specific Size**:
  ```bash
  make && ./bin/run_test.sh --size 4
  ```
- **Profile Memory Metrics**:
  ```bash
  make && ./bin/run_profiler.sh --ncu_argset=memory --size 16
  ```
