# Exercise 31: PagedAttention (KV Cache Management)

## Problem Description
**The Problem:** In Large Language Model (LLM) generation, the model must "remember" the Key and Value (KV) states of all previously generated tokens. This memory is called the KV cache. Historically, this memory was allocated as large contiguous chunks for each sequence. However, since the exact length of a generated response isn't known ahead of time, systems had to over-allocate memory, leading to massive fragmentation and wasted VRAM.

**The Solution:** Inspired by operating system virtual memory, researchers developed **PagedAttention**. Instead of storing KV caches in contiguous memory, they are divided into small, fixed-size blocks (e.g., 16 tokens per block). A `block_table` maps a sequence's logical blocks to fragmented physical blocks.

**Sample Input/Output:** 
- Input: A sequence has a logical length of 20 tokens and block size of 16. It maps to physical blocks `[104, 21]`.
  To compute attention for the 18th token (logical block 1, offset 2), the kernel looks up physical block 21 and reads the key/value vector at offset 2.
- Operation: Compute scaled dot-product attention using indirect addressing via the block table.
- Output: A single output context vector for the sequence, identical to what would be produced if memory were contiguous.

**Practical Importance:** 
PagedAttention forms the core of the **vLLM** serving framework, widely considered one of the most important breakthroughs in LLM deployment in 2023. By virtually eliminating memory fragmentation, PagedAttention allows a single GPU to serve 2x to 4x more concurrent users, drastically reducing the cost of hosting LLMs like Llama-3 or Mistral.

**Historical Anecdotes:** 
Operating systems solved the memory fragmentation problem in the 1960s using page tables mapping virtual addresses to physical RAM pages. Fast forward 60 years to 2023, and researchers at UC Berkeley realized that LLM KV caches were suffering from the exact same problem! PagedAttention is a beautiful example of adapting classic computer science concepts to modern hardware bottlenecks.

> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_baseline`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
**Typical CUDA Techniques:** 
- **Indirect Memory Addressing:** Rather than computing `K[token_idx]`, you must compute `logical_block = token_idx / block_size`, look up `physical_block = block_table[logical_block]`, and then compute the final memory address using `physical_block * block_size + (token_idx % block_size)`. This teaches you how to decouple logical array indexing from physical memory layout.
- **Shared Memory Accumulation:** Since multiple threads need the Query vector $Q$, we load it into Shared Memory first so the entire block can broadcast it quickly.
- **Data Types:** *Note: LLMs exclusively use `half` (FP16) or `bfloat16` to double memory bandwidth. We use `float` in this exercise for simplicity.*

## Objective
Implement the memory-fetching logic of PagedAttention for the decoding phase (1 query token per sequence).
You will learn about:
- Managing fragmented memory structures on the GPU.
- Calculating dynamic block-table lookups.
- Implementing memory-bound self-attention correctly.

## References
- *Efficient Memory Management for Large Language Model Serving with PagedAttention* by Kwon et al. (The vLLM Paper)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)

## Files Description
- **main.cc**: The test bench. Generates fragmented physical memory pools and logical block tables, then validates your PagedAttention against a CPU baseline. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement the indirect lookup and attention math. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized reference implementation.

## What You Should Do
1. Open `kernel.cu`.
2. Notice the `shared_mem` allocation. Load the sequence's Query vector into it.
3. In the first loop, traverse the sequence tokens from `0` to `ctx_len`. 
4. Implement the `logical_block` and `physical_block` lookup to fetch the correct Key vectors and compute the attention scores (logits).
5. Apply the Softmax reduction.
6. In the final loop, use the block table again to fetch the correct Value vectors and compute the final weighted sum.

## Typical Commands
- **Help Menus**:
  ```bash
  make && ./bin/run_test.sh -h
  ```
- **Test a Specific Size**:
  ```bash
  make && ./bin/run_test.sh --size 128
  ```
- **Profile Memory Metrics**:
  ```bash
  make && ./bin/run_profiler.sh --ncu_argset=memory --size 1024
  ```
