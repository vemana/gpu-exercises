# Exercise 14: N-Body Simulation

## Problem Description
**The Problem:** The N-Body problem simulates the evolution of a dynamical system of particles, usually under the influence of physical forces like gravity or electrostatics. In a naive approach, computing the force on every body requires interacting it with every other body, leading to an $O(N^2)$ algorithm.

**Sample Input/Output:**
- $N$ Bodies with Mass `m`, Position `p`, Velocity `v`
- For each body `i`:
  - Calculate total force $F_i = \sum_{j \neq i} G \frac{m_i m_j}{|p_j - p_i|^2} \hat{u}$
  - Update velocity $v_i' = v_i + \frac{F_i}{m_i} \Delta t$

**Practical Importance:** N-Body simulations are the foundation of astrophysics (simulating galaxies and dark matter halos) and molecular dynamics (simulating protein folding and drug interactions). Because the problem is overwhelmingly compute-bound ($O(N^2)$ math for $O(N)$ data), it is the quintessential GPU problem. Many optimization techniques, like software pipelining and data tiling in shared memory, are best taught using the N-Body problem.

**Historical Anecdotes:** The study of the N-Body problem dates back to Isaac Newton trying to understand the orbital perturbations of the Sun, Earth, and Moon. In 2007, Lars Nyland, Mark Harris, and Jan Prins published "Fast N-Body Simulation with CUDA," which became the defining tutorial for writing high-performance GPU code. It demonstrated how to perfectly saturate the GPU's floating-point units using shared memory tiling.

**References:**
- *Fast N-Body Simulation with CUDA* in GPU Gems 3 (Chapter 31)
- *Programming Massively Parallel Processors* (Chapter: N-Body Methods)


> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_baseline`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
**Typical CUDA Techniques:**
- **Shared Memory Tiling:** The cornerstone of N-body. Because every body interacts with every other body, you load a tile of $T$ bodies into shared memory. All threads in the block compute gravitational forces against this tile. Then the block loads the next tile. This turns $O(N^2)$ global memory reads into $O(N^2 / T)$ reads.
- **Loop Unrolling:** Use `#pragma unroll` on the inner force calculation loop. This forces the compiler to duplicate the loop body, reducing branch instructions and massively increasing Instruction Level Parallelism (ILP).

**Modern CUDA Primitives (Ampere & Beyond):**
- **Async Copy & Double Buffering:** On Ampere, use `cuda::memcpy_async` to fetch the *next* tile of bodies into shared memory while the CUDA cores are computing forces on the *current* tile. 
- **Hopper TMA:** On Hopper, the Tensor Memory Accelerator completely offloads the fetching of the body tiles from the CUDA cores. The hardware autonomously streams the entire array of bodies through shared memory, allowing the compute cores to achieve near 100% theoretical FLOPS utilization.

## Objective
Implement an $O(N^2)$ N-body simulation kernel.
You will learn about:
- Managing significant data reuse by caching data in shared memory (tiling).
- Using loop unrolling to maximize compute throughput.
- Maximizing instruction level parallelism (ILP) to hide memory latency.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your N-Body kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation using a simple global memory nested loop. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `nbody_kernel` to compute the new velocities for `num_bodies` after a single time step `dt` using the supplied positions, masses, and initial velocities.
3. Use shared memory to create a tiled implementation! Since every body interacts with every other body, you can load a tile of bodies into shared memory, compute interactions against that tile for your thread block, and then load the next tile.
4. Implement `launch_nbody` to configure the grid and block dimensions, and launch your kernel.
5. Make sure `launch_nbody` returns a `std::vector<LaunchConfig>` containing the configurations for all kernels launched so the test framework can automatically display your kernel's hardware utilization.
6. Compile using `make` and run `./bin/run_test.sh` to evaluate your correctness and performance, and `./bin/run_profiler.sh` to identify bottlenecks.

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
