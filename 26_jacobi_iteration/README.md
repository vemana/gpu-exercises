# Exercise 26: Jacobi Iteration (2D Laplace Equation)

## Problem Description
**The Problem:** The Jacobi method is an iterative algorithm for solving strictly diagonally dominant systems of linear equations. In the context of a 2D grid (like simulating steady-state heat conduction in a metal plate), you iteratively update each interior grid point with the average of its 4 neighbors (up, down, left, right). 

**Sample Input/Output:** 
- Input: A 3x3 2D grid:
  ```text
  [100, 100, 100]
  [  0,   0,   0]
  [  0,   0,   0]
  ```
- Operation: Execute one iteration of the 5-point Jacobi stencil on the interior point (at row 1, col 1). It averages its 4 neighbors: `(100 + 0 + 0 + 0) / 4`.
- Output: The updated 3x3 grid:
  ```text
  [100, 100, 100]
  [  0,  25,   0]
  [  0,   0,   0]
  ```

**Practical Importance:** 
Iterative PDE solvers are the workhorses of Computational Fluid Dynamics (CFD), weather forecasting, and physics simulations. Stencil computations on structured grids map extremely well to GPUs. Real-world solvers scale this across thousands of GPUs, moving petabytes of data per second. Understanding how to manage memory for these iterative stencils is foundational for working at National Labs or on scientific supercomputers.

**Historical Anecdotes:** 
Carl Gustav Jacob Jacobi developed this iterative method in the 1840s. He probably didn't foresee his algorithm running simultaneously on 80,000 parallel threads on a piece of silicon the size of a postage stamp. It was one of the earliest algorithms adapted to the Connection Machine and later GPUs because of its perfectly uniform communication pattern.

> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_baseline`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
**Typical CUDA Techniques:** 
- **Double Buffering:** In a Jacobi iteration, you cannot overwrite the grid while you are reading from it. You must maintain two buffers (`u_old` and `u_new`) and swap their pointers after every iteration. This is called "Ping-Ponging".
- **Halo Exchanges (for multi-GPU):** Though not tested here, in multi-GPU setups, you would divide the grid into chunks and exchange the border pixels ("halos") via NVLink between iterations.
- **Data Types:** *Note: HPC applications typically use `double` precision to prevent accumulation errors in iterative solvers. We use `float` in this exercise to maintain consistency with the rest of the repository.*

**Modern CUDA Primitives (Ampere & Beyond):**
- **CUDA Graphs:** Launching the kernel 100 times from the CPU incurs significant launch overhead. Modern HPC solvers encapsulate the entire iterative loop in a CUDA Graph to bypass the CPU and keep the GPU perfectly utilized.
- **Grid Synchronization:** Advanced CUDA allows threads to synchronize globally across the entire grid (using Cooperative Groups), enabling the solver to run in a single kernel launch instead of launching a new kernel per iteration.

## Objective
Implement the Jacobi iteration solver for the 2D Laplace equation.
You will learn about:
- Managing iterative Ping-Pong buffers (double buffering) on the device.
- Performing 2D 5-point stencil operations.
- Avoiding host-device synchronization bottlenecks in an iterative loop.

## References
- *Numerical Recipes in C* (Chapter 19: Partial Differential Equations)
- [CUDA Toolkit Documentation: Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)

## Files Description
- **main.cc**: The test bench. It initializes the grid, sets the boundary conditions, and checks correctness against a CPU reference. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement the Jacobi kernel and the launch loop. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized reference implementation.

## What You Should Do
1. Open `kernel.cu`.
2. Implement `jacobi_kernel` to apply the 5-point stencil to all interior points, leaving the boundary points untouched.
3. Implement `launch_jacobi`. This function must launch the kernel `num_iters` times in a `for` loop, swapping the `current` and `next` pointers after every launch.
4. If `num_iters` is odd, ensure the final result is correctly copied into the `d_u` buffer so the test framework can verify it.
5. Return the `LaunchConfig` of your kernel.

## Typical Commands
- **Help Menus**:
  ```bash
  make && ./bin/run_test.sh -h
  ```
- **Test a Specific Size**:
  ```bash
  make && ./bin/run_test.sh --size 256x256
  ```
- **Profile Memory Metrics**:
  ```bash
  make && ./bin/run_profiler.sh --ncu_argset=memory --size 256x256
  ```
