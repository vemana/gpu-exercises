# Exercise 15: 2D Convolution

## Problem Description
**The Problem:** 2D Convolution applies a small filter (or kernel) to an image by sliding it across all pixels. For each pixel, it computes the dot product of the filter weights and the neighboring pixel values. It requires handling 2D thread blocks, 2D grid coordinates, and boundary conditions (halo regions).

**Sample Input/Output:**
- Input Matrix (3x3): `[[1, 1, 1], [1, 1, 1], [1, 1, 1]]`
- Filter (3x3): `[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]` (Edge detection)
- Output Matrix (3x3): `[[2, 1, 2], [1, 0, 1], [2, 1, 2]]` (Assuming 0-padding for out-of-bounds pixels)

**Practical Importance:** 2D Convolution is the beating heart of Computer Vision and Convolutional Neural Networks (CNNs). Every time you apply a filter on Instagram, or a self-driving car detects a pedestrian, millions of 2D convolutions are being executed. Because neighboring pixels overlap in their filter windows, convolution heavily benefits from loading tiles of the image into shared memory or using specialized hardware instructions (like Tensor Cores).

**Historical Anecdotes:** Convolution has been used in signal processing for over a century. In the late 1980s, Yann LeCun pioneered the use of convolutions in neural networks for digit recognition. However, it wasn't until 2012, when Alex Krizhevsky implemented highly optimized 2D convolution kernels on GPUs (AlexNet), that deep learning exploded, fundamentally changing the trajectory of AI research forever.

**References:**
- *Programming Massively Parallel Processors* (Chapter: Convolution)
- *ImageNet Classification with Deep Convolutional Neural Networks* by Krizhevsky, Sutskever, and Hinton (2012)

## Newbie Guidance
**Typical CUDA Techniques:**
- **Constant Memory:** The convolution filter (e.g., a 3x3 or 5x5 matrix) is small, read-only, and accessed simultaneously by all threads. Store it in `__constant__` memory, which has a specialized hardware broadcast cache that makes it incredibly fast for this specific access pattern.
- **Shared Memory Halos:** Threads in a 16x16 block need a 18x18 tile of the image (to account for a 3x3 filter's borders). Load this padded tile into shared memory first to avoid redundant global memory reads for overlapping pixels.

**Modern CUDA Primitives (Ampere & Beyond):**
- **Tensor Cores (im2col):** Deep learning frameworks don't actually write sliding-window convolution kernels. They use an algorithm called `im2col` to reshape the image and the filters into two giant 2D matrices, and then use highly optimized GEMM (Matrix Multiplication) on Tensor Cores!
- **Hopper TMA (Tensor Memory Accelerator):** If you *do* write a sliding-window kernel, Hopper's TMA is game-changing. It natively understands 2D data boundaries and padding. You can tell TMA, "Fetch a 16x16 tile with a 1-pixel halo, and pad with zeros if it hits the image edge." The hardware does it asynchronously, completely eliminating all the complex `if (x > 0 && x < width)` boundary logic from your kernel.

## Objective
Implement a 2D convolution kernel (e.g. for image filtering).
You will learn about:
- 2D thread blocks and grids.
- Constant memory (for the filter kernel).
- Shared memory (for the input tile + halo region) to dramatically reduce global memory reads.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your convolution kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation using a simple global memory nested loop. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `conv2d_kernel` to apply the 3x3 `filter` to the input image `a` and store the result in `c`. Make sure to handle boundaries correctly (pixels outside the image boundaries contribute 0).
3. Try to use constant memory for the `filter` and shared memory for a tile of `a` to optimize performance!
4. Implement `launch_conv2d` to configure the grid and block dimensions, and launch your kernel.
5. Make sure `launch_conv2d` returns a `std::vector<LaunchConfig>` containing the configurations for all kernels launched so the test framework can automatically display your kernel's hardware utilization.
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
