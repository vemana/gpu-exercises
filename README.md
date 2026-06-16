# GPU Basics: Foundational CUDA Algorithms

Welcome! This repository contains a structured series of exercises designed to teach you foundational algorithms used in GPU kernel programming, focusing on **CUDA** and **NVIDIA GPUs**. 

Instead of just learning syntax, these exercises walk you through the core parallel patterns (like Map, Reduce, Scan, Sort, Stencil, and more) that power the world's most advanced GPU applications—from real-time graphics to massive deep learning models.

## Prerequisites

To build and run these exercises, you will need the following installed on your system:
- **NVIDIA GPU** with compatible drivers installed.
- **CUDA Toolkit** (which provides `nvcc`, the NVIDIA CUDA Compiler).
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

## Acknowledgements

All the code in this repository has been generated and refined using **Antigravity** and **Gemini 3.1 Pro**.
