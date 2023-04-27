# Parallel Prefix Sums
Implements parallel prefix sums on OpenMP and CUDA.

## Build
`cmake -B build && cmake --build build`

Or if you don't have an GPU and only want to test CPU part: `g++ src/main.cc src/cpu.cc -DCPU_ONLY -fopenmp -mavx2`

## Usage
`./build/prefix-sum -d <device> -a <algorithm> -n <# of elements> -t <# of threads> -r <# to repeat>`
- `-d`: Available options are `cpu`,`cuda`. Default = `cpu`.
- `-a`: For CPU, available options are `seq`, `scan`, `efficient`, `block`. For GPU, available options are `scan`, `efficient`. Default = `efficient`.
- `-n`: Default = 4096.
- `-t`: Default = `omp_max_threads()`
- `-r`: Default = 1. But we would do a warm-up (iteration #0) no matter how many times we repeat, so the actual iterations to be ran would be `-r` + 1.

## Test Environment
- CPU: AMD Ryzen 5 3600 (6c12t)
- GPU: Nvidia Geforce RTX 2060 Super
- OS: Windows 10 21H2
- Compiler:
  - C++: MSVC 19.28 (CUDA relies on MSVC on Windows)
  - CUDA: NVCC 11.7