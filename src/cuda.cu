#include "common.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>

static void handle_error(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file,
            line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err) (handle_error(err, __FILE__, __LINE__))

inline __device__ uint64_t min_d(const uint64_t a, const uint64_t b) {
  return a < b ? a : b;
}

__global__ void scan_block(int *A, const uint32_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    int tmp;
    const int max_s = min_d(blockDim.x, n - blockDim.x * blockIdx.x);
    for (int s = 1; s < max_s; s <<= 1) {
      tmp = A[i - s];
      __syncthreads();
      if (threadIdx.x >= s) {
        A[i] += tmp;
      }
      __syncthreads();
    }
  }
}

__global__ void gather_sec_sum(const int *A, const uint32_t n, int *sec_sums,
                               const uint64_t sec_size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if ((i + 1) * sec_size <= n) {
    sec_sums[i] = A[(i + 1) * sec_size - 1];
  }
}

inline __device__ __host__ uint32_t div_up(const uint32_t n,
                                           const uint32_t block_size) {
  return (n + block_size - 1) / block_size;
}

__global__ void add_sec_sum(int *A, const uint32_t n, const int *sec_sums,
                            const uint64_t sec_size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int target_block = blockIdx.x + sec_size / blockDim.x;
  if (i + sec_size < n && target_block / sec_size == blockIdx.x / sec_size) {
    A[i + sec_size] += sec_sums[i / sec_size];
  }
}

int cuda_wrapper(const int *arr, int *result, const int n) {
  const int block_size = 1024;
  int *arr_d, *sec_sums_d;
  uint64_t n_bytes = n * sizeof(int);
  uint32_t n_block = div_up(n, block_size);
  HANDLE_ERROR(cudaMalloc((void **)&arr_d, n_bytes));
  HANDLE_ERROR(cudaMalloc((void **)&sec_sums_d, n_block * sizeof(int)));
  if (sec_sums_d == NULL || arr_d == NULL) {
    return -1;
  }
  HANDLE_ERROR(cudaMemcpy(arr_d, arr, n_bytes, cudaMemcpyHostToDevice));
  scan_block<<<n_block, block_size>>>(arr_d, n);
  uint64_t sec_size = block_size;
  unsigned int n_sec = div_up(n_block, block_size);
  while (sec_size < n) {
    gather_sec_sum<<<n_sec, block_size>>>(arr_d, n, sec_sums_d, sec_size);
    scan_block<<<n_sec, block_size>>>(sec_sums_d, div_up(n, sec_size));
    n_block = div_up(n - sec_size, block_size);
    add_sec_sum<<<n_block, block_size>>>(arr_d, n, sec_sums_d, sec_size);
    sec_size *= block_size;
    n_sec = div_up(n_sec, block_size);
  }
  HANDLE_ERROR(cudaMemcpy(result, arr_d, n_bytes, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(arr_d));
  HANDLE_ERROR(cudaFree(sec_sums_d));
  return 0;
}