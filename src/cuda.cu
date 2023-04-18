#include "common.h"
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <vector>

/* ***********************************
 * Section 1: CUDA Utilities
 ************************************* */

static void handle_error(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file,
            line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err) (handle_error(err, __FILE__, __LINE__))

inline float cuda_event_tick(const cudaEvent_t &start, cudaEvent_t &stop) {
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
  cudaEventRecord(start);
  return milliseconds;
}

enum { DEV_CUDA, DEV_CPU };

class EventTimer {
public:
  cudaEvent_t start_cuda, stop_cuda;
  time_point_t time_chrono;
  int event_device;
  explicit EventTimer(int device = DEV_CPU) : event_device(device) {
    HANDLE_ERROR(cudaEventCreate(&start_cuda));
    HANDLE_ERROR(cudaEventCreate(&stop_cuda));
  }
  float tick(int next_event_device = DEV_CPU) {
    float duration = event_device == DEV_CPU
                         ? chrono_event_tick(time_chrono)
                         : cuda_event_tick(start_cuda, stop_cuda);
    if (event_device == DEV_CPU && next_event_device == DEV_CUDA) {
      cudaEventRecord(start_cuda);
    }
    if (event_device == DEV_CUDA && next_event_device == DEV_CPU) {
      time_chrono = std::chrono::high_resolution_clock::now();
    }
    event_device = next_event_device;
    return duration;
  }
};

/* ***********************************
 * Section 2: Kernel functions
 ************************************* */

inline __device__ int min_d(const int a, const int b) {
  return a < b ? a : b;
}

__global__ void scan_block(int *A, const int n) {
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

__global__ void sum_block_efficient(int *A, const int n) {
  int i = (blockDim.x * blockIdx.x + threadIdx.x) * 2 + 1;
  if (i < n) {
    int s;
    for (s = 1; s < min_d(blockDim.x, n - blockDim.x * blockIdx.x); s <<= 1) {
      if ((threadIdx.x + 1) % s == 0) {
        A[i] += A[i - s];
      }
      __syncthreads();
    }
    if (threadIdx.x == blockDim.x - 1) {
      A[i] += A[i - s];
    }
    for (s >>= 1; s >= 1; s >>= 1) {
      if ((threadIdx.x + 1) % s == 0 && threadIdx.x + s < blockDim.x) {
        A[i + s] += A[i];
      }
      __syncthreads();
    }
  }
}

__global__ void gather_offsets(const int *A, const uint32_t n, int *offsets,
                               const uint64_t sec_size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if ((i + 1) * sec_size <= n) {
    offsets[i] = A[(i + 1) * sec_size - 1];
  }
}

__global__ void add_offsets(int *A, const uint32_t n, const int *offsets,
                            const uint64_t sec_size, const uint32_t p) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int target_block = blockIdx.x + sec_size / blockDim.x;
  if (i + sec_size < n &&
      target_block / (sec_size * p) == blockIdx.x / (sec_size * p)) {
    A[i + sec_size] += offsets[i / sec_size];
  }
}

/* ***********************************
 * Section 3: CUDA Wrappers
 ************************************* */

inline uint64_t div_up(const uint64_t n, const uint64_t block_size) {
  return (n + block_size - 1) / block_size;
}

const sum_func_t ALGOS[] = {scan_block, sum_block_efficient};
const int UNIT_BLOCK_P[] = {1, 2};

int cuda_wrapper(const int *arr, int *result, const int n, uint8_t type) {
  const int block_size = 1024;
  const int p = UNIT_BLOCK_P[type];
  auto sum_block = ALGOS[type];
  int unit_size = block_size * p;
  int *arr_d, *offsets_d;
  uint64_t n_bytes = n * sizeof(int);
  uint32_t n_block = div_up(n, unit_size);
  std::vector<float> times;
  EventTimer timer;
  timer.tick(); // #0: Alloc memory on memory
  HANDLE_ERROR(cudaMalloc((void **)&arr_d, n_bytes));
  HANDLE_ERROR(cudaMalloc((void **)&offsets_d, n_block * sizeof(int)));
  times.push_back(timer.tick()); // #1: Copy array to device
  if (offsets_d == NULL || arr_d == NULL) {
    return -1;
  }
  HANDLE_ERROR(cudaMemcpy(arr_d, arr, n_bytes, cudaMemcpyHostToDevice));
  times.push_back(timer.tick(DEV_CUDA)); // #2: First scan
  // Sum up within each unit
  sum_block<<<n_block, block_size>>>(arr_d, n);
  // A section consists of <unit_size> units
  uint64_t sec_size = unit_size;
  unsigned int n_sec = div_up(n, sec_size * block_size);
  times.push_back(timer.tick(DEV_CUDA)); // #3: section sums
  while (sec_size < n) { // Loop until the whole array become a single section
    // Gather offsets of each unit into offsets_d
    gather_offsets<<<n_sec, block_size>>>(arr_d, n, offsets_d, sec_size);
    // sum up offsets within a section
    sum_block<<<n_sec, block_size>>>(offsets_d, div_up(n, sec_size));
    // For each unit, add corresponding offset. Then every section is done.
    n_block = div_up(n - sec_size, block_size);
    add_offsets<<<n_block, block_size>>>(arr_d, n, offsets_d, sec_size, p);
    // An old section becomes a unit in a new section.
    sec_size *= unit_size;
    n_sec = div_up(n, sec_size * block_size);
  }
  times.push_back(timer.tick()); // #4: Copy results back
  HANDLE_ERROR(cudaMemcpy(result, arr_d, n_bytes, cudaMemcpyDeviceToHost));
  times.push_back(timer.tick());
  HANDLE_ERROR(cudaFree(arr_d));
  HANDLE_ERROR(cudaFree(offsets_d));
  for (int i = 0; i < times.size(); i++) {
    printf("#%d: %.3f(ms)\n", i, times[i]);
  }
  printf("[C] %.3f ms\n", times[2] + times[3]);
  return 0;
}