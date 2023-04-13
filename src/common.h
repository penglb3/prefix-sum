#include <chrono>
#include <cstdint>
#include <vector>

enum { SCAN, EFFICIENT, SEQ, BLOCK_SEQ };
const std::vector<std::string> ALGO_NAMES = {"scan", "efficient", "seq", "block"};
enum { CPU, CUDA };
const std::vector<std::string> DEVICE_NAMES = {"cpu", "cuda"};

int cuda_wrapper(const int *arr, int *result, int n_elems, uint8_t type = SCAN);
int cpu_wrapper(const int *arr, int *result, int n_elems,
                uint8_t type = EFFICIENT);

extern int num_omp_threads;

#ifdef _MSC_VER
using time_point_t = std::chrono::steady_clock::time_point;
#else
using time_point_t = std::chrono::system_clock::time_point;
#endif

// Can't use std::function, or nvcc would complain.
using sum_func_t = void (*)(int *, const int);

inline float chrono_event_tick(time_point_t &start) {
  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  start = end;
  return time.count() * 1e-6;
}

inline void sequential_prefix_sum(int *arr, const int size) {
  for (uint32_t i = 1; i < size; i++) {
    arr[i] += arr[i - 1];
  }
}