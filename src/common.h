#include <chrono>
#include <cstdint>

int cuda_wrapper(const int *arr, int *result, int n_elems);
int omp_wrapper(const int *arr, int *result, int n_elems);

using time_point_t = std::chrono::steady_clock::time_point;

inline float chrono_event_tick(time_point_t &start) {
  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  start = end;
  return time.count() * 1e-6;
}

inline int sequential_prefix_sum(int *arr, uint32_t size) {
  for (uint32_t i = 1; i < size; i++) {
    arr[i] += arr[i - 1];
  }
  return 0;
}