#include "common.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <immintrin.h> // avx
#include <omp.h>

// TODO(PLB): Parallize this
void scan(int *A, const int n) {
  for (int s = 1; s < n; s <<= 1) {
    int tmp;
    __m256i a_i, a_i_s;
    // OpenMP won't provide any sync within a loop, so this has to be sequential
    for (int i = n - 8; i >= s; i -= 8) {
      a_i = _mm256_load_si256((__m256i *)(A + i));
      a_i_s = _mm256_load_si256((__m256i *)(A + i - s));
      a_i = _mm256_add_epi32(a_i, a_i_s);
      _mm256_store_si256((__m256i *)(A + i), a_i);
    }
    for (int i = s + (n - s) % 8 - 1; i >= s; i--) {
      A[i] += A[i - s];
    }
  }
}

int num_omp_threads = 0;

void sum_efficient(int *A, const int n) {
  int s;
  //
  for (s = 1; s < n; s <<= 1) {
    // Memory access is not continuous, so no point in SIMD
#pragma omp parallel for num_threads(num_omp_threads)
    for (int i = 2 * s - 1; i < n; i += s * 2) {
      A[i] += A[i - s];
    }
  }
  for (s >>= 1; s >= 1; s >>= 1) {
#pragma omp parallel for num_threads(num_omp_threads)
    for (int i = 2 * s - 1; i < n - s; i += s * 2) {
      A[i + s] += A[i];
    }
  }
}

void block(int *A, const int n) {
  int block_len = (n + num_omp_threads - 1) / num_omp_threads;
#pragma omp parallel for num_threads(num_omp_threads)
  for (int t = 0; t < num_omp_threads; t++) {
    sequential_prefix_sum(A + block_len * t,
                          std::min(block_len, n - block_len * t));
  }
  for (int t = 1; t < num_omp_threads; t++) {
#pragma omp parallel for num_threads(num_omp_threads)
    for (int i = 0; i < std::min(block_len, n - block_len * t); i++) {
      A[block_len * t + i] += A[block_len * t - 1];
    }
  }
}

const sum_func_t ALGOS[] = {scan, sum_efficient, sequential_prefix_sum, block};

int cpu_wrapper(const int *arr, int *result, int n_elems, uint8_t type) {
  if (num_omp_threads == 0) {
    num_omp_threads = omp_get_max_threads();
  }
  memcpy(result, arr, n_elems * sizeof(int));
  time_point_t time = std::chrono::high_resolution_clock::now();
  ALGOS[type](result, n_elems);
  printf("%.3f ms\n", chrono_event_tick(time));
  return 0;
}