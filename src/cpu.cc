#include "common.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <immintrin.h> // avx
#include <omp.h>

void scan(int *A, const int n) {
  for (int s = 1; s < n; s <<= 1) {
#ifndef SCAN_MULTITHREADING
    __m256i a_i, a_i_s;
    int i;
    // Ensure 32-alignment for AVX load/stores
    for (i = n - 1; i >= (n & ~7); i--) {
      A[i] += A[i - s];
    }
    // OpenMP won't provide any sync within a loop, so this has to be sequential
    for (i -= 7; i >= s; i -= 8) {
      a_i = _mm256_load_si256((__m256i *)(A + i));
      a_i_s = _mm256_load_si256((__m256i *)(A + i - s));
      a_i = _mm256_add_epi32(a_i, a_i_s);
      _mm256_store_si256((__m256i *)(A + i), a_i);
    }
    for (i += 7; i >= s; i--) {
      A[i] += A[i - s];
    }
#else // Correct multithreaded version, but performance SUCKS
    int T = num_omp_threads;
    std::vector<__m256i> a_i(T), a_i_s(T);
    for (int i = n - 8 * T; i >= s; i -= 8 * T) {
#pragma omp parallel for schedule(static) num_threads(T)
      for (int j = 0; j < T; j++) {
        a_i[j] = _mm256_loadu_si256((__m256i *)(A + i + 8 * j));
        a_i_s[j] = _mm256_loadu_si256((__m256i *)(A + i + 8 * j - s));
      }
#pragma omp parallel for schedule(static) num_threads(T)
      for (int j = 0; j < T; j++) {
        a_i[j] = _mm256_add_epi32(a_i[j], a_i_s[j]);
        _mm256_storeu_si256((__m256i *)(A + i + 8 * j), a_i[j]);
      }
    }
    for (int i = s + (n - s) % (8 * T) - 1; i >= s; i--) {
      A[i] += A[i - s];
    }
#endif
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
#ifndef BLOCK_SIMD
#pragma omp parallel for num_threads(num_omp_threads)
    for (int i = 0; i < std::min(block_len, n - block_len * t); i++) {
      A[block_len * t + i] += A[block_len * t - 1];
    }
#else // Below is SIMD version of the above loop, but it performs worse.
    __m256i a_i, a_i_s = _mm256_set1_epi32(A[block_len * t - 1]);
    int blk_len = std::min(block_len, n - block_len * t);
#pragma omp parallel for num_threads(num_omp_threads)
    for (int i = 0; i < (blk_len & ~7); i += 8) {
      a_i = _mm256_load_si256((__m256i *)(A + i + block_len * t));
      a_i = _mm256_add_epi32(a_i, a_i_s);
      _mm256_store_si256((__m256i *)(A + i + block_len * t), a_i);
    }
    for (int i = blk_len & ~7; i < blk_len; i++) {
      A[i + t * block_len] += A[block_len * t - 1];
    }
#endif
  }
}

const sum_func_t ALGOS[] = {scan, sum_efficient, sequential_prefix_sum, block};

int cpu_wrapper(const int *arr, int *result, int n_elems, uint8_t type) {
  memcpy(result, arr, n_elems * sizeof(int));
  time_point_t time = std::chrono::high_resolution_clock::now();
  ALGOS[type](result, n_elems);
  printf("[C] %.3f ms\n", chrono_event_tick(time));
  return 0;
}