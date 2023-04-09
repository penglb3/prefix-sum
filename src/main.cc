#include "common.h"
#include <algorithm>
#include <cstdio>
#include <omp.h>

int error_position(const int *arr, const int *sum, int n) {
  int s = 0;
  for (int i = 0; i < n; i++) {
    s += arr[i];
    if (s != sum[i]) {
      printf("#%d expect %d, got %d\n", i, s, sum[i]);
      return i;
    }
  }
  return -1;
};

int main(int argc, char *argv[]) {
  const int size = 1100000000;
  int *a = new int[size];
  int *c = new int[size];
  for (int i = 0; i < size; i++) {
    a[i] = 1;
  }
  int ok = cuda_wrapper(a, c, size);
  printf("Run OK? %d\n", ok == 0);
  printf("First error pos = %d\n", error_position(a, c, size));
  return 0;
}