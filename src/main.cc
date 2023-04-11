#include "common.h"
#include <algorithm>
#include <cstdio>
#include <omp.h>
#include <vector>
using std::vector;
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
  int size = 1024 * 1024 * 64;
  vector<int> a(size, 1);
  vector<int> c(size, 0);
  int ok = cuda_wrapper(a.data(), c.data(), size);
  printf("Run OK? %d\n", ok == 0);
  printf("First error pos = %d\n", error_position(a.data(), c.data(), size));
  return 0;
}