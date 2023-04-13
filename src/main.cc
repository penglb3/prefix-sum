#include "common.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
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

class InputParser {
public:
  InputParser(int argc, const char *argv[]) {
    for (int i = 1; i < argc; ++i) {
      this->tokens_.emplace_back(argv[i]);
    }
  }
  /// @author iain
  const std::string &get_option(const std::string &option) const {
    std::vector<std::string>::const_iterator itr;
    itr = std::find(this->tokens_.begin(), this->tokens_.end(), option);
    if (itr != this->tokens_.end() && ++itr != this->tokens_.end()) {
      return *itr;
    }
    static const std::string empty_string;
    return empty_string;
  }
  /// @author iain
  bool option_exists(const std::string &option) const {
    return std::find(this->tokens_.begin(), this->tokens_.end(), option) !=
           this->tokens_.end();
  }

private:
  std::vector<std::string> tokens_;
};

using wrapper_t = std::function<int(const int *, int *, int, uint8_t)>;
const wrapper_t WRAPPER[] = {cpu_wrapper, cuda_wrapper};

using std::string;
int main(int argc, const char *argv[]) {
  InputParser parser(argc, argv);
  int size = 1024 * 4, device = CPU, algo = SCAN;
  string arg = parser.get_option("-s");
  if (!arg.empty()) {
    size = atoi(arg.c_str());
  }
  arg = parser.get_option("-d");
  for (int i = 0; i < DEVICE_NAMES.size(); i++) {
    if (arg == DEVICE_NAMES[i]) {
      device = i;
    }
  }
  arg = parser.get_option("-a");
  for (int i = 0; i < ALGO_NAMES.size(); i++) {
    if (arg == ALGO_NAMES[i]) {
      algo = i;
    }
  }
  arg = parser.get_option("-t");
  if (!arg.empty()) {
    num_omp_threads = atoi(arg.c_str());
    printf("#Thread = %d\n", num_omp_threads);
  }
  printf("Running %s algorithm on %s, size = %d\n", ALGO_NAMES[algo].c_str(),
         DEVICE_NAMES[device].c_str(), size);
  vector<int> a(size, 1);
  vector<int> c(size, 0);
  printf("Computation start\n");
  time_point_t time = std::chrono::high_resolution_clock::now();
  int ok = WRAPPER[device](a.data(), c.data(), size, algo);
  float time_ms = chrono_event_tick(time);
  printf("End to end latency: %.3f ms\n", time_ms);
  printf("First error pos = %d\n", error_position(a.data(), c.data(), size));
  return 0;
}