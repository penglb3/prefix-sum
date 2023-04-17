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
#ifndef CPU_ONLY
const wrapper_t WRAPPER[] = {cpu_wrapper, cuda_wrapper};
#else
const wrapper_t WRAPPER[] = {cpu_wrapper};
#endif

using std::string;

void get_param(InputParser &parser, const char *opt, int &arg,
               vector<string> viable_values = {}) {
  string args = parser.get_option(opt);
  if (viable_values.empty()) {
    if (!args.empty()) {
      arg = atoi(args.c_str());
    }
  } else {
    for (int i = 0; i < viable_values.size(); i++) {
      if (args == viable_values[i]) {
        arg = i;
      }
    }
  }
}

int main(int argc, const char *argv[]) {
  InputParser parser(argc, argv);
  int size = 1024 * 4, device = CPU, algo = SCAN, repeat = 1;
  get_param(parser, "-n", size);
  get_param(parser, "-r", repeat);
  get_param(parser, "-d", device, DEVICE_NAMES);
  get_param(parser, "-a", algo, ALGO_NAMES);
  num_omp_threads = omp_get_max_threads();
  get_param(parser, "-t", num_omp_threads);

  printf("Running %s algorithm on %s, size = %d\n", ALGO_NAMES[algo].c_str(),
         DEVICE_NAMES[device].c_str(), size);
  vector<int> a(size, 1);
  vector<int> c(size, 0);
  // Warm up! Especially important for cuda.
  printf("Test start.\n");
  float time_ms;
  time_point_t time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i <= repeat; i++) { // One extra time, because #0 is warm up.
    WRAPPER[device](a.data(), c.data(), size, algo);
    time_ms = chrono_event_tick(time);
    printf("[#%d] End to end latency: %.3f ms\n", i, time_ms);
  }
  printf("First error pos = %d\n", error_position(a.data(), c.data(), size));
  return 0;
}