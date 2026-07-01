#include <cassert>
#include <iostream>

#include "predictor/feature_extractor.hpp"
#include "scheduler/priority_queue.hpp"

int main() {
  scheduler::features::FeatureExtractor fx;
  auto f = fx.extract("SELECT * FROM users WHERE id = 1");
  assert(f[21] == 1.0f);

  scheduler::runtime::SJFScheduler sjf;
  scheduler::runtime::ScheduledRequest long_req{"long", "essay", 500, 500.0};
  scheduler::runtime::ScheduledRequest short_req{"short", "hi", 10, 10.0};
  sjf.submit(std::move(long_req));
  sjf.submit(std::move(short_req));

  auto first = sjf.acquire(100.0);
  assert(first.has_value());
  assert(first->request_id == "short");

  std::cout << "scheduler_tests: OK\n";
  return 0;
}
