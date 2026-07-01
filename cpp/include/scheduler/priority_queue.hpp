#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace scheduler::runtime {

struct ScheduledRequest {
  std::string request_id;
  std::string prompt;
  int32_t max_tokens{128};
  double predicted_tokens{0.0};
  double priority{0.0};
  int64_t enqueue_ns{0};
};

struct QueueStats {
  std::size_t depth{0};
  std::uint64_t total_enqueued{0};
  std::uint64_t total_dequeued{0};
  double avg_wait_ms{0.0};
};

/// Thread-safe SJF priority queue with aging support.
class PriorityQueue {
 public:
  PriorityQueue();
  ~PriorityQueue();

  PriorityQueue(const PriorityQueue&) = delete;
  PriorityQueue& operator=(const PriorityQueue&) = delete;

  void enqueue(ScheduledRequest req);
  std::optional<ScheduledRequest> dequeue(double timeout_ms);
  bool cancel(const std::string& request_id);
  void apply_aging(double factor, double max_boost);
  [[nodiscard]] QueueStats stats() const;
  [[nodiscard]] std::size_t depth() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class SJFScheduler {
 public:
  SJFScheduler();
  ~SJFScheduler();

  SJFScheduler(const SJFScheduler&) = delete;
  SJFScheduler& operator=(const SJFScheduler&) = delete;

  void submit(ScheduledRequest req);
  std::optional<ScheduledRequest> acquire(double timeout_ms);
  void complete(const std::string& request_id, bool success);
  bool cancel(const std::string& request_id);
  [[nodiscard]] QueueStats stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace scheduler::runtime
