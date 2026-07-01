#include "scheduler/priority_queue.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace scheduler::runtime {

namespace {

int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct HeapItem {
  double priority;
  int64_t enqueue_ns;
  ScheduledRequest req;

  bool operator>(const HeapItem& o) const {
    if (priority != o.priority) return priority > o.priority;
    return enqueue_ns > o.enqueue_ns;
  }
};

}  // namespace

struct PriorityQueue::Impl {
  mutable std::mutex mu;
  std::condition_variable cv;
  std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> heap;
  std::unordered_set<std::string> cancelled;
  std::unordered_map<std::string, HeapItem> pending;
  std::uint64_t total_enqueued{0};
  std::uint64_t total_dequeued{0};
  std::vector<double> wait_ms;
};

PriorityQueue::PriorityQueue() : impl_(std::make_unique<Impl>()) {}
PriorityQueue::~PriorityQueue() = default;

void PriorityQueue::enqueue(ScheduledRequest req) {
  std::lock_guard lock(impl_->mu);
  req.enqueue_ns = now_ns();
  req.priority = req.predicted_tokens > 0 ? req.predicted_tokens
                                          : static_cast<double>(req.max_tokens);
  HeapItem item{req.priority, req.enqueue_ns, std::move(req)};
  impl_->pending[item.req.request_id] = item;
  impl_->heap.push(item);
  impl_->total_enqueued++;
  impl_->cv.notify_one();
}

std::optional<ScheduledRequest> PriorityQueue::dequeue(double timeout_ms) {
  std::unique_lock lock(impl_->mu);
  if (!impl_->cv.wait_for(lock, std::chrono::duration<double, std::milli>(timeout_ms),
                          [this] { return !impl_->heap.empty(); })) {
    return std::nullopt;
  }
  while (!impl_->heap.empty()) {
    auto item = impl_->heap.top();
    impl_->heap.pop();
    if (impl_->cancelled.count(item.req.request_id)) {
      impl_->cancelled.erase(item.req.request_id);
      impl_->pending.erase(item.req.request_id);
      continue;
    }
    const double wait =
        static_cast<double>(now_ns() - item.enqueue_ns) / 1'000'000.0;
    impl_->wait_ms.push_back(wait);
    if (impl_->wait_ms.size() > 10000) impl_->wait_ms.erase(impl_->wait_ms.begin());
    impl_->pending.erase(item.req.request_id);
    impl_->total_dequeued++;
    return item.req;
  }
  return std::nullopt;
}

bool PriorityQueue::cancel(const std::string& request_id) {
  std::lock_guard lock(impl_->mu);
  if (!impl_->pending.count(request_id)) return false;
  impl_->cancelled.insert(request_id);
  return true;
}

void PriorityQueue::apply_aging(double factor, double max_boost) {
  std::lock_guard lock(impl_->mu);
  const int64_t t = now_ns();
  for (auto& [id, item] : impl_->pending) {
    const double wait_s = static_cast<double>(t - item.enqueue_ns) / 1e9;
    const double boost = std::min(max_boost, factor * wait_s * 1000.0);
    item.priority = std::max(1.0, item.req.priority - boost);
    item.req.priority = item.priority;
  }
}

QueueStats PriorityQueue::stats() const {
  std::lock_guard lock(impl_->mu);
  QueueStats s;
  s.depth = impl_->heap.size();
  s.total_enqueued = impl_->total_enqueued;
  s.total_dequeued = impl_->total_dequeued;
  if (!impl_->wait_ms.empty()) {
    double sum = 0;
    for (double w : impl_->wait_ms) sum += w;
    s.avg_wait_ms = sum / impl_->wait_ms.size();
  }
  return s;
}

std::size_t PriorityQueue::depth() const { return stats().depth; }

struct SJFScheduler::Impl {
  PriorityQueue queue;
  std::uint64_t completed{0};
  std::uint64_t errors{0};
};

SJFScheduler::SJFScheduler() : impl_(std::make_unique<Impl>()) {}
SJFScheduler::~SJFScheduler() = default;

void SJFScheduler::submit(ScheduledRequest req) { impl_->queue.enqueue(std::move(req)); }

std::optional<ScheduledRequest> SJFScheduler::acquire(double timeout_ms) {
  impl_->queue.apply_aging(0.001, 500.0);
  return impl_->queue.dequeue(timeout_ms);
}

void SJFScheduler::complete(const std::string& request_id, bool success) {
  (void)request_id;
  if (success) impl_->completed++;
  else impl_->errors++;
}

bool SJFScheduler::cancel(const std::string& request_id) {
  return impl_->queue.cancel(request_id);
}

QueueStats SJFScheduler::stats() const { return impl_->queue.stats(); }

}  // namespace scheduler::runtime
