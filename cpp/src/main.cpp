#include <iostream>

#include "predictor/feature_extractor.hpp"
#include "predictor/inference_engine.hpp"
#include "scheduler/priority_queue.hpp"

int main() {
  scheduler::features::FeatureExtractor extractor;
  auto features = extractor.extract("Explain transformer attention in detail.");
  std::cout << "features[0]=" << features[0] << "\n";

  scheduler::predictor::InferenceEngine engine("shared/models/output_length.onnx");
  double f_ms = 0, p_ms = 0;
  const double pred = engine.predict("Write a Python sort function.", &f_ms, &p_ms);
  std::cout << "predicted_tokens=" << pred << " feature_ms=" << f_ms
            << " predict_ms=" << p_ms << "\n";

  scheduler::runtime::SJFScheduler scheduler;
  scheduler::runtime::ScheduledRequest req;
  req.request_id = "demo-1";
  req.prompt = "Hi";
  req.max_tokens = 20;
  req.predicted_tokens = pred;
  scheduler.submit(std::move(req));

  if (auto acquired = scheduler.acquire(100.0)) {
    std::cout << "acquired=" << acquired->request_id << "\n";
    scheduler.complete(acquired->request_id, true);
  }
  return 0;
}
