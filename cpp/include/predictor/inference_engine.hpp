#pragma once

#include <array>
#include <memory>
#include <string>
#include <string_view>

#include "predictor/feature_extractor.hpp"

namespace scheduler::predictor {

/// ONNX Runtime inference engine with heuristic fallback.
class InferenceEngine {
 public:
  explicit InferenceEngine(std::string model_path);
  ~InferenceEngine();

  [[nodiscard]] double predict(std::string_view prompt, double* feature_ms,
                               double* predict_ms) const;

  [[nodiscard]] bool loaded() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace scheduler::predictor
