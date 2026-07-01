#include "predictor/inference_engine.hpp"

#include <chrono>
#include <fstream>
#include <numeric>

namespace scheduler::predictor {

struct InferenceEngine::Impl {
  features::FeatureExtractor extractor;
  std::string model_path;
  bool onnx_loaded{false};
  std::array<float, features::kNumFeatures> mean{};
  std::array<float, features::kNumFeatures> std{};
};

InferenceEngine::InferenceEngine(std::string model_path)
    : impl_(std::make_unique<Impl>()) {
  impl_->model_path = std::move(model_path);
  impl_->onnx_loaded = !impl_->model_path.empty();
}

InferenceEngine::~InferenceEngine() = default;

bool InferenceEngine::loaded() const { return impl_->onnx_loaded; }

double InferenceEngine::predict(std::string_view prompt, double* feature_ms,
                                double* predict_ms) const {
  const auto t0 = std::chrono::steady_clock::now();
  auto features = impl_->extractor.extract(prompt);
  const auto t1 = std::chrono::steady_clock::now();
  if (feature_ms) {
    *feature_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  const auto t2 = std::chrono::steady_clock::now();
  double score = 0.0;
  for (std::size_t i = 0; i < features::kNumFeatures; ++i) {
    score += static_cast<double>(features[i]) * (0.5 + 0.02 * static_cast<double>(i));
  }
  score = std::max(5.0, score * 0.8 + static_cast<double>(prompt.size()) * 0.05);
  const auto t3 = std::chrono::steady_clock::now();
  if (predict_ms) {
    *predict_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
  }
  return score;
}

}  // namespace scheduler::predictor
