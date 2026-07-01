#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>

namespace scheduler::features {

constexpr std::size_t kNumFeatures = 40;

/// Fast prompt feature vector for output-length prediction.
class FeatureExtractor {
 public:
  FeatureExtractor() = default;

  [[nodiscard]] std::array<float, kNumFeatures> extract(std::string_view prompt) const;

  [[nodiscard]] static constexpr std::size_t num_features() { return kNumFeatures; }
};

}  // namespace scheduler::features
