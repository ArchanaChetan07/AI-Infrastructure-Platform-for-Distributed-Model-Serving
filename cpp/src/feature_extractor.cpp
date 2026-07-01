#include "predictor/feature_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <unordered_set>

namespace scheduler::features {

namespace {

int count_char(std::string_view s, char c) {
  return static_cast<int>(std::count(s.begin(), s.end(), c));
}

bool contains(std::string_view s, std::string_view needle) {
  return s.find(needle) != std::string_view::npos;
}

float shannon_entropy(std::string_view text) {
  if (text.empty()) return 0.0f;
  int counts[256]{};
  for (unsigned char c : text) counts[c]++;
  const float n = static_cast<float>(text.size());
  float h = 0.0f;
  for (int c : counts) {
    if (c == 0) continue;
    const float p = static_cast<float>(c) / n;
    h -= p * std::log2(p);
  }
  return h;
}

}  // namespace

std::array<float, kNumFeatures> FeatureExtractor::extract(std::string_view prompt) const {
  std::array<float, kNumFeatures> f{};
  const std::string text(prompt);
  const int chars = static_cast<int>(text.size());

  int words = 0;
  int vocab = 0;
  std::unordered_set<std::string> seen;
  bool in_word = false;
  for (size_t i = 0; i < text.size(); ++i) {
    if (std::isalnum(static_cast<unsigned char>(text[i]))) {
      if (!in_word) {
        words++;
        in_word = true;
      }
    } else if (in_word) {
      in_word = false;
    }
  }
  for (const auto& w : std::string_view(text)) {
    (void)w;
  }

  int tokens = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    if (!std::isspace(static_cast<unsigned char>(text[i]))) {
      if (i == 0 || std::isspace(static_cast<unsigned char>(text[i - 1]))) tokens++;
    }
  }
  if (tokens == 0) tokens = std::max(1, words);

  int upper = 0, lower = 0, space = 0, digit = 0;
  for (char c : text) {
    if (std::isupper(static_cast<unsigned char>(c))) upper++;
    if (std::islower(static_cast<unsigned char>(c))) lower++;
    if (std::isspace(static_cast<unsigned char>(c))) space++;
    if (std::isdigit(static_cast<unsigned char>(c))) digit++;
  }
  const float n = static_cast<float>(std::max(chars, 1));

  f[0] = static_cast<float>(tokens);
  f[1] = static_cast<float>(chars);
  f[2] = static_cast<float>(words);
  f[3] = chars / static_cast<float>(tokens);
  f[4] = words > 0 ? chars / static_cast<float>(words) : 0.0f;
  f[5] = static_cast<float>(vocab);
  f[6] = words > 0 ? static_cast<float>(seen.size()) / words : 0.0f;
  f[7] = shannon_entropy(prompt);
  f[8] = f[7];
  f[9] = static_cast<float>(count_char(prompt, '?') + count_char(prompt, '!') +
                            count_char(prompt, '.') + count_char(prompt, ','));
  f[10] = static_cast<float>(count_char(prompt, '?'));
  f[11] = static_cast<float>(count_char(prompt, '!'));
  f[12] = static_cast<float>(count_char(prompt, '.'));
  f[13] = static_cast<float>(count_char(prompt, ','));
  f[14] = static_cast<float>(digit);
  f[15] = upper / n;
  f[16] = lower / n;
  f[17] = space / n;
  f[18] = contains(prompt, "#") || contains(prompt, "**") ? 1.0f : 0.0f;
  f[19] = contains(prompt, "```") ? 1.0f : 0.0f;
  f[20] = contains(prompt, "def ") || contains(prompt, "import ") ? 1.0f : 0.0f;
  f[21] = (contains(prompt, "SELECT") || contains(prompt, "select") ||
           contains(prompt, "FROM") || contains(prompt, "from"))
              ? 1.0f
              : 0.0f;
  f[22] = contains(prompt, "{") && contains(prompt, ":") ? 1.0f : 0.0f;
  f[23] = contains(prompt, "<?xml") || contains(prompt, "<html") ? 1.0f : 0.0f;
  f[24] = contains(prompt, "---") ? 1.0f : 0.0f;
  f[25] = contains(prompt, "http") ? 1.0f : 0.0f;
  f[26] = static_cast<float>(count_char(prompt, '\n'));
  f[27] = 1.0f;
  f[28] = contains(prompt, "system:") ? 1.0f : 0.0f;
  f[29] = contains(prompt, "assistant:") ? 1.0f : 0.0f;
  f[30] = contains(prompt, "user:") ? 1.0f : 0.0f;
  f[31] = 0.0f;
  f[32] = 0.0f;
  f[33] = std::min(1.0f, f[10] * 0.1f + (words > 0 ? 0.1f : 0.0f));
  f[34] = f[19] > 0 || f[20] > 0 ? 0.8f : 0.1f;
  f[35] = 0.3f;
  f[36] = static_cast<float>(std::max(1, count_char(prompt, '.') + count_char(prompt, '?')));
  f[37] = words > 0 ? chars / f[36] : 0.0f;
  f[38] = 0.0f;
  f[39] = 0.0f;
  return f;
}

}  // namespace scheduler::features
