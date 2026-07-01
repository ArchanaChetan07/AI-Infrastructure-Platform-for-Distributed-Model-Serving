"""Feature schema and ordering for output-length prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, List, Tuple

# Canonical feature order — must match training and inference.
FEATURE_NAMES: Final[Tuple[str, ...]] = (
    "prompt_token_count",
    "prompt_char_count",
    "word_count",
    "avg_token_length",
    "avg_word_length",
    "vocabulary_size",
    "unique_word_ratio",
    "prompt_entropy",
    "information_density",
    "punctuation_count",
    "question_mark_count",
    "exclamation_count",
    "period_count",
    "comma_count",
    "digit_count",
    "uppercase_ratio",
    "lowercase_ratio",
    "whitespace_ratio",
    "has_markdown",
    "has_code_fence",
    "has_python",
    "has_sql",
    "has_json",
    "has_xml",
    "has_yaml",
    "url_count",
    "newline_count",
    "conversation_turn_count",
    "has_system_prompt",
    "has_assistant_prompt",
    "has_user_prompt",
    "language_code",
    "unicode_ratio",
    "reasoning_score",
    "coding_probability",
    "stopword_ratio",
    "sentence_count",
    "avg_sentence_length",
    "named_entity_count",
    "embedding_norm",
)

NUM_FEATURES: Final[int] = len(FEATURE_NAMES)


@dataclass(frozen=True)
class FeatureVector:
    """Typed container for extracted features."""

    values: Tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.values) != NUM_FEATURES:
            raise ValueError(f"Expected {NUM_FEATURES} features, got {len(self.values)}")

    def as_list(self) -> List[float]:
        return list(self.values)

    def as_dict(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.values, strict=True))

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> FeatureVector:
        return cls(tuple(float(data[name]) for name in FEATURE_NAMES))

    @classmethod
    def zeros(cls) -> FeatureVector:
        return cls(tuple(0.0 for _ in FEATURE_NAMES))
