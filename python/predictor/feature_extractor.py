"""Fast prompt feature extraction for output-length prediction."""

from __future__ import annotations

import math
import re
import string
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from predictor.features import FEATURE_NAMES, NUM_FEATURES, FeatureVector

# Lightweight tokenization — avoids heavy deps at import time.
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_SENTENCE_RE = re.compile(r"[.!?]+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```")
_PYTHON_RE = re.compile(r"\b(def |class |import |from |print\(|if __name__|lambda )", re.IGNORECASE)
_SQL_RE = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|JOIN)\b", re.IGNORECASE)
_JSON_RE = re.compile(r'[\[{].*["\']\s*:\s*["\']', re.DOTALL)
_XML_RE = re.compile(r"<\?xml|<[a-zA-Z][^>]*>")
_YAML_RE = re.compile(r"^[\s-]*[a-zA-Z_][\w-]*\s*:", re.MULTILINE)
_MARKDOWN_RE = re.compile(r"(^#{1,6}\s|^\*\s|^\-\s|\*\*|__|\[.+\]\(.+\))", re.MULTILINE)

_STOPWORDS = frozenset(
    "a an the and or but if in on at to for of is are was were be been being "
    "it this that with as by from not".split()
)

_LANG_HINTS: dict[str, re.Pattern[str]] = {
    "en": re.compile(r"\b(the|and|is|are|you|your)\b", re.IGNORECASE),
    "es": re.compile(r"\b(el|la|los|las|que|de|en)\b", re.IGNORECASE),
    "fr": re.compile(r"\b(le|la|les|de|et|est|une|un)\b", re.IGNORECASE),
    "de": re.compile(r"\b(der|die|das|und|ist|ein|eine)\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class ExtractionResult:
    """Feature extraction output with timing metadata."""

    features: FeatureVector
    latency_ms: float
    prompt: str


class FeatureExtractor:
    """Extract numerical features from prompts in <1 ms (typical)."""

    def __init__(self, enable_ner: bool = False, enable_embedding: bool = False) -> None:
        self._enable_ner = enable_ner
        self._enable_embedding = enable_embedding
        self._ner: Any = None
        self._embedder: Any = None

    def extract(self, prompt: str) -> ExtractionResult:
        """Extract features from a single prompt string."""
        t0 = time.perf_counter()
        text = prompt or ""
        features = self._extract_features(text)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ExtractionResult(features=features, latency_ms=latency_ms, prompt=text)

    def extract_batch(self, prompts: Sequence[str]) -> list[ExtractionResult]:
        return [self.extract(p) for p in prompts]

    def _extract_features(self, text: str) -> FeatureVector:
        chars = len(text)
        words = _WORD_RE.findall(text)
        word_count = len(words)
        tokens = text.split()
        token_count = len(tokens) if tokens else max(1, word_count)

        vocab = set(w.lower() for w in words)
        vocab_size = len(vocab)
        unique_ratio = vocab_size / word_count if word_count else 0.0

        total_chars = max(chars, 1)
        uppercase = sum(1 for c in text if c.isupper())
        lowercase = sum(1 for c in text if c.islower())
        whitespace = sum(1 for c in text if c.isspace())
        digits = sum(1 for c in text if c.isdigit())
        punct = sum(1 for c in text if c in string.punctuation)

        q_marks = text.count("?")
        excls = text.count("!")
        periods = text.count(".")
        commas = text.count(",")

        entropy = _shannon_entropy(text)
        info_density = entropy / math.log2(max(vocab_size, 2)) if vocab_size > 1 else 0.0

        sentences = _SENTENCE_RE.split(text)
        sentence_count = max(1, len([s for s in sentences if s.strip()]))
        avg_sentence_len = word_count / sentence_count

        stopwords = sum(1 for w in words if w.lower() in _STOPWORDS)
        stopword_ratio = stopwords / word_count if word_count else 0.0

        lang_code = _detect_language(text)
        unicode_chars = sum(1 for c in text if ord(c) > 127)
        unicode_ratio = unicode_chars / total_chars

        reasoning_score = _reasoning_score(text, q_marks, word_count)
        coding_prob = _coding_probability(text)

        turns = text.count("user:") + text.count("assistant:") + text.count("system:")
        if turns == 0 and "\n" in text:
            turns = text.count("\n\n") + 1

        ner_count = self._named_entity_count(text) if self._enable_ner else 0.0
        embed_norm = self._embedding_norm(text) if self._enable_embedding else 0.0

        values = (
            float(token_count),
            float(chars),
            float(word_count),
            chars / token_count,
            (chars / word_count) if word_count else 0.0,
            float(vocab_size),
            unique_ratio,
            entropy,
            info_density,
            float(punct),
            float(q_marks),
            float(excls),
            float(periods),
            float(commas),
            float(digits),
            uppercase / total_chars,
            lowercase / total_chars,
            whitespace / total_chars,
            1.0 if _MARKDOWN_RE.search(text) else 0.0,
            1.0 if _CODE_FENCE_RE.search(text) else 0.0,
            1.0 if _PYTHON_RE.search(text) else 0.0,
            1.0 if _SQL_RE.search(text) else 0.0,
            1.0 if _JSON_RE.search(text) else 0.0,
            1.0 if _XML_RE.search(text) else 0.0,
            1.0 if _YAML_RE.search(text) else 0.0,
            float(len(_URL_RE.findall(text))),
            float(text.count("\n")),
            float(max(turns, 1)),
            1.0 if re.search(r"\bsystem\s*:", text, re.I) else 0.0,
            1.0 if re.search(r"\bassistant\s*:", text, re.I) else 0.0,
            1.0 if re.search(r"\buser\s*:", text, re.I) else 0.0,
            _lang_to_float(lang_code),
            unicode_ratio,
            reasoning_score,
            coding_prob,
            stopword_ratio,
            float(sentence_count),
            avg_sentence_len,
            ner_count,
            embed_norm,
        )
        assert len(values) == NUM_FEATURES
        return FeatureVector(values)

    def _named_entity_count(self, text: str) -> float:
        if self._ner is None:
            try:
                import spacy  # type: ignore[import-untyped]

                self._ner = spacy.load("en_core_web_sm")
            except Exception:
                return float(len(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)))
        doc = self._ner(text[:512])
        return float(len(doc.ents))

    def _embedding_norm(self, text: str) -> float:
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                return math.sqrt(len(text)) / 100.0
        vec = self._embedder.encode(text[:256], normalize_embeddings=True)
        return float((vec**2).sum() ** 0.5)

    @staticmethod
    def feature_names() -> tuple[str, ...]:
        return FEATURE_NAMES


def extract_from_messages(messages: Sequence[Mapping[str, Any]]) -> ExtractionResult:
    """Extract features from OpenAI-style chat messages."""
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c.get("text", c)) for c in content if isinstance(c, dict))
        parts.append(f"{role}: {content}")
    return FeatureExtractor().extract("\n".join(parts))


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _detect_language(text: str) -> str:
    scores = {lang: len(pat.findall(text)) for lang, pat in _LANG_HINTS.items()}
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best if scores[best] > 0 else "en"


def _lang_to_float(code: str) -> float:
    mapping = {"en": 0.0, "es": 0.25, "fr": 0.5, "de": 0.75}
    return mapping.get(code, 1.0)


def _reasoning_score(text: str, q_marks: int, word_count: int) -> float:
    reasoning_kw = ("why how explain reason analyze compare evaluate prove step think").split()
    hits = sum(1 for w in _WORD_RE.findall(text.lower()) if w in reasoning_kw)
    base = hits / max(word_count, 1) * 10.0
    return min(1.0, base + q_marks * 0.1)


def _coding_probability(text: str) -> float:
    signals = sum(
        [
            bool(_CODE_FENCE_RE.search(text)),
            bool(_PYTHON_RE.search(text)),
            bool(_SQL_RE.search(text)),
            "function" in text.lower(),
            "class " in text.lower(),
            "import " in text.lower(),
        ]
    )
    return min(1.0, signals / 3.0)
