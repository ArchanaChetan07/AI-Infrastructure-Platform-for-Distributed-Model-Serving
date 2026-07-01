"""Scheduler aging policies to prevent starvation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class AgingPolicy(str, Enum):
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    EXPONENTIAL = "exponential"


@dataclass
class AgingConfig:
    """Aging configuration."""

    policy: AgingPolicy = AgingPolicy.LINEAR
    factor: float = 0.001
    max_boost: float = 500.0
    base_priority: float = 0.0


class AgingFunction(Protocol):
    def boost(self, wait_time_sec: float, base_priority: float) -> float: ...


class LinearAging:
    def __init__(self, config: AgingConfig) -> None:
        self.config = config

    def boost(self, wait_time_sec: float, base_priority: float) -> float:
        boost = self.config.factor * wait_time_sec
        return base_priority - min(boost, self.config.max_boost)


class LogarithmicAging:
    def __init__(self, config: AgingConfig) -> None:
        self.config = config

    def boost(self, wait_time_sec: float, base_priority: float) -> float:
        boost = self.config.factor * math.log1p(wait_time_sec) * 100.0
        return base_priority - min(boost, self.config.max_boost)


class ExponentialAging:
    def __init__(self, config: AgingConfig) -> None:
        self.config = config

    def boost(self, wait_time_sec: float, base_priority: float) -> float:
        boost = self.config.factor * (math.exp(min(wait_time_sec, 10.0) / 5.0) - 1.0) * 50.0
        return base_priority - min(boost, self.config.max_boost)


def create_aging(config: AgingConfig) -> AgingFunction:
    mapping = {
        AgingPolicy.LINEAR: LinearAging,
        AgingPolicy.LOGARITHMIC: LogarithmicAging,
        AgingPolicy.EXPONENTIAL: ExponentialAging,
    }
    return mapping[config.policy](config)
