"""Basic descriptive statistics helpers.

Added as a single large batch with light test coverage, deliberately, to act as the
"risky commit" in the Chapter 4 case study: a big diff relative to the amount of new
test coverage it brings.
"""
from __future__ import annotations


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    return sum(values) / len(values)


def variance(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("variance requires at least 2 values")
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - 1)


def stdev(values: list[float]) -> float:
    return variance(values) ** 0.5


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2
    return s[mid]


def mode(values: list[float]) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    counts: dict[float, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)


def range_(values: list[float]) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    return max(values) - min(values)


def normalize(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0 for _ in values]
    return [(x - lo) / (hi - lo) for x in values]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(mean(values[start:i + 1]))
    return out
