"""Small calculator library used as the demo app for the CI/CD risk-gate case study."""
from __future__ import annotations


def add(a: float, b: float) -> float:
    return a + b


def subtract(a: float, b: float) -> float:
    return a - b


def multiply(a: float, b: float) -> float:
    return a * b


def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("cannot divide by zero")
    return a / b


def percentage(part: float, whole: float) -> float:
    if whole == 0:
        raise ValueError("whole cannot be zero")
    return (part / whole) * 100
