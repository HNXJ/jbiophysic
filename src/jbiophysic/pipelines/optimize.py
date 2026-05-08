"""Optimization pipeline placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationResult:
    best_loss: float
    status: str
    truth_mode: str = "truth_safe_unverified"
