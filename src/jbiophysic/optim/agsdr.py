"""Adaptive GSDR scheduling helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AGSDRSchedule:
    alpha_min: float = 0.05
    alpha_max: float = 0.8
    alpha_up: float = 0.05
    alpha_down: float = 0.01


def adapt_alpha(alpha: float, *, plateau: bool, improving: bool, schedule: AGSDRSchedule = AGSDRSchedule()) -> float:
    """Update alpha based on plateau/improvement flags."""
    if plateau:
        alpha += schedule.alpha_up
    if improving:
        alpha -= schedule.alpha_down
    return min(max(alpha, schedule.alpha_min), schedule.alpha_max)
