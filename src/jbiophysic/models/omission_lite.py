"""Minimal omission-task condition descriptors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OmissionCondition:
    name: str
    bottom_up_input: bool
    top_down_prediction: bool


BASELINE = OmissionCondition("baseline", False, False)
UNEXPECTED_SENSORY = OmissionCondition("unexpected_sensory", True, False)
PREDICTED_STANDARD = OmissionCondition("predicted_standard", True, True)
OMISSION = OmissionCondition("omission", False, True)
