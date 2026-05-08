"""Plain-text report helpers."""

from __future__ import annotations


def evidence_status(passed: bool, label: str) -> str:
    return f"{label}: {'PASS' if passed else 'FAIL'}"
