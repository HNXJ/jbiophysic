"""JSON manifest IO helpers."""

from __future__ import annotations

import json
from pathlib import Path


def write_json_manifest(path: str | Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json_manifest(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())
