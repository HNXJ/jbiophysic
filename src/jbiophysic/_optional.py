# src/jbiophysic/_optional.py
from __future__ import annotations

import importlib
from types import ModuleType


def optional_import(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def require_optional(name: str, extra: str) -> ModuleType:
    module = optional_import(name)
    if module is None:
        raise ImportError(
            f"Optional dependency '{name}' is required for this feature. "
            f"Install with: pip install -e '.[{extra}]'"
        )
    return module
