"""Configuration management for jbiophysic models and experiments.

Provides frozen dataclass-based configuration with YAML loading, validation,
and deterministic serialization.
"""

from .spectrolaminar import SpectrolaminarMotifConfig

__all__ = ["SpectrolaminarMotifConfig"]
