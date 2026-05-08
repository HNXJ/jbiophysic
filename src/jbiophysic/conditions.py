from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class Condition:
    """A condition describes a simulation scenario (e.g., stimulus, duration)."""
    name: str
    duration_ms: float
    dt_ms: float
    stimulus: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
