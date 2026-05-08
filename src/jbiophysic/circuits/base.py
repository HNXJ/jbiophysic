from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass(frozen=True)
class Circuit:
    """A circuit encapsulates parameters, state, and a step function."""
    params: Any
    state: Any
    connectivity: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    step_fn: Callable | None = None

@dataclass(frozen=True)
class SimulationResult:
    """The result of a simulation run."""
    outputs: Any
    time_ms: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
