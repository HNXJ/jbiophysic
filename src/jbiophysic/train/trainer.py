from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class TrainingResult:
    """The result of a training/optimization run."""
    best_params: Any
    history: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    invalid_count: int = 0
    nan_inf_found: bool = False

def trainer(net, conditions, objectives, optimizer, key=None, config=None):
    """Skeletal trainer contract."""
    raise NotImplementedError(
        "Trainer contract is defined; implementation comes in the optimization phase."
    )
