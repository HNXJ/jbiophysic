# src/jbiophysic/common/types/visualization.py


from dataclasses import dataclass, field
from typing import Any

from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NodeView:
    node_id: str
    area: str
    population: str
    xyz: list[float]
    color: str = "#CFB87C"
    size: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeView:
    source: str
    target: str
    weight: float
    sign: str = "+"
    pathway: str = "feedforward"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkScenePayload:
    nodes: list[NodeView]
    edges: list[EdgeView]
    frames: list[dict[str, Any]] | None = None
    layout_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RasterPayload:
    spike_times: list[float]
    neuron_ids: list[int]
    t_start: float
    t_end: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeSeriesPayload:
    time: list[float]
    values: list[list[float]]
    labels: list[str]
    meta: dict[str, Any] = field(default_factory=dict)
