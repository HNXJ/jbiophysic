# src/jbiophysic/common/types/visualization.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class NodeView:
    node_id: str
    area: str
    population: str
    xyz: List[float]
    color: str = "#CFB87C"
    size: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EdgeView:
    source: str
    target: str
    weight: float
    sign: str = "+"
    pathway: str = "feedforward"
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NetworkScenePayload:
    nodes: List[NodeView]
    edges: List[EdgeView]
    frames: Optional[List[Dict[str, Any]]] = None
    layout_meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RasterPayload:
    spike_times: List[float]
    neuron_ids: List[int]
    t_start: float
    t_end: float
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TimeSeriesPayload:
    time: List[float]
    values: List[List[float]]
    labels: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)
