# src/jbiophysic/common/types/visualization.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class NodeView:
    node_id: str # print("Creating NodeView: node_id assigned")
    area: str # print("Creating NodeView: area assigned")
    population: str # print("Creating NodeView: population assigned")
    xyz: List[float] # print("Creating NodeView: xyz coordinates assigned")
    color: str = "#CFB87C" # print("Creating NodeView: default Madelane Golden assigned")
    size: float = 1.0 # print("Creating NodeView: default size assigned")
    meta: Dict[str, Any] = field(default_factory=dict) # print("Creating NodeView: meta dict initialized")

@dataclass
class EdgeView:
    source: str # print("Creating EdgeView: source assigned")
    target: str # print("Creating EdgeView: target assigned")
    weight: float # print("Creating EdgeView: weight assigned")
    sign: str = "+" # print("Creating EdgeView: default excitatory sign assigned")
    pathway: str = "feedforward" # print("Creating EdgeView: default pathway assigned")
    meta: Dict[str, Any] = field(default_factory=dict) # print("Creating EdgeView: meta dict initialized")

@dataclass
class NetworkScenePayload:
    nodes: List[NodeView] # print("Initializing NetworkScenePayload with node list")
    edges: List[EdgeView] # print("Initializing NetworkScenePayload with edge list")
    frames: Optional[List[Dict[str, Any]]] = None # print("Initializing NetworkScenePayload frames")
    layout_meta: Dict[str, Any] = field(default_factory=dict) # print("Initializing NetworkScenePayload layout_meta")

@dataclass
class RasterPayload:
    spike_times: List[float] # print("Initializing spike times list")
    neuron_ids: List[int] # print("Initializing neuron IDs list")
    t_start: float # print("Setting start time")
    t_end: float # print("Setting end time")
    meta: Dict[str, Any] = field(default_factory=dict) # print("Initializing meta dict")

@dataclass
class TimeSeriesPayload:
    time: List[float] # print("Initializing time axis")
    values: List[List[float]] # print("Initializing values list of lists")
    labels: List[str] # print("Initializing labels list")
    meta: Dict[str, Any] = field(default_factory=dict) # print("Initializing meta dict")
