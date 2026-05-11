"""Plotly 3-D anatomy visualization for jbiophysic networks.

This module visualizes circuit scaffolds and anatomical layouts only. It does
not execute neuronal dynamics, TFNE source projection, CSD/LFP field solves, or
biological mechanism validation.

The main entry point is :func:`visualize_network_3d`, which accepts population
objects, dictionaries, table-like records, or simple networkx-like graphs and
renders them as an interactive Plotly HTML-compatible 3-D figure.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import math
import numpy as np


_COLOR_BY_CELL_TYPE = {
    "E": "#C8A000",
    "PV": "#00E5F0",
    "SST": "#D000D7",
    "VIP": "#D8D8D8",
    "I": "#FF7F0E",
    "unknown": "#AAAAAA",
}

_SYMBOL_BY_CELL_TYPE = {
    "E": "circle",
    "PV": "diamond",
    "SST": "square",
    "VIP": "circle-open",
    "I": "x",
    "unknown": "circle",
}

_PATHWAY_COLORS = {
    "thalamic_input": "#00FF00",
    "thalamic input": "#00FF00",
    "feedforward": "#1E40FF",
    "feedback": "#00FFFF",
}


@dataclass(frozen=True)
class _ColumnMeta:
    area: str
    center_m: tuple[float, float]
    radius_m: float
    z0_m: float
    z1_m: float
    layer_boundaries_m: tuple[float, ...]


def _require_plotly():
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception as exc:  # pragma: no cover - message tested through caller behavior.
        raise ImportError(
            "Plotly is required for jbiophysic.viz.network3d; "
            "install with the repo visualization extra."
        ) from exc
    return go


def _as_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _repeat_default(value: Any, n: int, *, dtype: str | None = None) -> np.ndarray:
    arr = np.asarray([value] * n)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _unit_scale(coordinate_unit: str, display_unit: str) -> float:
    cu = coordinate_unit.lower().replace("µ", "u")
    du = display_unit.lower().replace("µ", "u")
    units = {
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "mm": 1.0e-3,
        "millimeter": 1.0e-3,
        "millimeters": 1.0e-3,
        "um": 1.0e-6,
        "micrometer": 1.0e-6,
        "micrometers": 1.0e-6,
        "nm": 1.0e-9,
    }
    if cu not in units:
        raise ValueError(f"Unsupported coordinate_unit: {coordinate_unit!r}")
    if du not in units:
        raise ValueError(f"Unsupported display_unit: {display_unit!r}")
    return units[cu] / units[du]


def _position_columns_from_mapping(
    network: Mapping[str, Any],
    *,
    positions_key: str,
) -> np.ndarray:
    if positions_key in network:
        return _positions_to_xyz_m(network[positions_key])
    for keys in (("x_m", "y_m", "z_m"), ("x", "y", "z")):
        if all(k in network for k in keys):
            return _positions_to_xyz_m(np.column_stack([network[k] for k in keys]))
    for keys in (("x_um", "y_um", "z_um"),):
        if all(k in network for k in keys):
            return _positions_to_xyz_m(np.column_stack([network[k] for k in keys]) * 1.0e-6)
    raise ValueError(
        f"Could not find coordinates. Provide {positions_key!r}, x_m/y_m/z_m, or x/y/z."
    )


def _positions_to_xyz_m(positions: Any) -> np.ndarray:
    arr = np.asarray(positions, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("positions must be an array with shape [N, D], where D is 1, 2, or 3")
    if arr.shape[1] == 1:
        zeros = np.zeros((arr.shape[0], 2), dtype=float)
        arr = np.column_stack([arr[:, 0], zeros])
    elif arr.shape[1] == 2:
        arr = np.column_stack([arr[:, 0], arr[:, 1], np.zeros(arr.shape[0], dtype=float)])
    elif arr.shape[1] == 3:
        arr = arr.astype(float, copy=False)
    else:
        raise ValueError("positions must have D=1, D=2, or D=3 columns")
    if not np.all(np.isfinite(arr)):
        raise ValueError("positions contain NaN or Inf")
    return arr.astype(float, copy=True)


def _coerce_sequence_column(records: Sequence[Mapping[str, Any]], key: str, default: Any) -> np.ndarray:
    return np.asarray([row.get(key, default) for row in records])


def _records_to_mapping(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("record sequence is empty")
    keys = set().union(*(row.keys() for row in records))
    return {key: _coerce_sequence_column(records, key, None) for key in keys}


def _extract_networkx_like(network: Any) -> tuple[dict[str, Any], list[tuple[Any, Any]] | None]:
    nodes_method = getattr(network, "nodes", None)
    if nodes_method is None:
        raise TypeError("not a networkx-like object")
    try:
        nodes = list(nodes_method(data=True))
    except TypeError:
        raise TypeError("not a networkx-like object") from None
    if not nodes:
        raise ValueError("networkx-like graph has no nodes")

    ids: list[Any] = []
    rows: list[dict[str, Any]] = []
    for node_id, data in nodes:
        if not isinstance(data, Mapping):
            data = {}
        row = dict(data)
        row.setdefault("neuron_id", node_id)
        ids.append(node_id)
        rows.append(row)

    mapping = _records_to_mapping(rows)
    edge_list = None
    edges_method = getattr(network, "edges", None)
    if edges_method is not None:
        try:
            edge_list = list(edges_method())
        except TypeError:
            edge_list = None
    return mapping, edge_list


def _extract_population_object(pop: Any) -> dict[str, Any]:
    if all(hasattr(pop, attr) for attr in ("x_m", "y_m", "z_m")):
        n = len(getattr(pop, "x_m"))
        out: dict[str, Any] = {
            "positions_m": np.column_stack(
                [
                    _as_array(getattr(pop, "x_m"), dtype=float),
                    _as_array(getattr(pop, "y_m"), dtype=float),
                    _as_array(getattr(pop, "z_m"), dtype=float),
                ]
            )
        }
        for attr in ("neuron_id", "cell_type", "layer", "area"):
            if hasattr(pop, attr):
                out[attr] = _as_array(getattr(pop, attr))
        out.setdefault("neuron_id", np.arange(n))
        out.setdefault("cell_type", _repeat_default("unknown", n, dtype="U16"))
        out.setdefault("layer", _repeat_default("unknown", n, dtype="U32"))
        out.setdefault("area", _repeat_default("cortex", n, dtype="U32"))
        return out

    if hasattr(pop, "__dict__"):
        data = {
            key: value
            for key, value in vars(pop).items()
            if not key.startswith("_") and not callable(value)
        }
        if data:
            return data

    raise TypeError("Unsupported network object. Provide a mapping, records, Population object, or graph.")


def _coerce_network_to_table(
    network: Any,
    *,
    positions_key: str,
    node_id_key: str,
    cell_type_key: str,
    layer_key: str,
    area_key: str,
) -> tuple[list[dict[str, Any]], list[tuple[Any, Any]] | None, list[dict[str, Any]]]:
    edge_list: list[tuple[Any, Any]] | None = None
    metadata_pathways: list[dict[str, Any]] = []

    if isinstance(network, Mapping):
        mapping = dict(network)
    elif isinstance(network, Sequence) and network and isinstance(network[0], Mapping):
        mapping = _records_to_mapping(network)  # type: ignore[arg-type]
    else:
        try:
            mapping, edge_list = _extract_networkx_like(network)
        except (TypeError, ValueError):
            mapping = _extract_population_object(network)

    if "__pathways__" in mapping and mapping["__pathways__"] is not None:
        metadata_pathways = list(mapping["__pathways__"])

    positions_m = _position_columns_from_mapping(mapping, positions_key=positions_key)
    n = positions_m.shape[0]

    def column(key: str, default: Any, dtype: str | None = None) -> np.ndarray:
        if key in mapping and mapping[key] is not None:
            arr = _as_array(mapping[key])
            if len(arr) != n:
                raise ValueError(f"{key!r} length {len(arr)} does not match positions length {n}")
            return arr.astype(dtype) if dtype is not None else arr
        return _repeat_default(default, n, dtype=dtype)

    neuron_id = column(node_id_key, np.arange(n), None)
    cell_type = column(cell_type_key, "unknown", "U32")
    layer = column(layer_key, "unknown", "U64")
    area = column(area_key, "cortex", "U64")

    rows: list[dict[str, Any]] = []
    reserved = {
        positions_key,
        "positions_m",
        "x_m",
        "y_m",
        "z_m",
        "x",
        "y",
        "z",
        "x_um",
        "y_um",
        "z_um",
        node_id_key,
        cell_type_key,
        layer_key,
        area_key,
        "__pathways__",
        "__column_meta__",
    }
    extra_columns: dict[str, np.ndarray] = {}
    for key, value in mapping.items():
        if key in reserved or key.startswith("__"):
            continue
        try:
            arr = _as_array(value)
        except Exception:
            continue
        if len(arr) == n:
            extra_columns[key] = arr

    for i in range(n):
        row = {
            "neuron_id": neuron_id[i].item() if hasattr(neuron_id[i], "item") else neuron_id[i],
            "cell_type": str(cell_type[i]),
            "layer": str(layer[i]),
            "area": str(area[i]),
            "x_m": float(positions_m[i, 0]),
            "y_m": float(positions_m[i, 1]),
            "z_m": float(positions_m[i, 2]),
            "jittered": False,
        }
        for key, arr in extra_columns.items():
            value = arr[i]
            row[key] = value.item() if hasattr(value, "item") else value
        rows.append(row)

    if edge_list is None and "edges" in mapping and mapping["edges"] is not None:
        edge_list = [tuple(edge[:2]) for edge in list(mapping["edges"])]

    return rows, edge_list, metadata_pathways


def _rows_to_positions(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([[row["x_m"], row["y_m"], row["z_m"]] for row in rows], dtype=float)


def _set_positions(rows: list[dict[str, Any]], positions_m: np.ndarray) -> None:
    for row, xyz in zip(rows, positions_m, strict=True):
        row["x_m"] = float(xyz[0])
        row["y_m"] = float(xyz[1])
        row["z_m"] = float(xyz[2])


def _jitter_close_positions(
    positions_m: np.ndarray,
    *,
    min_separation_m: float,
    seed: int,
    max_iter: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    if min_separation_m <= 0:
        return positions_m.copy(), np.zeros(positions_m.shape[0], dtype=bool)

    rng = np.random.default_rng(seed)
    out = np.array(positions_m, dtype=float, copy=True)
    moved = np.zeros(out.shape[0], dtype=bool)

    for _ in range(max_iter):
        changed = False
        for i in range(out.shape[0]):
            for j in range(i + 1, out.shape[0]):
                delta = out[j] - out[i]
                dist = float(np.linalg.norm(delta))
                if dist >= min_separation_m:
                    continue
                if dist < 1.0e-18:
                    direction = rng.normal(size=3)
                    norm = float(np.linalg.norm(direction))
                    if norm < 1.0e-18:
                        direction = np.asarray([1.0, 0.0, 0.0])
                    else:
                        direction = direction / norm
                else:
                    direction = delta / dist
                shift = 0.5 * (min_separation_m - dist + 1.0e-12) * direction
                out[i] -= shift
                out[j] += shift
                moved[i] = True
                moved[j] = True
                changed = True
        if not changed:
            break

    return out, moved


def _minimum_pairwise_distance(positions_m: np.ndarray) -> float:
    if positions_m.shape[0] < 2:
        return math.inf
    best = math.inf
    for i in range(positions_m.shape[0]):
        d = np.linalg.norm(positions_m[i + 1 :] - positions_m[i], axis=1)
        if d.size:
            best = min(best, float(np.min(d)))
    return best


def _unique_ordered(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


def _get_column_meta(network: Any, rows: Sequence[Mapping[str, Any]]) -> list[_ColumnMeta]:
    meta_source = None
    if isinstance(network, Mapping):
        meta_source = network.get("__column_meta__")

    if meta_source:
        result: list[_ColumnMeta] = []
        for item in meta_source:
            result.append(
                _ColumnMeta(
                    area=str(item["area"]),
                    center_m=(float(item["center_m"][0]), float(item["center_m"][1])),
                    radius_m=float(item["radius_m"]),
                    z0_m=float(item["z0_m"]),
                    z1_m=float(item["z1_m"]),
                    layer_boundaries_m=tuple(float(x) for x in item["layer_boundaries_m"]),
                )
            )
        return result

    metas: list[_ColumnMeta] = []
    areas = _unique_ordered(row["area"] for row in rows)
    for area in areas:
        sub = [row for row in rows if str(row["area"]) == area]
        xyz = np.asarray([[row["x_m"], row["y_m"], row["z_m"]] for row in sub], dtype=float)
        center = (float(np.mean(xyz[:, 0])), float(np.mean(xyz[:, 1])))
        radial = np.sqrt((xyz[:, 0] - center[0]) ** 2 + (xyz[:, 1] - center[1]) ** 2)
        radius = float(max(np.percentile(radial, 95), np.max(radial), 1.0e-6))
        z0 = float(np.min(xyz[:, 2]))
        z1 = float(np.max(xyz[:, 2]))
        layers = _unique_ordered(row["layer"] for row in sub)
        if set(layers) >= {"superficial", "mid", "deep"}:
            zs = {layer: [float(row["z_m"]) for row in sub if str(row["layer"]) == layer] for layer in layers}
            b1 = 0.5 * (max(zs["superficial"]) + min(zs["mid"]))
            b2 = 0.5 * (max(zs["mid"]) + min(zs["deep"]))
            boundaries = (z0, b1, b2, z1)
        else:
            boundaries = tuple(np.linspace(z0, z1, max(2, len(layers) + 1)))
        metas.append(
            _ColumnMeta(
                area=area,
                center_m=center,
                radius_m=radius,
                z0_m=z0,
                z1_m=z1,
                layer_boundaries_m=boundaries,
            )
        )
    return metas


def _add_column_wireframes(go: Any, fig: Any, metas: Sequence[_ColumnMeta], *, scale: float) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 96)
    for meta in metas:
        cx, cy = meta.center_m
        r = meta.radius_m
        for z in meta.layer_boundaries_m:
            fig.add_trace(
                go.Scatter3d(
                    x=(cx + r * np.cos(theta)) * scale,
                    y=(cy + r * np.sin(theta)) * scale,
                    z=np.full_like(theta, z) * scale,
                    mode="lines",
                    line={"width": 2, "color": "rgba(180, 200, 255, 0.35)"},
                    name=f"{meta.area} layer boundary",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            x = (cx + r * math.cos(angle)) * scale
            y = (cy + r * math.sin(angle)) * scale
            fig.add_trace(
                go.Scatter3d(
                    x=[x, x],
                    y=[y, y],
                    z=[meta.z0_m * scale, meta.z1_m * scale],
                    mode="lines",
                    line={"width": 1, "color": "rgba(180, 200, 255, 0.25)"},
                    name=f"{meta.area} shell",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )


def _edge_xyz(
    edge: tuple[Any, Any],
    id_to_position: Mapping[Any, np.ndarray],
) -> tuple[list[float], list[float], list[float]] | None:
    pre, post = edge[:2]
    if pre not in id_to_position or post not in id_to_position:
        return None
    p0 = id_to_position[pre]
    p1 = id_to_position[post]
    return [float(p0[0]), float(p1[0]), None], [float(p0[1]), float(p1[1]), None], [
        float(p0[2]),
        float(p1[2]),
        None,
    ]


def _add_edges(
    go: Any,
    fig: Any,
    rows: Sequence[Mapping[str, Any]],
    edges: Sequence[tuple[Any, Any]],
    *,
    scale: float,
    max_edges: int,
    seed: int,
) -> None:
    if not edges:
        return
    selected = list(edges)
    if len(selected) > max_edges:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(selected), size=max_edges, replace=False))
        selected = [selected[int(i)] for i in idx]

    id_to_position = {
        row["neuron_id"]: np.asarray([row["x_m"], row["y_m"], row["z_m"]], dtype=float) * scale
        for row in rows
    }
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for edge in selected:
        coords = _edge_xyz(edge, id_to_position)
        if coords is None:
            continue
        x, y, z = coords
        xs.extend(x)
        ys.extend(y)
        zs.extend(z)

    if xs:
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line={"width": 1, "color": "rgba(180, 180, 180, 0.20)"},
                name=f"sampled edges (n={len(selected)})",
                hoverinfo="skip",
            )
        )


def _as_pathway_endpoint(value: Any, *, scale: float) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError("pathway endpoints must be 3-element coordinates")
    return arr * scale


def _add_pathways(go: Any, fig: Any, pathways: Sequence[Mapping[str, Any]], *, scale: float) -> None:
    for pathway in pathways:
        name = str(pathway.get("name", pathway.get("pathway", "pathway")))
        start = _as_pathway_endpoint(pathway["start_m"], scale=scale)
        end = _as_pathway_endpoint(pathway["end_m"], scale=scale)
        color = str(pathway.get("color", _PATHWAY_COLORS.get(name.lower(), "#FFFFFF")))
        width = float(pathway.get("width", 5.0))
        show_arrow = bool(pathway.get("arrow", True))

        fig.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line={"width": width, "color": color},
                name=name,
                hovertemplate=(
                    f"{name}<br>"
                    "start=(%{x:.1f}, %{y:.1f}, %{z:.1f})<br>"
                    "end shown by line<extra></extra>"
                ),
            )
        )

        if show_arrow:
            vec = end - start
            norm = float(np.linalg.norm(vec))
            if norm > 1.0e-12:
                unit = vec / norm
                fig.add_trace(
                    go.Cone(
                        x=[end[0]],
                        y=[end[1]],
                        z=[end[2]],
                        u=[unit[0]],
                        v=[unit[1]],
                        w=[unit[2]],
                        sizemode="absolute",
                        sizeref=max(norm * 0.08, 8.0),
                        anchor="tip",
                        colorscale=[[0.0, color], [1.0, color]],
                        showscale=False,
                        name=f"{name} arrow",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )


def _node_hover_text(row: Mapping[str, Any], *, scale: float, display_unit: str) -> str:
    x = float(row["x_m"]) * scale
    y = float(row["y_m"]) * scale
    z = float(row["z_m"]) * scale
    return (
        f"neuron_id: {row['neuron_id']}<br>"
        f"area: {row['area']}<br>"
        f"layer: {row['layer']}<br>"
        f"cell_type: {row['cell_type']}<br>"
        f"x_{display_unit}: {x:.3f}<br>"
        f"y_{display_unit}: {y:.3f}<br>"
        f"z_{display_unit}: {z:.3f}<br>"
        f"jittered: {row.get('jittered', False)}"
    )


def visualize_network_3d(
    network: Any,
    *,
    output_html: str | Path | None = None,
    title: str = "jbiophysic network anatomy",
    positions_key: str = "positions_m",
    coordinate_unit: str = "m",
    display_unit: str = "um",
    dimensions: str = "auto",
    node_id_key: str = "neuron_id",
    cell_type_key: str = "cell_type",
    layer_key: str = "layer",
    area_key: str = "area",
    edges: Sequence[tuple[Any, Any]] | None = None,
    pathways: Sequence[Mapping[str, Any]] | None = None,
    show_layers: bool = True,
    show_column_shells: bool = True,
    show_edges: bool = False,
    max_edges: int = 500,
    seed: int = 17,
    min_separation_m: float | None = None,
    jitter_duplicates: bool = True,
    template: str = "plotly_dark",
    return_node_table: bool = False,
):
    """Render a network anatomy/scaffold as an interactive Plotly 3-D figure.

    Parameters
    ----------
    network:
        Population object, mapping, table-like records, or networkx-like graph.
        Coordinates may be 1-D, 2-D, or 3-D; they are always rendered in 3-D.
    output_html:
        Optional path for writing a standalone HTML figure.
    coordinate_unit, display_unit:
        Unit conversion for axes and hover labels. The default converts meters
        to micrometers.
    min_separation_m:
        Threshold for deterministic duplicate/near-duplicate jitter. If omitted,
        exact duplicates are still gently separated when ``jitter_duplicates`` is
        true.
    return_node_table:
        If true, return ``(figure, node_rows)`` where ``node_rows`` is a list of
        dictionaries containing plotted coordinates and metadata.
    """
    del dimensions  # retained for API stability; dimensionality is inferred from coordinates.

    go = _require_plotly()
    scale = _unit_scale(coordinate_unit, display_unit)

    rows, inferred_edges, metadata_pathways = _coerce_network_to_table(
        network,
        positions_key=positions_key,
        node_id_key=node_id_key,
        cell_type_key=cell_type_key,
        layer_key=layer_key,
        area_key=area_key,
    )

    if min_separation_m is None:
        min_separation_m = 1.0e-12 if jitter_duplicates else 0.0

    if jitter_duplicates and min_separation_m > 0:
        original = _rows_to_positions(rows)
        jittered, moved = _jitter_close_positions(
            original,
            min_separation_m=min_separation_m,
            seed=seed,
        )
        _set_positions(rows, jittered)
        for row, was_moved in zip(rows, moved, strict=True):
            row["jittered"] = bool(was_moved)

    edge_source = list(edges) if edges is not None else inferred_edges
    pathway_source = list(pathways) if pathways is not None else metadata_pathways

    fig = go.Figure()

    if show_column_shells or show_layers:
        metas = _get_column_meta(network, rows)
        _add_column_wireframes(go, fig, metas, scale=scale)

    if show_edges and edge_source:
        _add_edges(
            go,
            fig,
            rows,
            edge_source,
            scale=scale,
            max_edges=max_edges,
            seed=seed,
        )

    if pathway_source:
        _add_pathways(go, fig, pathway_source, scale=scale)

    cell_types = _unique_ordered(row["cell_type"] for row in rows)
    for cell_type in cell_types:
        sub = [row for row in rows if str(row["cell_type"]) == cell_type]
        xyz = np.asarray([[row["x_m"], row["y_m"], row["z_m"]] for row in sub], dtype=float) * scale
        color = _COLOR_BY_CELL_TYPE.get(cell_type, _COLOR_BY_CELL_TYPE["unknown"])
        symbol = _SYMBOL_BY_CELL_TYPE.get(cell_type, _SYMBOL_BY_CELL_TYPE["unknown"])
        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                marker={
                    "size": 5 if cell_type == "E" else 6,
                    "color": color,
                    "symbol": symbol,
                    "opacity": 0.88,
                    "line": {"width": 0.5, "color": "rgba(255,255,255,0.30)"},
                },
                name=cell_type,
                text=[_node_hover_text(row, scale=scale, display_unit=display_unit) for row in sub],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    axis_suffix = f"({display_unit})"
    fig.update_layout(
        title=title,
        template=template,
        scene={
            "xaxis_title": f"X {axis_suffix}",
            "yaxis_title": f"Y {axis_suffix}",
            "zaxis_title": f"Depth {axis_suffix}",
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.7, "y": 1.8, "z": 1.25}},
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
    )

    if output_html is not None:
        out = Path(output_html)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)

    if return_node_table:
        return fig, rows
    return fig


def _sample_cylinder_no_overlap(
    rng: np.random.Generator,
    *,
    n: int,
    radius_m: float,
    z0_m: float,
    z1_m: float,
    min_separation_m: float,
    max_attempts: int = 200_000,
) -> np.ndarray:
    points: list[np.ndarray] = []
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        r = radius_m * math.sqrt(float(rng.uniform()))
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        candidate = np.asarray(
            [
                r * math.cos(theta),
                r * math.sin(theta),
                float(rng.uniform(z0_m, z1_m)),
            ],
            dtype=float,
        )
        if not points:
            points.append(candidate)
            continue
        existing = np.asarray(points, dtype=float)
        if float(np.min(np.linalg.norm(existing - candidate, axis=1))) >= min_separation_m:
            points.append(candidate)

    if len(points) != n:
        raise RuntimeError(
            f"Could not sample {n} non-overlapping points in cylinder after {max_attempts} attempts. "
            "Reduce min_separation_m or increase radius/depth."
        )
    return np.asarray(points, dtype=float)


def build_laminar_population_anatomy(
    *,
    seed: int = 17,
    tube_depth_m: float = 1.0e-3,
    tube_radius_m: float = 0.1e-3,
    layer_depths_m: tuple[float, float, float] = (0.3e-3, 0.1e-3, 0.6e-3),
    min_separation_m: float = 4.0e-6,
    area: str = "cortex",
    start_neuron_id: int = 0,
) -> dict[str, Any]:
    """Generate a deterministic laminar E/PV/SST anatomy scaffold.

    Counts match the TFNE-Izhikevich laminar E/I starter:
    superficial = 30 E + 4 PV + 7 SST; mid = 5 E + 5 PV;
    deep = 40 E + 6 PV + 3 SST. Total = 100.
    """
    d_sup, d_mid, d_deep = layer_depths_m
    if not math.isclose(d_sup + d_mid + d_deep, tube_depth_m, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("layer_depths_m must sum to tube_depth_m")

    rng = np.random.default_rng(seed)
    layer_specs = [
        ("superficial", 0.0, d_sup, {"E": 30, "PV": 4, "SST": 7}),
        ("mid", d_sup, d_sup + d_mid, {"E": 5, "PV": 5, "SST": 0}),
        ("deep", d_sup + d_mid, tube_depth_m, {"E": 40, "PV": 6, "SST": 3}),
    ]

    positions: list[np.ndarray] = []
    layers: list[str] = []
    cell_types: list[str] = []
    for layer_name, z0, z1, counts in layer_specs:
        for cell_type, count in counts.items():
            if count <= 0:
                continue
            pts = _sample_cylinder_no_overlap(
                rng,
                n=count,
                radius_m=tube_radius_m,
                z0_m=z0,
                z1_m=z1,
                min_separation_m=min_separation_m,
            )
            positions.append(pts)
            layers.extend([layer_name] * count)
            cell_types.extend([cell_type] * count)

    positions_m = np.vstack(positions)
    n = positions_m.shape[0]
    if n != 100:
        raise AssertionError(f"expected 100 neurons, got {n}")

    return {
        "positions_m": positions_m,
        "neuron_id": np.arange(start_neuron_id, start_neuron_id + n, dtype=int),
        "cell_type": np.asarray(cell_types, dtype="U16"),
        "layer": np.asarray(layers, dtype="U32"),
        "area": np.asarray([area] * n, dtype="U32"),
        "__column_meta__": [
            {
                "area": area,
                "center_m": [0.0, 0.0],
                "radius_m": tube_radius_m,
                "z0_m": 0.0,
                "z1_m": tube_depth_m,
                "layer_boundaries_m": [0.0, d_sup, d_sup + d_mid, tube_depth_m],
            }
        ],
    }


def _network_from_rows(rows: Sequence[Mapping[str, Any]], *, column_meta: Sequence[Mapping[str, Any]]):
    positions_m = np.asarray([[row["x_m"], row["y_m"], row["z_m"]] for row in rows], dtype=float)
    return {
        "positions_m": positions_m,
        "neuron_id": np.asarray([row["neuron_id"] for row in rows]),
        "cell_type": np.asarray([row["cell_type"] for row in rows], dtype="U32"),
        "layer": np.asarray([row["layer"] for row in rows], dtype="U64"),
        "area": np.asarray([row["area"] for row in rows], dtype="U64"),
        "__column_meta__": list(column_meta),
    }


def _coerce_population_to_base_network(population: Any) -> dict[str, Any]:
    rows, _, _ = _coerce_network_to_table(
        population,
        positions_key="positions_m",
        node_id_key="neuron_id",
        cell_type_key="cell_type",
        layer_key="layer",
        area_key="area",
    )
    return _network_from_rows(rows, column_meta=[])


def build_two_cortex_laminar_anatomy_from_population(
    population: Any,
    *,
    offset_m: float = 0.55e-3,
    lower_area: str = "lower-cortex",
    higher_area: str = "higher-cortex",
    seed: int = 17,
    min_separation_m: float = 4.0e-6,
    column_radius_m: float | None = None,
    layer_boundaries_m: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Duplicate one laminar population into lower- and higher-cortex columns."""
    base = _coerce_population_to_base_network(population)
    positions = np.asarray(base["positions_m"], dtype=float)
    n = positions.shape[0]

    if column_radius_m is None:
        radial = np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)
        column_radius_m = float(max(np.max(radial), 0.1e-3))

    if layer_boundaries_m is None:
        z0 = float(np.min(positions[:, 2]))
        z1 = float(np.max(positions[:, 2]))
        layer_boundaries_m = (z0, z0 + 0.3e-3, z0 + 0.4e-3, z1)

    lower_center = np.asarray([offset_m / 2.0, 0.0, 0.0], dtype=float)
    higher_center = np.asarray([-offset_m / 2.0, 0.0, 0.0], dtype=float)

    lower_positions = positions + lower_center
    higher_positions = positions + higher_center

    all_positions = np.vstack([lower_positions, higher_positions])
    all_positions, moved = _jitter_close_positions(
        all_positions,
        min_separation_m=min_separation_m,
        seed=seed,
    )

    neuron_ids = np.concatenate(
        [
            np.asarray([f"L-{x}" for x in base["neuron_id"]], dtype=object),
            np.asarray([f"H-{x}" for x in base["neuron_id"]], dtype=object),
        ]
    )
    cell_type = np.concatenate([base["cell_type"], base["cell_type"]])
    layer = np.concatenate([base["layer"], base["layer"]])
    area = np.concatenate(
        [
            np.asarray([lower_area] * n, dtype="U32"),
            np.asarray([higher_area] * n, dtype="U32"),
        ]
    )

    z_mid = 0.5 * (float(min(layer_boundaries_m)) + float(max(layer_boundaries_m)))
    z_superficial = float(layer_boundaries_m[1]) * 0.5
    z_deep = 0.5 * (float(layer_boundaries_m[2]) + float(layer_boundaries_m[-1]))

    network = {
        "positions_m": all_positions,
        "neuron_id": neuron_ids,
        "cell_type": cell_type,
        "layer": layer,
        "area": area,
        "jittered": moved,
        "__column_meta__": [
            {
                "area": lower_area,
                "center_m": [float(lower_center[0]), 0.0],
                "radius_m": column_radius_m,
                "z0_m": float(layer_boundaries_m[0]),
                "z1_m": float(layer_boundaries_m[-1]),
                "layer_boundaries_m": [float(x) for x in layer_boundaries_m],
            },
            {
                "area": higher_area,
                "center_m": [float(higher_center[0]), 0.0],
                "radius_m": column_radius_m,
                "z0_m": float(layer_boundaries_m[0]),
                "z1_m": float(layer_boundaries_m[-1]),
                "layer_boundaries_m": [float(x) for x in layer_boundaries_m],
            },
        ],
        "__pathways__": [
            {
                "name": "thalamic_input",
                "start_m": [float(lower_center[0] + 0.38e-3), 0.0, z_mid],
                "end_m": [float(lower_center[0] + column_radius_m), 0.0, z_mid],
                "color": _PATHWAY_COLORS["thalamic_input"],
                "width": 6,
                "arrow": True,
            },
            {
                "name": "feedforward",
                "start_m": [float(lower_center[0]), 0.0, z_superficial],
                "end_m": [float(higher_center[0]), 0.0, z_mid],
                "color": _PATHWAY_COLORS["feedforward"],
                "width": 5,
                "arrow": True,
            },
            {
                "name": "feedback",
                "start_m": [float(higher_center[0]), 0.0, z_mid],
                "end_m": [float(lower_center[0]), 0.0, z_deep],
                "color": _PATHWAY_COLORS["feedback"],
                "width": 5,
                "arrow": True,
            },
        ],
    }

    min_dist = _minimum_pairwise_distance(np.asarray(network["positions_m"], dtype=float))
    if min_dist < min_separation_m:
        raise RuntimeError(
            f"two-column anatomy has overlapping coordinates: min distance {min_dist:.3e} m "
            f"< threshold {min_separation_m:.3e} m"
        )

    return network


def build_two_cortex_laminar_anatomy(
    *,
    seed: int = 17,
    offset_m: float = 0.55e-3,
    min_separation_m: float = 4.0e-6,
) -> dict[str, Any]:
    """Build a deterministic two-column lower/higher cortex laminar anatomy demo."""
    base = build_laminar_population_anatomy(
        seed=seed,
        min_separation_m=min_separation_m,
        area="starter-cortex",
    )
    return build_two_cortex_laminar_anatomy_from_population(
        base,
        offset_m=offset_m,
        seed=seed,
        min_separation_m=min_separation_m,
        column_radius_m=0.1e-3,
        layer_boundaries_m=(0.0, 0.3e-3, 0.4e-3, 1.0e-3),
    )
