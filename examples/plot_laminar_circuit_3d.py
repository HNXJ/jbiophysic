#!/usr/bin/env python3
"""Render an interactive Plotly 3-D laminar circuit anatomy demo.

This example only builds and visualizes anatomy/scaffold coordinates. It does
not run Izhikevich dynamics, TFNE source projection, Poisson field solves, CSD,
LFP, or biological mechanism validation.

Example
-------
PYTHONPATH=src python examples/plot_laminar_circuit_3d.py \
    --out outputs/visualization/laminar_two_cortex_network3d.html
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

from jbiophysic.viz.network3d import (
    build_two_cortex_laminar_anatomy,
    build_two_cortex_laminar_anatomy_from_population,
    visualize_network_3d,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_starter_population(repo_root: Path, seed: int) -> Any | None:
    candidates = [
        repo_root / "examples" / "tfne_izhikevich_laminar_ei100.py",
        repo_root / "scratch" / "tfne_izhikevich_laminar_ei100.py",
    ]
    for path in candidates:
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location("tfne_izhikevich_laminar_ei100", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "build_population"):
            continue

        if hasattr(module, "SimConfig"):
            cfg = module.SimConfig(seed=seed)
            return module.build_population(cfg)

        return module.build_population()

    return None


def build_network(seed: int):
    repo_root = _repo_root()
    population = _load_starter_population(repo_root, seed)
    if population is None:
        return build_two_cortex_laminar_anatomy(seed=seed)

    return build_two_cortex_laminar_anatomy_from_population(
        population,
        seed=seed,
        offset_m=0.55e-3,
        min_separation_m=4.0e-6,
        column_radius_m=0.1e-3,
        layer_boundaries_m=(0.0, 0.3e-3, 0.4e-3, 1.0e-3),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a two-column lower/higher cortex laminar anatomy scaffold as interactive HTML."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/visualization/laminar_two_cortex_network3d.html"),
        help="Output HTML path.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Deterministic anatomy/jitter seed.")
    parser.add_argument("--template", default="plotly_dark", help="Plotly template.")
    parser.add_argument("--show-edges", action="store_true", help="Show sampled edges if provided.")
    parser.add_argument("--no-shells", action="store_true", help="Disable column/layer wireframes.")
    args = parser.parse_args()

    network = build_network(args.seed)

    visualize_network_3d(
        network,
        output_html=args.out,
        title="TFNE-Izhikevich laminar circuit anatomy: lower/higher cortex",
        show_edges=args.show_edges,
        show_column_shells=not args.no_shells,
        show_layers=not args.no_shells,
        template=args.template,
        seed=args.seed,
        min_separation_m=4.0e-6,
        jitter_duplicates=True,
    )

    print(f"Wrote {args.out}")
    print("Scientific status: anatomy/scaffold visualization only; truth_safe_unverified.")


if __name__ == "__main__":
    main()
