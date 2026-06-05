"""Dependency-safe jaxfne playground smoke entrypoint.

Default mode writes a request manifest and does not require jaxfne. Pass
``--execute`` to run the jaxfne public API smoke when jaxfne is installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from jbiophysic.playgrounds import (
    JaxfnePlaygroundSpec,
    build_request_manifest,
    run_playground_smoke,
    write_playground_receipt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="suite2_four_celltype")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-ms", type=float, default=10.0)
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument("--cell-type", default="E")
    parser.add_argument("--out", default="outputs/jaxfne_playground_smoke")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    spec = JaxfnePlaygroundSpec(
        name=args.name,
        seed=args.seed,
        duration_ms=args.duration_ms,
        dt_ms=args.dt_ms,
        cell_type=args.cell_type,
    )
    out_dir = Path(args.out)
    if args.execute:
        receipt = run_playground_smoke(spec)
        target = out_dir / "jaxfne_playground_smoke_receipt.json"
    else:
        receipt = build_request_manifest(spec)
        target = out_dir / "jaxfne_playground_request_manifest.json"

    path = write_playground_receipt(receipt, target)
    print(path)


if __name__ == "__main__":
    main()
