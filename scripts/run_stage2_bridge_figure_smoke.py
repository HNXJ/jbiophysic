#!/usr/bin/env python
"""Stage 2B bridge figure artifact smoke test.

Generates reviewer-ready artifact bundle with:
- manifest.json
- jaxfne_bridge_report.json
- operator_status.json
- figure_metrics.csv
- asset_hashes.json
- figures/ with bridge-validation or simulation-result figures

Honest classification:
- If jaxfne executes: simulation_executed=true, figures from real output
- If jaxfne unavailable: simulation_executed=false, bridge-validation figures only
- Never fake simulation claims
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jbiophysic.bridges.jaxfne import (
    build_single_neuron_run,
    run_and_report,
    validate_manifest_json,
)


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def create_bridge_manifest_summary_figure(
    output_dir: Path,
    manifest: dict,
    report: dict,
) -> Path:
    """Create bridge manifest summary figure.

    Always shows manifest metadata and validation status.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.axis("off")

    simulation_executed = report.get("simulation_executed", False)
    dispatch_status = report.get("dispatch_status", "unknown")
    figure_type = report.get("figure_type", "unknown")

    # Build text content
    text_lines = [
        f"jbiophysic Stage 2B Bridge Manifest Summary",
        f"{'=' * 50}",
        "",
        f"Simulation Executed: {simulation_executed}",
        f"Figure Type: {figure_type}",
        f"Dispatch Status: {dispatch_status}",
        "",
        f"Run Type: {manifest.get('run_type', 'unknown')}",
        f"Source Type: {manifest.get('source_type', 'unknown')}",
        f"Source Scale: {manifest.get('source_scale', 'unknown')}",
        f"Source Calibration: {manifest.get('source_calibration_status', 'unknown')}",
        "",
        f"Truth Mode: {manifest.get('truth_mode', 'unknown')}",
        f"Claim Level: {manifest.get('claim_level', 'unknown')}",
        f"Physical Amplitude Allowed: {manifest.get('physical_amplitude_claim_allowed', False)}",
        "",
        f"Field Solver Status: {manifest.get('field_solver_status', 'unknown')}",
        f"Run ID: {manifest.get('run_id', 'unknown')[:12]}...",
        "",
        f"Parameters:",
        f"  Duration: {manifest.get('parameters', {}).get('duration_ms', '?')} ms",
        f"  Timestep: {manifest.get('parameters', {}).get('dt_ms', '?')} ms",
        f"  Steps: {manifest.get('parameters', {}).get('n_steps', '?')}",
        "",
        f"jaxfne Version: {manifest.get('jaxfne_engine_version', 'unknown')}",
    ]

    text_content = "\n".join(text_lines)
    ax.text(
        0.05,
        0.95,
        text_content,
        transform=ax.transAxes,
        fontfamily="monospace",
        fontsize=9,
        verticalalignment="top",
    )

    # Title with simulation classification
    title = f"bridge_manifest_summary"
    if not simulation_executed:
        title += " [bridge_validation_not_simulation]"

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()

    path = output_dir / "figures" / "bridge_manifest_summary.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return path


def create_dispatch_status_panel_figure(
    output_dir: Path,
    manifest: dict,
    report: dict,
) -> Path:
    """Create dispatch status panel figure."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.axis("off")

    simulation_executed = report.get("simulation_executed", False)
    dispatch_status = report.get("dispatch_status", "unknown")
    success = report.get("success", False)
    jaxfne_version = report.get("jaxfne_engine_version", "unknown")
    supported_api = report.get("supported_jaxfne_api_found", False)
    errors = report.get("errors", [])

    # Build text content
    text_lines = [
        f"jbiophysic Stage 2B Dispatch Status Panel",
        f"{'=' * 50}",
        "",
        f"Simulation Executed: {simulation_executed}",
        f"Dispatch Status: {dispatch_status}",
        f"Success: {success}",
        "",
        f"jaxfne Engine Version: {jaxfne_version}",
        f"Supported API Found: {supported_api}",
        "",
    ]

    if errors:
        text_lines.append("Errors:")
        for i, err in enumerate(errors[:5]):  # Limit to 5 errors
            text_lines.append(f"  {i+1}. {str(err)[:60]}")
        if len(errors) > 5:
            text_lines.append(f"  ... and {len(errors) - 5} more")
        text_lines.append("")

    text_lines.extend([
        "Classification:",
        f"  truth_mode: {manifest.get('truth_mode', '?')}",
        f"  claim_level: {manifest.get('claim_level', '?')}",
        f"  field_solver_status: {manifest.get('field_solver_status', '?')}",
        f"  physical_amplitude_claim_allowed: {manifest.get('physical_amplitude_claim_allowed', False)}",
    ])

    text_content = "\n".join(text_lines)
    ax.text(
        0.05,
        0.95,
        text_content,
        transform=ax.transAxes,
        fontfamily="monospace",
        fontsize=9,
        verticalalignment="top",
    )

    fig.suptitle("dispatch_status_panel", fontsize=12, fontweight="bold")
    fig.tight_layout()

    path = output_dir / "figures" / "dispatch_status_panel.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return path


def create_simulation_voltage_trace_figure(
    output_dir: Path,
    manifest: dict,
    report: dict,
) -> Path | None:
    """Create voltage trace figure if simulation output exists.

    Returns None if no simulation output available.
    """
    # Try to extract v_trace from report
    harmonized = report.get("harmonized_output", {})
    v_trace = harmonized.get("v_trace")

    if v_trace is None:
        return None

    # Try to get jaxfne output
    jaxfne_output = report.get("jaxfne_output", {})
    v_trace_jaxfne = jaxfne_output.get("v_trace")

    if v_trace_jaxfne is not None:
        v_trace = v_trace_jaxfne

    if v_trace is None:
        return None

    # Convert to numpy if needed
    if not isinstance(v_trace, np.ndarray):
        try:
            v_trace = np.asarray(v_trace)
        except Exception:
            return None

    if len(v_trace) == 0:
        return None

    # Compute time axis
    params = manifest.get("parameters", {})
    duration_ms = params.get("duration_ms", 500.0)
    n_steps = params.get("n_steps", len(v_trace))
    dt_ms = duration_ms / max(1, n_steps - 1)
    time_ms = np.linspace(0, duration_ms, len(v_trace))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(time_ms, v_trace, "b-", linewidth=1.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane Potential (mV)")
    ax.set_title(
        f"single_neuron_voltage_trace [simulation_result from jaxfne]"
    )
    ax.grid(True, alpha=0.3)

    # Add metadata text box
    metadata_text = (
        f"Source: {manifest.get('source_type', '?')}\n"
        f"Scale: {manifest.get('source_scale', '?')}\n"
        f"Duration: {duration_ms} ms\n"
        f"Timestep: {dt_ms:.4f} ms"
    )
    ax.text(
        0.98,
        0.02,
        metadata_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.tight_layout()

    path = output_dir / "figures" / "single_neuron_voltage_trace.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return path


def main():
    """Main Stage 2B execution."""
    parser = argparse.ArgumentParser(
        description="Stage 2B bridge figure artifact smoke test"
    )
    parser.add_argument("--out", default="outputs/stage2_bridge_figure_smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-ms", type=float, default=500.0)
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument("--format", default="png", choices=["png", "svg"])
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    print(f"Stage 2B Bridge Figure Smoke Test")
    print(f"Output directory: {output_dir}")
    print()

    # Build manifest
    print("Building manifest...")
    manifest = build_single_neuron_run(
        cell_type="izhikevich",
        params={
            "a": 0.02,
            "b": 0.2,
            "c": -65.0,
            "d": 8.0,
            "I_inj_nA": 10.0,
        },
        stimulus_pattern={
            "kind": "step",
            "start_ms": 100.0,
            "stop_ms": 400.0,
        },
        duration_ms=args.duration_ms,
        dt_ms=args.dt_ms,
        seed=args.seed,
    )

    # Validate
    print("Validating manifest...")
    is_valid, errors = validate_manifest_json(manifest, strict_mode=True)
    if not is_valid:
        print(f"ERROR: Manifest validation failed: {errors}")
        sys.exit(1)

    # Run and report
    print("Attempting jaxfne dispatch...")
    report = run_and_report(manifest, str(output_dir))

    # Determine simulation status
    simulation_executed = report.get("simulation_executed", False)
    dispatch_status = report.get("dispatch_status", "unknown")
    success = report.get("success", False)

    print(f"  dispatch_status: {dispatch_status}")
    print(f"  success: {success}")
    print(f"  simulation_executed: {simulation_executed}")

    # Generate figures
    print("\nGenerating figures...")
    figures_generated = []

    # Always generate bridge manifest summary
    fig_path = create_bridge_manifest_summary_figure(output_dir, manifest, report)
    figures_generated.append(
        {
            "path": fig_path,
            "filename": fig_path.name,
            "kind": "manifest_summary",
            "derived_from": "manifest_metadata",
        }
    )
    print(f"  Created: {fig_path.name}")

    # Always generate dispatch status panel
    fig_path = create_dispatch_status_panel_figure(output_dir, manifest, report)
    figures_generated.append(
        {
            "path": fig_path,
            "filename": fig_path.name,
            "kind": "dispatch_status",
            "derived_from": "dispatch_metadata",
        }
    )
    print(f"  Created: {fig_path.name}")

    # Generate simulation voltage trace if execution succeeded
    if simulation_executed:
        fig_path = create_simulation_voltage_trace_figure(output_dir, manifest, report)
        if fig_path and fig_path.exists():
            figures_generated.append(
                {
                    "path": fig_path,
                    "filename": fig_path.name,
                    "kind": "voltage_trace",
                    "derived_from": "jaxfne_output",
                }
            )
            print(f"  Created: {fig_path.name}")

    # Determine figure classification
    if simulation_executed:
        figure_type = "simulation_result"
    else:
        figure_type = "bridge_validation_not_simulation"

    # Update report with Stage 2B fields
    report["artifact_stage"] = "stage2b_figure_smoke"
    report["simulation_executed"] = simulation_executed
    report["figure_type"] = figure_type
    report["figure_count"] = len(figures_generated)
    report["figures"] = [
        {
            "path": f.get("filename", "?"),
            "filename": f.get("filename", "?"),
            "kind": f.get("kind", "?"),
            "derived_from": f.get("derived_from", "?"),
            "sha256": "",  # Will be filled in asset_hashes
            "size_bytes": f["path"].stat().st_size if f["path"].exists() else 0,
        }
        for f in figures_generated
    ]

    # Write JSON files
    print("\nWriting JSON artifacts...")

    # Write manifest
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print("  manifest.json")

    # Write report
    with open(output_dir / "jaxfne_bridge_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("  jaxfne_bridge_report.json")

    # Write operator status
    with open(output_dir / "operator_status.json", "w") as f:
        json.dump(manifest.get("operator_status", {}), f, indent=2)
    print("  operator_status.json")

    # Write figure metrics CSV
    print("\nWriting figure metrics...")
    utc_now = datetime.now(timezone.utc).isoformat()

    with open(output_dir / "figure_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "figure_name",
                "figure_path",
                "figure_type",
                "derived_from",
                "simulation_executed",
                "dispatch_status",
                "source_type",
                "source_scale",
                "source_calibration_status",
                "physical_amplitude_claim_allowed",
                "size_bytes",
                "sha256",
                "created_utc",
            ],
        )
        writer.writeheader()

        for fig_info in figures_generated:
            fig_path = fig_info["path"]
            size_bytes = fig_path.stat().st_size if fig_path.exists() else 0
            hash_val = sha256_file(fig_path) if fig_path.exists() else "unknown"

            writer.writerow(
                {
                    "run_id": manifest.get("run_id", "?"),
                    "figure_name": fig_info.get("filename", "?"),
                    "figure_path": str(fig_path.relative_to(output_dir)),
                    "figure_type": fig_info.get("kind", "?"),
                    "derived_from": fig_info.get("derived_from", "?"),
                    "simulation_executed": simulation_executed,
                    "dispatch_status": dispatch_status,
                    "source_type": manifest.get("source_type", "?"),
                    "source_scale": manifest.get("source_scale", "?"),
                    "source_calibration_status": manifest.get(
                        "source_calibration_status", "?"
                    ),
                    "physical_amplitude_claim_allowed": manifest.get(
                        "physical_amplitude_claim_allowed", False
                    ),
                    "size_bytes": size_bytes,
                    "sha256": hash_val,
                    "created_utc": utc_now,
                }
            )

    print("  figure_metrics.csv")

    # Write asset hashes
    print("\nComputing asset hashes...")
    assets = {}

    for asset_file in ["manifest.json", "jaxfne_bridge_report.json", "operator_status.json", "figure_metrics.csv"]:
        path = output_dir / asset_file
        if path.exists():
            assets[asset_file] = sha256_file(path)

    for fig_info in figures_generated:
        fig_path = fig_info["path"]
        if fig_path.exists():
            rel_path = str(fig_path.relative_to(output_dir))
            assets[rel_path] = sha256_file(fig_path)

    asset_hashes = {
        "created_utc": utc_now,
        "output_dir": str(output_dir),
        "assets": assets,
    }

    with open(output_dir / "asset_hashes.json", "w") as f:
        json.dump(asset_hashes, f, indent=2)
    print("  asset_hashes.json")

    # Summary
    print("\n" + "=" * 60)
    print("STAGE 2B SMOKE TEST SUMMARY")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Dispatch status: {dispatch_status}")
    print(f"Success: {success}")
    print(f"Simulation executed: {simulation_executed}")
    print(f"Figure type: {figure_type}")
    print(f"Figures generated: {len(figures_generated)}")
    print()
    print("Artifacts:")
    for item in ["manifest.json", "jaxfne_bridge_report.json", "operator_status.json", "figure_metrics.csv", "asset_hashes.json"]:
        path = output_dir / item
        size = path.stat().st_size if path.exists() else 0
        print(f"  {item}: {size} bytes")
    print()
    print("Figures:")
    for fig_info in figures_generated:
        fig_path = fig_info["path"]
        size = fig_path.stat().st_size if fig_path.exists() else 0
        print(f"  {fig_info.get('filename', '?')}: {size} bytes ({fig_info.get('derived_from', '?')})")

    print()
    if simulation_executed:
        print("DECISION CANDIDATE: ACCEPT_STAGE2B_SIMULATION_FIGURES_VALIDATED")
    else:
        print("DECISION CANDIDATE: ACCEPT_STAGE2B_BRIDGE_FIGURES_ONLY")

    return 0


if __name__ == "__main__":
    sys.exit(main())
