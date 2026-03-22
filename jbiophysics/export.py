"""
jbiophysics.export — Multi-format results export.

Supports: Plotly HTML, SVG, Markdown, dict serialization.
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ResultsReport:
    """
    Container for simulation/optimization results.
    
    Attributes:
        traces: Voltage traces array (cells × time)
        params: Optimized parameters (jaxley format)
        loss_history: Loss values per epoch
        dt: Timestep in ms
        t_max: Total simulation time in ms
        metadata: Arbitrary metadata dict
    """
    traces: np.ndarray
    params: Any = None
    loss_history: List[float] = field(default_factory=list)
    dt: float = 0.1
    t_max: float = 1500.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def time_axis(self) -> np.ndarray:
        return np.arange(0, self.traces.shape[-1]) * self.dt
    
    @property
    def num_cells(self) -> int:
        return self.traces.shape[0]
    
    def to_plotly(self, output_path: Optional[str] = None) -> Any:
        """Generate interactive Plotly HTML dashboard."""
        from jbiophysics.viz.dashboard import generate_dashboard
        fig = generate_dashboard(self)
        if output_path:
            fig.write_html(output_path, include_plotlyjs="cdn")
            print(f"📊 Plotly dashboard saved → {output_path}")
        return fig
    
    def to_svg(self, output_path: str, panel: str = "raster") -> str:
        """Export a specific panel as SVG."""
        from jbiophysics.viz import raster, psd, traces as trace_viz
        panel_map = {"raster": raster.plot_raster, "psd": psd.plot_psd, "traces": trace_viz.plot_traces}
        if panel not in panel_map:
            raise ValueError(f"Unknown panel '{panel}'. Options: {list(panel_map.keys())}")
        fig = panel_map[panel](self)
        fig.write_image(output_path, format="svg")
        print(f"🖼️  SVG exported → {output_path}")
        return output_path
    
    def to_markdown(self, output_path: str, caption: str = "", include_figures: bool = True) -> str:
        """Generate markdown summary with optional embedded figures."""
        lines = [f"# Simulation Results", "",
                 f"**Cells**: {self.num_cells} | **dt**: {self.dt} ms | **Duration**: {self.t_max} ms", ""]
        if caption:
            lines.extend([f"> {caption}", ""])
        if self.loss_history:
            lines.extend([f"## Optimization",
                          f"- **Method**: {self.metadata.get('method', 'N/A')}",
                          f"- **Epochs**: {self.metadata.get('epochs', len(self.loss_history))}",
                          f"- **Final loss**: {self.loss_history[-1]:.6f}",
                          f"- **Best loss**: {min(self.loss_history):.6f}", ""])
        threshold = -20.0
        spikes = (self.traces[:, :-1] < threshold) & (self.traces[:, 1:] >= threshold)
        firing_rates = np.sum(spikes, axis=1) / (self.t_max / 1000.0)
        lines.extend([f"## Firing Rates",
                       f"- **Mean**: {np.mean(firing_rates):.1f} Hz",
                       f"- **Std**: {np.std(firing_rates):.1f} Hz",
                       f"- **Range**: [{np.min(firing_rates):.1f}, {np.max(firing_rates):.1f}] Hz", ""])
        md_content = "\n".join(lines)
        with open(output_path, "w") as f:
            f.write(md_content)
        print(f"📝 Markdown report → {output_path}")
        return md_content
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {"num_cells": self.num_cells, "dt": self.dt, "t_max": self.t_max,
                "loss_history": [float(x) for x in self.loss_history],
                "metadata": self.metadata, "traces_shape": list(self.traces.shape)}
    
    def export(self, formats: Union[str, List[str]] = "plotly",
               output_dir: str = "./results", caption: str = "") -> Dict[str, str]:
        """Multi-format batch export."""
        if isinstance(formats, str):
            formats = [formats]
        os.makedirs(output_dir, exist_ok=True)
        outputs = {}
        if "plotly" in formats:
            path = os.path.join(output_dir, "dashboard.html")
            self.to_plotly(path); outputs["plotly"] = path
        if "svg" in formats:
            path = os.path.join(output_dir, "raster.svg")
            self.to_svg(path, panel="raster"); outputs["svg"] = path
        if "markdown" in formats:
            path = os.path.join(output_dir, "report.md")
            self.to_markdown(path, caption=caption); outputs["markdown"] = path
        print(f"✅ Exported {len(outputs)} format(s) → {output_dir}/")
        return outputs
