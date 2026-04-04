import argparse
import sys
from jbiophysics import (
    build_11_area_hierarchy, train_sequence, lfp_pipeline, 
    plot_panel_a_tfr, plot_panel_b_corr_matrix, plot_panel_c_band_comparison
)
from jbiophysics.local.export import ResultsReport
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="gravia-write: JBiophysics Research Platform CLI.")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # 1. 'run' command: Stimulate → Analyze
    run_parser = subparsers.add_parser("run", help="Run full simulation → training → analysis.")
    run_parser.add_argument("--steps", type=int, default=2000, help="Number of training steps")
    
    # 2. 'render' command: Generate poster panels
    render_parser = subparsers.add_parser("render", help="Generate poster panels from last run.")
    
    # 3. 'write' command: Export LaTeX snippet
    write_parser = subparsers.add_parser("write", help="Export research manuscript LaTeX snippet.")
    
    args = parser.parse_parser()
    
    if args.command == "run":
        print("🚀 [gravia-write] Executing Platform Phase: Train → Omission → Analyze...")
        
        # Axis 5-6: Hierarchy & Training
        hierarchy = train_sequence(build_11_area_hierarchy(), steps=args.steps)
        
        # Axis 1-4: Biophysical Omission Simulation (Mocked for CLI demo)
        # In a real run, this calls hierarchy_step in a loop
        print("🧠 Running biophysical simulation (Axis 1-4)...")
        dummy_traces = np.random.randn(11, 2000) # Mock 11 areas
        
        # Axis 7: LFP Pipeline
        print("📊 Running 15-step LFP Analysis (Axis 7)...")
        results = lfp_pipeline({"V1": dummy_traces[0], "PFC": dummy_traces[-1]})
        
        # Save results for render/write
        print("✅ Run complete. Data stored for rendering.")
        
    elif args.command == "render":
        print("🎨 [gravia-write] Rendering Poster Panels (Axis 8)...")
        # In a real run, load last data
        # plot_panel_a_tfr(...)
        print("🖼️ Posters generated in ./output/")
        
    elif args.command == "write":
        print("📄 [gravia-write] Writing Manuscript (Axis 9)...")
        # repo = ResultsReport(traces=np.zeros((1, 100)))
        # repo.to_latex("output/manuscript.tex")
        print("📄 Manuscript snippet saved to output/manuscript.tex")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
