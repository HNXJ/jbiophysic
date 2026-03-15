import streamlit as st
import os
import glob
from pathlib import Path

# --- Configuration ---
FIGURES_ROOT = "/Users/hamednejat/workspace/Computational/mscz/figures"

st.set_page_config(layout="wide", page_title="jbiophys Research Explorer")

def get_subfolders(path):
    return [f.name for f in Path(path).iterdir() if f.is_dir()]

def main():
    st.title("🧠 jbiophys Research Explorer")
    st.markdown("---")

    if not os.path.exists(FIGURES_ROOT):
        st.error(f"Figures root not found: {FIGURES_ROOT}")
        return

    # 1. Sidebar: Select Run
    runs = get_subfolders(FIGURES_ROOT)
    if not runs:
        st.warning("No runs found in figures directory.")
        return

    selected_run = st.sidebar.selectbox("📂 Select Optimization Run", sorted(runs, reverse=True))
    run_path = Path(FIGURES_ROOT) / selected_run

    # 2. Main Page: History or Snapshots
    st.header(f"Run: {selected_run}")
    
    # Check for Optimization History
    history_file = run_path / "optimization_history.html"
    if history_file.exists():
        with st.expander("📈 View Optimization Trajectory", expanded=False):
            with open(history_file, 'r', encoding='utf-8') as f:
                st.components.v1.html(f.read(), height=1200, scrolling=True)

    # 3. Snapshot Navigation
    snapshots = get_subfolders(run_path)
    if snapshots:
        st.subheader("🖼️ Parameter Snapshots")
        selected_snapshot = st.select_slider("Evolution Timeline", options=sorted(snapshots, key=lambda x: int(x.split('-')[1])))
        
        snapshot_path = run_path / selected_snapshot
        
        # Display HTML figures in columns or tabs
        col1, col2 = st.columns(2)
        
        dynamics_file = snapshot_path / "dynamics.html"
        arch_file = snapshot_path / "architecture.html"
        suite_file = snapshot_path / "biophysical_suite.html"

        if dynamics_file.exists():
            with col1:
                st.caption("Interactive Dynamics (Raster/Spectrogram)")
                with open(dynamics_file, 'r', encoding='utf-8') as f:
                    st.components.v1.html(f.read(), height=1000, scrolling=True)

        if arch_file.exists():
            with col2:
                st.caption("Interactive 3D Architecture")
                with open(arch_file, 'r', encoding='utf-8') as f:
                    st.components.v1.html(f.read(), height=1000, scrolling=True)

        if suite_file.exists():
            st.markdown("---")
            st.caption("Detailed Biophysical Analysis")
            with open(suite_file, 'r', encoding='utf-8') as f:
                st.components.v1.html(f.read(), height=1000, scrolling=True)

if __name__ == "__main__":
    main()
