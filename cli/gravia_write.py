# cli/gravia_write.py
import sys
import yaml
import json
import os

def get_manuscript_paths():
    """Returns local output paths decoupled from original git tracking."""
    return {"results_trace": "output/simulation_trace.json", "results_md": "output/manuscript/sections/results.md"}

# Use try/except for local module if manuscript is placed under output
try:
    sys.path.append('output')
    from manuscript.sections import generate_all_sections
except ImportError:
    generate_all_sections = None

def load_context():
    """Load config, gamma, and simulation results."""
    with open("configs/experiment.yaml") as f:
        config = yaml.safe_load(f)
    try:
        with open("gamma/trace.json") as f:
            gamma = json.load(f)
    except: gamma = {}
    try:
        with open("pipeline/results.json") as f:
            results = json.load(f)
    except: results = {}
    return config, gamma, results

def gravia_write():
    """CLI Agent: Generate manuscript sections from data."""
    config, gamma, results = load_context()
    sections = generate_all_sections(config, gamma, results)
    for name, text in sections.items():
        with open(f"manuscript/sections/{name}.md", "w") as f:
            f.write(text)
    print("✅ Gravia-Agent: Manuscript sections synchronized.")

if __name__ == "__main__":
    gravia_write()


