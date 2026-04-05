# cli/gravia_write.py
import yaml
import json
from manuscript.sections import generate_all_sections

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

# configs/experiment.yaml
experiment:
  name: omission_v1_pfc
  areas: [v1, v2, v4, mt, teo, fst, fef, pfc]
  training: {steps: 2000, learning_rate: 0.001}
  simulation: {steps: 3000, dt: 0.1, omission_time: 2000}
  modulation: {da_baseline: 0.05, da_omission: 0.2, ach_baseline: 0.1, ach_omission: 0.0}
  stdp: {enabled: true, delta: 0.01}
