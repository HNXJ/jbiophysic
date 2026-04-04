# codes/scripts/run_full_experiment.py
import sys
import os

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from build_manuscript import main

if __name__ == "__main__":
    print("🧪 Running Full Scientific Experiment Pipeline...")
    main()
