import os
from pathlib import Path

def load_jbiophys_config(config_path=None):
    """
    Standard loader for JBiophys configuration.
    Looks for config.md in the repository root.
    """
    if config_path is None:
        # Try repository root relative to this loader file
        config_path = Path(__file__).resolve().parent.parent / "config.md"
        if not config_path.exists():
            # Fallback to current working directory
            config_path = Path("config.md")

    config = {}
    if not config_path.exists():
        # print(f"⚠️ Warning: Config not found at {config_path}. Using defaults.")
        return config

    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, val = line.split('=', 1)
                config[key.strip()] = val.strip().strip('"').strip("'")
    return config

# Global config instance
CFG = load_jbiophys_config()
