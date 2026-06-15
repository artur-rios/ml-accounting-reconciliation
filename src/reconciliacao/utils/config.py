from pathlib import Path
import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)
