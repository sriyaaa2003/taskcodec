"""YAML configuration loading, merging and hashing."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(experiment: str | None = None, overrides: dict | None = None,
                config_dir: Path | None = None) -> dict[str, Any]:
    """Load base.yaml, merge `<experiment>.yaml` over it, then `overrides`."""
    d = Path(config_dir) if config_dir else CONFIG_DIR
    with open(d / "base.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if experiment:
        p = d / f"{experiment}.yaml"
        if not p.exists():
            raise FileNotFoundError(f"experiment config not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]
