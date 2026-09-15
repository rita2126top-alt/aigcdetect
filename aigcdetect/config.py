from __future__ import annotations
import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable
import yaml


def load_config(path, _seen=None):
    path = Path(path).expanduser().resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f"Circular YAML extends: {path}")
    seen.add(path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    base = cfg.pop("extends", None)
    if base:
        parent = load_config(path.parent / base, seen)
        def merge(a, b):
            for key, value in b.items():
                if isinstance(value, dict) and isinstance(a.get(key), dict):
                    merge(a[key], value)
                else:
                    a[key] = value
            return a
        cfg = merge(parent, cfg)
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str] | None):
    out = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value: {item}")
        key, raw = item.split("=", 1)
        parts = key.split(".")
        if not all(parts):
            raise ValueError(f"Invalid config key: {key}")
        cur = out
        for part in parts[:-1]:
            if not isinstance(cur.get(part), dict):
                raise ValueError(f"Unknown config section: {key}")
            cur = cur[part]
        if parts[-1] not in cur:
            raise ValueError(f"Unknown config key: {key}; add it to YAML explicitly first")
        cur[parts[-1]] = yaml.safe_load(raw)
    return out


def validate_config(cfg):
    for section in ("paths", "runtime", "model", "data", "training", "loss"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"Missing config section: {section}")
    t, d, p = cfg["training"], cfg["data"], cfg["model"]["prompt"]
    for value in (t["epochs"], t["batch_size"], t.get("accumulate_steps", 1), p["test_samples"], p["train_samples"]):
        if not isinstance(value, int) or value < 1:
            raise ValueError("Epochs, batch sizes, accumulation and sample counts must be positive integers")
    if int(d.get("num_workers", 0)) < 0 or int(d["load_size"]) < int(d["image_size"]):
        raise ValueError("Invalid worker count or image sizes")
    if t.get("selection_metric", "ap") not in ("ap", "auc", "acc", "balanced_acc", "f1"):
        raise ValueError("Unsupported validation selection metric")
    return cfg


def resolve_path(path, repo_root):
    p = Path(os.path.expandvars(str(path))).expanduser()
    return str((p if p.is_absolute() else Path(repo_root) / p).resolve())
