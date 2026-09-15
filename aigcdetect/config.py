from __future__ import annotations
import copy
from pathlib import Path
from typing import Any, Dict, Iterable
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _parse_value(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except Exception:
        return value


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str] | None) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw = item.split("=", 1)
        cur = out
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = _parse_value(raw)
    return out


def resolve_path(path: str, repo_root: str | Path) -> str:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path(repo_root).expanduser().resolve() / p
    return str(p.resolve())
