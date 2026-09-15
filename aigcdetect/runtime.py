from __future__ import annotations
import copy
from pathlib import Path
import torch
from .config import resolve_path, validate_config


def prepare_config_paths(cfg, repo_root):
    cfg = copy.deepcopy(validate_config(cfg))
    for key, value in cfg["paths"].items():
        if value is not None:
            cfg["paths"][key] = resolve_path(value, repo_root)
    for key in ("train_root", "val_root", "test_root"):
        if cfg["data"].get(key):
            cfg["data"][key] = resolve_path(cfg["data"][key], repo_root)
    if cfg["training"].get("resume"):
        cfg["training"]["resume"] = resolve_path(cfg["training"]["resume"], repo_root)
    return cfg


def choose_device(device_str="cuda"):
    device = torch.device(device_str)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; install a matching CUDA PyTorch build or select runtime.device=cpu")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"Requested {device}, but only {torch.cuda.device_count()} visible CUDA devices")
    return device
