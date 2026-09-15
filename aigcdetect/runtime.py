from __future__ import annotations
import os
from pathlib import Path
import torch
from .config import resolve_path


def prepare_config_paths(cfg, repo_root):
    cfg["paths"]["ppmclip_root"] = resolve_path(cfg["paths"]["ppmclip_root"], repo_root)
    cfg["paths"]["clip_model"] = resolve_path(cfg["paths"]["clip_model"], repo_root)
    cfg["paths"]["output_dir"] = resolve_path(cfg["paths"]["output_dir"], repo_root)
    for key in ("train_root", "val_root", "test_root"):
        cfg["data"][key] = resolve_path(cfg["data"][key], repo_root)
    return cfg


def choose_device(device_str="cuda"):
    if str(device_str).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(device_str)
