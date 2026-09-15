#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from aigcdetect.config import load_config, apply_overrides
from aigcdetect.runtime import prepare_config_paths, choose_device
from aigcdetect.data import build_loaders
from aigcdetect.model import ClassAnchoredDynamicPPMCLIP
from aigcdetect.engine.train import seed_all, train
p=argparse.ArgumentParser(); p.add_argument("--config", default="configs/default.yaml"); p.add_argument("--set", action="append", default=[])
a=p.parse_args(); repo=Path(__file__).resolve().parents[1]
cfg=prepare_config_paths(apply_overrides(load_config(a.config), a.set), repo); seed_all(int(cfg["training"].get("seed", 0)))
device=choose_device(cfg["runtime"].get("device","cuda")); train_loader,val_loader,_=build_loaders(cfg)
model=ClassAnchoredDynamicPPMCLIP(cfg).to(device)
print(json.dumps({"trainable":sum(p.numel() for p in model.parameters() if p.requires_grad)}, indent=2))
print("best checkpoint:", train(cfg,model,train_loader,val_loader,device))
