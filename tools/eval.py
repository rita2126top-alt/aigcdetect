#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from aigcdetect.config import load_config, apply_overrides
from aigcdetect.runtime import prepare_config_paths, choose_device
from aigcdetect.data import build_loaders
from aigcdetect.model import ClassAnchoredDynamicPPMCLIP
from aigcdetect.engine.train import load_checkpoint
from aigcdetect.engine.evaluate import evaluate_loader, save_eval
p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/default.yaml"); p.add_argument("--checkpoint",required=True); p.add_argument("--set",action="append",default=[])
a=p.parse_args(); repo=Path(__file__).resolve().parents[1]
cfg=prepare_config_paths(apply_overrides(load_config(a.config),a.set),repo); device=choose_device(cfg["runtime"].get("device","cuda"))
_,val,tests=build_loaders(cfg); model=ClassAnchoredDynamicPPMCLIP(cfg).to(device); load_checkpoint(a.checkpoint,model,map_location=device)
allm={}
for name,loader in {"val":val,**tests}.items():
 m,pred=evaluate_loader(model,loader,device,name); save_eval(cfg["paths"]["output_dir"],name,m,pred); allm[name]=m; print(name,m)
print(json.dumps(allm,indent=2))
