#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
import torch
from PIL import Image
from aigcdetect.config import load_config
from aigcdetect.runtime import prepare_config_paths, choose_device
from aigcdetect.data.folder_dataset import build_transform
from aigcdetect.model import ClassAnchoredDynamicPPMCLIP
from aigcdetect.engine.train import load_checkpoint
p=argparse.ArgumentParser(); p.add_argument("image"); p.add_argument("--config",default="configs/default.yaml"); p.add_argument("--checkpoint",required=True)
a=p.parse_args(); repo=Path(__file__).resolve().parents[1]; cfg=prepare_config_paths(load_config(a.config),repo)
device=choose_device(cfg["runtime"].get("device","cuda")); model=ClassAnchoredDynamicPPMCLIP(cfg).to(device); load_checkpoint(a.checkpoint,model,map_location=device); model.eval()
tf=build_transform(False,cfg["data"].get("load_size",256),cfg["data"].get("image_size",224)); x=tf(Image.open(a.image).convert("RGB"))[None].to(device)
with torch.no_grad(): out=model(x,mode="test")
p=out["probabilities"][0]; print({"real":float(p[0]),"fake":float(p[1]),"prediction":"fake" if p[1]>=.5 else "real","private_length":int(out["selected_private_length"][0])})
