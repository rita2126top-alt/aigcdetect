#!/usr/bin/env python
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import argparse, os, sys, torch
from aigcdetect.config import load_config
from aigcdetect.runtime import prepare_config_paths
p=argparse.ArgumentParser();p.add_argument("--config",default="configs/default.yaml");a=p.parse_args();repo=Path(__file__).resolve().parents[1];cfg=prepare_config_paths(load_config(a.config),repo)
checks={"python":sys.version.split()[0],"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"ppmclip_root":os.path.isdir(cfg["paths"]["ppmclip_root"]),"clip_model":os.path.isfile(cfg["paths"]["clip_model"]),"train_root":os.path.isdir(cfg["data"]["train_root"]),"val_root":os.path.isdir(cfg["data"]["val_root"]),"test_root":os.path.isdir(cfg["data"]["test_root"])}
for k,v in checks.items():print(f"{k:20s}: {v}")
