#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from aigcdetect.config import load_config, apply_overrides
from aigcdetect.runtime import prepare_config_paths, choose_device
from aigcdetect.data.folder_dataset import make_loader
from aigcdetect.model import ClassAnchoredDynamicPPMCLIP
from aigcdetect.engine.train import load_checkpoint
from aigcdetect.engine.evaluate import evaluate_loader, save_eval
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--checkpoint',required=True); p.add_argument('--set',action='append',default=[])
a=p.parse_args(); cfg=prepare_config_paths(apply_overrides(load_config(a.config),a.set),REPO_ROOT); device=choose_device(cfg['runtime'].get('device','cuda'))
model=ClassAnchoredDynamicPPMCLIP(cfg).to(device); load_checkpoint(a.checkpoint,model,map_location=device)
d=cfg['data']; t=cfg['training']; common=dict(batch_size=int(t['batch_size']),workers=int(d.get('num_workers',8)),load_size=int(d.get('load_size',256)),image_size=int(d.get('image_size',224)),train=False)
conditions=[('clean',{}),('jpeg95',{'jpeg_quality':95}),('jpeg75',{'jpeg_quality':75}),('jpeg50',{'jpeg_quality':50}),('blur1',{'blur_radius':1.0}),('blur2',{'blur_radius':2.0})]
allm={}
for subset in d.get('test_sets',[]):
    root=str(Path(d['test_root'])/subset)
    for cname,kw in conditions:
        loader=make_loader(root,**common,**kw); name=f'{subset}__{cname}'; m,pred=evaluate_loader(model,loader,device,name); save_eval(cfg['paths']['output_dir'],name,m,pred); allm[name]=m; print(name,m)
print(json.dumps(allm,indent=2))
