#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
p=argparse.ArgumentParser();p.add_argument("--snapshot",required=True);p.add_argument("--output",required=True);p.add_argument("--split",default="test");p.add_argument("--generator",default=None,help="Optional exact generator leaf name");a=p.parse_args()
from datasets import load_from_disk
src=Path(a.snapshot)/"data"/a.split; out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
for leaf in sorted(src.iterdir()):
    if not leaf.is_dir(): continue
    if a.generator and leaf.name != a.generator: continue
    ds=load_from_disk(str(leaf)); print(leaf.name,len(ds))
    for i,row in enumerate(ds):
        label=int(row["label"]); cls="0_real" if label==0 else "1_fake"; d=out/leaf.name/cls; d.mkdir(parents=True,exist_ok=True)
        image=row["image"]
        if hasattr(image,"save"): image.convert("RGB").save(d/f"{i:08d}.jpg",quality=95)
