#!/usr/bin/env python
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aigcdetect.cli import parser, config_for
from aigcdetect.data.preparation import export_arrow, split_holdout


def main(argv=None):
    p = parser("Prepare local GenImage train/validation/test without test-set model selection")
    p.add_argument("--stage", choices=["all", "train", "test"], default="all")
    p.add_argument("--copy", action="store_true", help="Copy holdout images instead of symlinking")
    a = p.parse_args(argv)
    c = config_for(a)
    paths, d, dl = c["paths"], c["data"], c["downloads"]
    if a.stage in ("all", "train"):
        raw = Path(paths["genimage_export"]) / "train"
        export_arrow(paths["genimage_snapshot"], raw, "train", dl["train_generator"])
        print(json.dumps(split_holdout(raw / dl["train_generator"], d["train_root"], d["val_root"],
                        float(dl["val_fraction"]), int(dl["split_seed"]), "copy" if a.copy else "symlink")))
    if a.stage in ("all", "test"):
        print(json.dumps(export_arrow(paths["genimage_snapshot"], d["test_root"], "test")))


if __name__ == "__main__":
    main()
