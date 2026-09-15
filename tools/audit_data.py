#!/usr/bin/env python
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aigcdetect.cli import parser, config_for
from aigcdetect.data.folder_dataset import test_directories
from aigcdetect.data.preparation import audit_splits
from aigcdetect.io_utils import atomic_json


def main(argv=None):
    p = parser("Audit image counts and train/val/test overlap")
    p.add_argument("--hash", action="store_true", help="Detect byte-identical copies, not just identical paths")
    p.add_argument("--decode", action="store_true", help="Check every encoded image for corruption")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--output")
    a = p.parse_args(argv)
    c = config_for(a)
    roots = {"train": c["data"]["train_root"], "val": c["data"]["val_root"]}
    if not a.train_only:
        roots.update(dict(test_directories(c)))
    report = audit_splits(roots, a.hash, a.decode)
    if a.output:
        atomic_json(a.output, report)
    print(json.dumps(report, indent=2))
    if report["leaks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
