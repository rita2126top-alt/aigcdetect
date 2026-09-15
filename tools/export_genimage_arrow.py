#!/usr/bin/env python
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aigcdetect.data.preparation import export_arrow


def main(argv=None):
    p = argparse.ArgumentParser(description="Export GenImage original bytes; never re-encode images")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--generator")
    a = p.parse_args(argv)
    print(json.dumps(export_arrow(a.snapshot, a.output, a.split, a.generator), indent=2))


if __name__ == "__main__":
    main()
