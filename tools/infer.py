#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from aigcdetect.cli import infer_main

if __name__ == "__main__":
    infer_main()
