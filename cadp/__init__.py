"""Additive CADP extension. Original PPM-CLIP source stays untouched."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
_ROOT = Path(__file__).resolve().parents[1]
_UPSTREAM = _ROOT / "PPM_CLIP"
# Editable/source installation intentionally loads the pinned local submodule.
if not (_UPSTREAM / "clip" / "model.py").is_file():
    raise ImportError("Missing PPM_CLIP sources. Run git submodule update --init --recursive.")
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))
__version__ = "0.2.0"
