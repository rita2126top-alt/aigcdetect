"""Check unchanged original source against its recorded SHA256 manifest."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "provenance/upstream.json").read_text())
    checked, errors = 0, []
    for relative, expected in manifest["sha256"].items():
        if "__pycache__" in Path(relative).parts or relative.endswith(".pyc"):
            continue  # interpreter caches are not source code
        for base in (ROOT / "PPM_CLIP", ROOT):
            if base == ROOT and relative == ".gitignore":
                continue  # project ignore rules were added at initial import
            path = base / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(str(path))
            checked += 1
    print(json.dumps({"upstream_commit": manifest["commit"], "checked_files": checked,
                      "status": "failed" if errors else "passed", "mismatches": errors}, indent=2))
    if errors:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
