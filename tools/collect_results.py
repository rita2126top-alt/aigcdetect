"""Aggregate already-completed runs only; absence is never treated as a zero score."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
import statistics
from aigcdetect.io_utils import atomic_json


def collect(root):
    by_variant = {}
    for path in sorted(Path(root).glob("*_seed*/eval/summary.json")):
        variant, seed = path.parents[1].name.rsplit("_seed", 1)
        data = json.loads(path.read_text())
        by_variant.setdefault(variant, []).append({"seed": int(seed), "macro": data["macro"], "source": str(path)})
    if not by_variant:
        raise FileNotFoundError(f"No completed experiment summaries below {root}")
    result = {}
    for variant, runs in by_variant.items():
        stats = {}
        for metric in runs[0]["macro"]:
            values = [r["macro"][metric] for r in runs if isinstance(r["macro"].get(metric), (int, float))]
            if values:
                stats[metric] = {"mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else None, "n": len(values)}
        result[variant] = {"runs": runs, "metrics": stats}
    return {"variants": result, "note": "Macro across domains first, then mean and sample std across completed seeds; n=1 std=null"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root")
    p.add_argument("--output")
    args = p.parse_args()
    data = collect(args.root)
    atomic_json(args.output or str(Path(args.root) / "aggregate.json"), data)
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()
