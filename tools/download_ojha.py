"""Download official-source archives to YAML-defined local storage; no auto-extraction."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
from aigcdetect.cli import parser, config_for
from aigcdetect.io_utils import atomic_json, sha256_file


def main():
    p = parser(__doc__)
    p.add_argument("--subset", choices=["train", "val", "cnn_test", "diffusion", "all"], default="all")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    c = config_for(args)
    entries = c["downloads"]["ojha_archives"]
    names = list(entries) if args.subset == "all" else [args.subset]
    base = Path(c["paths"]["download_dir"]) / "ojha"
    records = []
    for name in names:
        item = entries[name]
        record = {"subset": name, "id": item["id"], "destination": str(base / item["filename"])}
        print(json.dumps(record))
        if args.dry_run:
            continue
        import gdown
        base.mkdir(parents=True, exist_ok=True)
        target = Path(record["destination"])
        done = target.with_suffix(target.suffix + ".download.json")
        if target.exists():
            if not done.exists():
                raise FileExistsError(f"Unverified existing archive: {target}. Move it aside before downloading.")
            previous = json.loads(done.read_text())
            if previous["id"] != item["id"] or previous["sha256"] != sha256_file(target):
                raise ValueError(f"Existing archive identity/checksum differs: {target}")
        else:
            partial = str(target) + ".partial"
            result = gdown.download(id=item["id"], output=partial, quiet=False, resume=True)
            if result is None:
                raise RuntimeError("Source quota/access/link failure; update the ID from the official repository")
            Path(partial).replace(target)
            atomic_json(done, {**record, "sha256": sha256_file(target)})
        records.append(json.loads(done.read_text()))
    if not args.dry_run:
        atomic_json(base / f"download_{args.subset}.json", records)
        print("Archives only. Inspect with 7z l, extract with 7z x into separate directories, then organize train/val/test roots from the guide.")

if __name__ == "__main__":
    main()
