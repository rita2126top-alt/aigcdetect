#!/usr/bin/env python
"""Network access is confined to this explicit download entry point."""
from __future__ import annotations
import fnmatch
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aigcdetect.cli import parser, config_for
from aigcdetect.io_utils import sha256_file, atomic_json

CLIP_SHA = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
CLIP_URL = f"https://openaipublic.azureedge.net/clip/models/{CLIP_SHA}/ViT-L-14.pt"
GENERATORS = ["ADM", "BigGAN", "glide", "Midjourney", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "VQDM", "wukong"]


def download_clip(target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) == CLIP_SHA:
            print(f"Already verified: {target}")
            return
        raise FileExistsError(f"Existing CLIP file has a different SHA256: {target}; move it explicitly before retrying")
    partial = target.with_suffix(target.suffix + ".partial")
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with urllib.request.urlopen(urllib.request.Request(CLIP_URL, headers=headers), timeout=120) as response:
                mode = "ab" if offset and response.status == 206 else "wb"
                with partial.open(mode) as f:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        f.write(chunk)
            if sha256_file(partial) != CLIP_SHA:
                partial.unlink(missing_ok=True)
                raise ValueError("CLIP SHA256 mismatch; incomplete/corrupt transfer")
            os.replace(partial, target)
            print(f"Downloaded and SHA256 verified: {target}")
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))


def download_genimage(c, mode="paper", dry_run=False):
    from huggingface_hub import HfApi, snapshot_download
    dl, paths = c["downloads"], c["paths"]
    api = HfApi()
    info = api.repo_info(dl["genimage_repo"], repo_type="dataset", revision=dl["genimage_revision"])
    revision = info.sha
    patterns = [f"data/test/{g}/*" for g in GENERATORS]
    if mode == "paper":
        patterns.append(f"data/train/{dl['train_generator']}/*")
    elif mode == "all-train":
        patterns += [f"data/train/{g}/*" for g in GENERATORS]
    files = api.list_repo_files(dl["genimage_repo"], repo_type="dataset", revision=revision)
    for pattern in patterns:
        if not any(fnmatch.fnmatch(x, pattern) and x.endswith(".arrow") for x in files):
            raise ValueError(f"No Arrow files for {pattern} at {revision}; refusing an empty download")
    record = {"repository": dl["genimage_repo"], "revision": revision, "patterns": patterns, "mode": mode}
    print(json.dumps(record, indent=2))
    if dry_run:
        return
    snapshot_download(dl["genimage_repo"], repo_type="dataset", revision=revision,
                      allow_patterns=patterns + ["README.md", "manifest.json"],
                      local_dir=paths["genimage_snapshot"], max_workers=int(dl.get("max_workers", 4)))
    atomic_json(Path(paths["genimage_snapshot"]) / "DOWNLOAD_MANIFEST.json", record)


def main(argv=None):
    p = parser("Download model/dataset into YAML-defined local directories")
    p.add_argument("--kind", choices=["clip", "genimage"], required=True)
    p.add_argument("--mode", choices=["test", "paper", "all-train"], default="paper")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    c = config_for(a)
    if a.kind == "clip":
        if c["model"]["backbone"] != "ViT-L/14":
            raise ValueError("Bundled downloader supplies ViT-L/14 only; set a local checkpoint for other backbones")
        if a.dry_run:
            print(json.dumps({"url": CLIP_URL, "sha256": CLIP_SHA, "output": c["paths"]["clip_model"]}))
        else:
            download_clip(c["paths"]["clip_model"])
    else:
        download_genimage(c, a.mode, a.dry_run)


if __name__ == "__main__":
    main()
