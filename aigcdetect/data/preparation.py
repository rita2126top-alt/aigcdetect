"""Byte-preserving Arrow export, content-grouped holdout and leakage audits."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import os
import random
import shutil
from pathlib import Path
from PIL import Image
from ..io_utils import atomic_json, sha256_file
from .folder_dataset import BinaryImageFolder, path_label, EXTS


def arrow_rows(leaf):
    leaf = Path(leaf)
    if (leaf / "state.json").is_file():
        from datasets import load_from_disk, Image as HFImage
        ds = load_from_disk(str(leaf))
        if isinstance(ds.features.get("image"), HFImage):
            ds = ds.cast_column("image", HFImage(decode=False))
        yield from ds
        return
    import pyarrow as pa
    shards = sorted(leaf.glob("*.arrow"))
    if not shards:
        raise FileNotFoundError(f"No save_to_disk bundle or Arrow shards: {leaf}")
    for shard in shards:
        with pa.memory_map(str(shard), "r") as source:
            try:
                reader = pa.ipc.open_stream(source)
                batches = reader
            except pa.ArrowInvalid:
                source.seek(0)
                reader = pa.ipc.open_file(source)
                batches = (reader.get_batch(i) for i in range(reader.num_record_batches))
            for batch in batches:
                yield from batch.to_pylist()


def image_bytes(value, leaf):
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return bytes(value["bytes"])
        path = value.get("path")
        if path:
            path = Path(path)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("External/parent image paths are not accepted; export embedded bytes")
            resolved = (Path(leaf) / path).resolve()
            if not resolved.is_relative_to(Path(leaf).resolve()):
                raise ValueError("Image path escapes its local bundle")
            return resolved.read_bytes()
    if isinstance(value, (bytes, bytearray, memoryview, list)):
        return bytes(value)
    raise ValueError("Expected embedded image bytes; refusing silent JPEG re-encoding")


def export_arrow(snapshot, output, split="test", generator=None):
    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    source = snapshot / "data" / split
    if not source.is_dir():
        raise FileNotFoundError(f"Missing physical split {source}; validation is not an independent test split")
    leaves = [x for x in sorted(source.iterdir()) if x.is_dir() and (generator is None or x.name == generator)]
    if not leaves:
        raise ValueError(f"No matching generator {generator!r} under {source}")
    if output.is_relative_to(snapshot):
        raise ValueError("Export outside the downloaded snapshot")
    counts = {}
    for leaf in leaves:
        dest = output / leaf.name
        dest.mkdir(parents=True, exist_ok=True)
        identity = {"source": str(leaf), "split": split,
                    "state": json.loads((leaf / "state.json").read_text()) if (leaf / "state.json").exists() else None,
                    "revision": json.loads((snapshot / "DOWNLOAD_MANIFEST.json").read_text()).get("revision") if (snapshot / "DOWNLOAD_MANIFEST.json").exists() else None,
                    "shards": {x.name: x.stat().st_size for x in sorted(leaf.glob("*.arrow"))}}
        identity_path = dest / "export_source.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise FileExistsError(f"Export source/revision differs in {dest}; use a new output directory")
        atomic_json(identity_path, identity)
        expected_files = set()
        manifest = dest / "export_manifest.csv"
        temporary = manifest.with_suffix(".csv.tmp")
        n = 0
        with temporary.open("w", newline="", encoding="utf-8") as file:
            fields = ["row", "source_path", "label", "generator", "sha256", "output"]
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for i, row in enumerate(arrow_rows(leaf)):
                source_path = str(row.get("image_path", ""))
                inferred = path_label(Path(source_path.replace("\\", "/")).parts[:-1])
                label = row.get("label", inferred)
                if label not in (0, 1) or (inferred is not None and int(label) != inferred):
                    raise ValueError(f"Invalid/conflicting label in {leaf.name} row {i}: {label}")
                data = image_bytes(row.get("image"), leaf)
                if not data:
                    raise ValueError(f"Empty image bytes at {leaf.name}:{i}")
                expected_md5 = row.get("md5")
                if expected_md5 and hashlib.md5(data).hexdigest() != str(expected_md5).lower():
                    raise ValueError(f"Source MD5 mismatch at {leaf.name}:{i}")
                with Image.open(io.BytesIO(data)) as image:
                    extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tiff"}.get(image.format)
                    image.verify()
                if not extension:
                    raise ValueError(f"Unsupported original image encoding at {leaf.name}:{i}")
                digest = hashlib.sha256(data).hexdigest()
                relative = Path("0_real" if label == 0 else "1_fake") / f"{i:08d}_{digest[:12]}{extension}"
                expected_files.add(str(relative))
                target = dest / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if sha256_file(target) != digest:
                        raise FileExistsError(f"Existing exported file has different bytes: {target}")
                else:
                    tmp = target.with_suffix(target.suffix + ".partial")
                    tmp.write_bytes(data)
                    os.replace(tmp, target)
                writer.writerow(dict(row=i, source_path=source_path, label=int(label), generator=leaf.name,
                                     sha256=digest, output=str(relative)))
                n += 1
        if n == 0:
            raise ValueError(f"Empty generator bundle: {leaf}")
        actual_files = {str(x.relative_to(dest)) for x in dest.rglob("*") if x.is_file() and x.suffix.lower() in EXTS}
        if actual_files != expected_files:
            raise FileExistsError(f"Unexpected stale image files in {dest}; use a clean export directory")
        os.replace(temporary, manifest)
        counts[leaf.name] = n
    atomic_json(output / f"export_{split}_summary.json", {"snapshot": str(snapshot), "split": split,
                "counts": counts, "encoding": "original bytes, no recompression"})
    return counts


def split_holdout(source, train_root, val_root, val_fraction=0.1, seed=0, mode="symlink"):
    source, train_root, val_root = (Path(x).resolve() for x in (source, train_root, val_root))
    if not 0 < val_fraction < 1 or mode not in ("symlink", "copy"):
        raise ValueError("Require 0 < val_fraction < 1 and mode=symlink|copy")
    roots = (source, train_root, val_root)
    if any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(roots) for b in roots[i+1:]):
        raise ValueError("Source/train/validation directories must be separate non-nested roots")
    ds = BinaryImageFolder(str(source))
    groups, labels = {}, {}
    for path, label in ds.samples:
        digest = sha256_file(path)
        if digest in labels and labels[digest] != label:
            raise ValueError("Identical bytes have conflicting real/fake labels")
        labels[digest] = label
        groups.setdefault(digest, []).append(path)
    assignment = {}
    rng = random.Random(seed)
    for label in (0, 1):
        keys = sorted(k for k, v in labels.items() if v == label)
        if len(keys) < 2:
            raise ValueError(f"Need at least two distinct images for class {label}")
        rng.shuffle(keys)
        n_val = max(1, min(len(keys)-1, round(len(keys)*val_fraction)))
        assignment.update({key: ("val" if i < n_val else "train") for i, key in enumerate(keys)})
    plan = {"source": str(source), "seed": seed, "val_fraction": val_fraction, "mode": mode,
            "groups_sha256": hashlib.sha256(json.dumps(assignment, sort_keys=True).encode()).hexdigest()}
    for directory in (train_root, val_root):
        metadata = directory / "split_config.json"
        if directory.exists() and any(directory.iterdir()):
            if not metadata.exists() or json.loads(metadata.read_text()) != plan:
                raise FileExistsError(f"Existing nonmatching split in {directory}; choose a new directory")
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(metadata, plan)
    counts = {"train": 0, "val": 0}
    for digest, sources in sorted(groups.items()):
        part = assignment[digest]
        base = train_root if part == "train" else val_root
        for path in sources:
            target = base / Path(path).relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                if not target.exists() or sha256_file(target) != digest:
                    raise FileExistsError(f"Existing split target differs or is broken: {target}")
            elif mode == "symlink":
                target.symlink_to(Path(path).resolve())
            else:
                shutil.copy2(path, target)
            counts[part] += 1
    for directory, part in ((train_root, "train"), (val_root, "val")):
        atomic_json(directory / "split_summary.json", {"counts": counts, "partition": part,
                    "content_grouped": True, "source_is_training_data": "caller must verify source provenance"})
    return counts


def audit_splits(roots, content_hash=False, decode=False):
    indexes, counts = {}, {}
    for name, root in roots.items():
        ds = BinaryImageFolder(root)
        index = {}
        for path, label in ds.samples:
            if decode:
                with Image.open(path) as image:
                    image.verify()
            key = sha256_file(path) if content_hash else str(Path(path).resolve())
            index.setdefault(key, []).append(path)
        indexes[name] = index
        counts[name] = {"images": len(ds), "unique": len(index),
                        "real": sum(y == 0 for _, y in ds.samples), "fake": sum(y == 1 for _, y in ds.samples)}
    leaks = []
    for i, first in enumerate(roots):
        for second in list(roots)[i+1:]:
            if first not in ("train", "val") and second not in ("train", "val"):
                continue  # shared real test examples across domains are not train/test leakage
            shared = set(indexes[first]).intersection(indexes[second])
            if shared:
                leaks.append({"first": first, "second": second, "count": len(shared),
                              "example": [indexes[first][k][0] for k in sorted(shared)[:5]]})
    return {"counts": counts, "leaks": leaks, "check": "sha256" if content_hash else "resolved_path",
            "status": "failed" if leaks else "passed"}
