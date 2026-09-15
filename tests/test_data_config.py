import hashlib
import io
import json
import pickle
from pathlib import Path
import numpy as np
import pytest
import yaml
from PIL import Image
from datasets import Dataset, Features, Value, Image as HFImage
from aigcdetect.data.preparation import export_arrow, split_holdout, audit_splits
from aigcdetect.data.folder_dataset import BinaryImageFolder, build_transform, make_loader
from aigcdetect.config import load_config, apply_overrides
from aigcdetect.engine.metrics import binary_metrics


def png(n):
    im = Image.fromarray(np.random.default_rng(n).integers(0, 256, (28, 28, 3), dtype=np.uint8))
    f = io.BytesIO(); im.save(f, format="PNG"); return f.getvalue()


@pytest.mark.parametrize("schema", ["binary", "hf_image", "ipc"])
def test_arrow_original_bytes_and_holdout(tmp_path, schema):
    leaf = tmp_path / "snapshot/data/train/gen"
    leaf.mkdir(parents=True)
    payloads = [png(n) for n in range(8)]
    labels = [0] * 4 + [1] * 4
    data = {"image": payloads, "label": labels,
            "image_path": [f"gen/{'nature' if y == 0 else 'ai'}/{i}.png" for i, y in enumerate(labels)],
            "md5": [hashlib.md5(b).hexdigest() for b in payloads]}
    if schema == "ipc":
        import pyarrow as pa
        table = pa.table(data)
        with pa.OSFile(str(leaf / "data.arrow"), "wb") as stream:
            with pa.ipc.new_stream(stream, table.schema) as writer:
                writer.write_table(table)
    elif schema == "hf_image":
        data["image"] = [{"bytes": b, "path": None} for b in payloads]
        Dataset.from_dict(data, features=Features({"image": HFImage(), "label": Value("int64"), "image_path": Value("string"), "md5": Value("string")})).save_to_disk(str(leaf))
    else:
        Dataset.from_dict(data).save_to_disk(str(leaf))
    out = tmp_path / "export"
    assert export_arrow(tmp_path / "snapshot", out, "train") == {"gen": 8}
    assert set(p.read_bytes() for p in out.rglob("*.png")) == set(payloads)
    assert export_arrow(tmp_path / "snapshot", out, "train") == {"gen": 8}
    counts = split_holdout(out / "gen", tmp_path / "train", tmp_path / "val", seed=4)
    assert counts == {"train": 6, "val": 2}
    assert split_holdout(out / "gen", tmp_path / "train", tmp_path / "val", seed=4) == counts
    assert audit_splits({"train": tmp_path / "train", "val": tmp_path / "val"}, True, True)["status"] == "passed"
    assert audit_splits({"train": tmp_path / "train", "val": tmp_path / "train"}, True)["status"] == "failed"
    with pytest.raises(FileExistsError):
        split_holdout(out / "gen", tmp_path / "train", tmp_path / "val", seed=5)


def test_loader_paths_and_errors(tmp_path):
    root = tmp_path / "real" / "dataset"
    fake = root / "nested/1_fake"; fake.mkdir(parents=True)
    (fake / "one.png").write_bytes(png(0))
    ds = BinaryImageFolder(root)
    assert ds.samples[0][1] == 1
    loader = make_loader(root, batch_size=48, workers=0, load_size=28, image_size=28, train=True)
    assert len(loader) == 1 and next(iter(loader))[1].item() == 1
    pickle.loads(pickle.dumps(build_transform(True, 28, 28)))
    (fake / "one.png").write_bytes(b"bad")
    with pytest.raises(Exception):
        ds[0]


def test_config_and_metrics(tmp_path):
    p = tmp_path / "base.yaml"; p.write_text("section:\n  value: 1\n")
    q = tmp_path / "child.yaml"; q.write_text("extends: base.yaml\nsection:\n  other: 2\n")
    c = load_config(q)
    assert c == {"section": {"value": 1, "other": 2}}
    assert apply_overrides(c, ["section.value=3"])["section"]["value"] == 3
    with pytest.raises((KeyError, ValueError)):
        apply_overrides(c, ["section.typo=3"])
    p.write_text("extends: child.yaml")
    with pytest.raises(ValueError):
        load_config(q)
    assert binary_metrics([0, 1], [.1, .9])["ap"] == 1
    assert binary_metrics([0], [.3])["auc"] is None
    json.dumps(binary_metrics([0], [.3]), allow_nan=False)
    for y, s in (([], []), ([0], [float("nan")]), ([2], [.5])):
        with pytest.raises(ValueError):
            binary_metrics(y, s)
