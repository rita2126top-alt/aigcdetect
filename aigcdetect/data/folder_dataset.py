from __future__ import annotations
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
LABELS = {"0_real": 0, "real": 0, "nature": 0, "1_fake": 1, "fake": 1, "ai": 1}


def path_label(parts):
    # Only relative parent components; never infer a label from /home/real/... .
    for component in reversed(tuple(parts)):
        if component.lower() in LABELS:
            return LABELS[component.lower()]
    return None


class BinaryImageFolder(Dataset):
    def __init__(self, root: str, transform=None, classes=None):
        root = Path(root).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")
        self.root, self.transform, self.samples = str(root), transform, []
        classes = set(classes or [])
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in EXTS:
                continue
            relative = p.relative_to(root)
            if classes and not (classes & set(relative.parts[:-1])):
                continue
            label = path_label(relative.parts[:-1])
            if label is not None:
                self.samples.append((str(p), label))
        if not self.samples:
            raise ValueError(f"No labeled images in {root}; expected 0_real/1_fake or nature/ai")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as im:
                image = im.convert("RGB")
            return self.transform(image) if self.transform else image, label, path
        except Exception as exc:
            raise RuntimeError(f"Cannot decode image {path}: {exc}") from exc


@dataclass
class ImagePreparation:
    """Pickle-safe preprocessing for Linux fork and Windows/spawn workers."""
    load_size: int
    jpeg_quality: int | None = None
    blur_radius: float | None = None

    def __call__(self, image):
        if min(image.size) < self.load_size:
            image = transforms.Resize((self.load_size, self.load_size), interpolation=InterpolationMode.BILINEAR)(image)
        if self.jpeg_quality is not None:
            with io.BytesIO() as buffer:
                image.save(buffer, format="JPEG", quality=self.jpeg_quality)
                buffer.seek(0)
                with Image.open(buffer) as decoded:
                    image = decoded.convert("RGB")
        if self.blur_radius is not None:
            image = image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        return image


def build_transform(train: bool, load_size=256, image_size=224, jpeg_quality=None, blur_radius=None):
    if load_size < image_size or image_size < 1:
        raise ValueError("Require load_size >= image_size > 0")
    if jpeg_quality is not None and not 1 <= int(jpeg_quality) <= 100:
        raise ValueError("JPEG quality must be in [1, 100]")
    if blur_radius is not None and float(blur_radius) < 0:
        raise ValueError("Blur radius must be nonnegative")
    ops = [ImagePreparation(int(load_size), jpeg_quality, blur_radius)]
    ops += [transforms.RandomCrop(image_size), transforms.RandomHorizontalFlip()] if train else [transforms.CenterCrop(image_size)]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(root, batch_size, workers, train, load_size, image_size,
                jpeg_quality=None, blur_radius=None, classes=None, seed=0):
    ds = BinaryImageFolder(root, build_transform(train, load_size, image_size, jpeg_quality, blur_radius), classes)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=False, drop_last=False,
                      worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))


def assert_disjoint(first, second):
    a = {str(Path(p).resolve()) for p, _ in first.samples}
    overlap = a.intersection(str(Path(p).resolve()) for p, _ in second.samples)
    if overlap:
        raise ValueError(f"Train/validation path leakage: {next(iter(overlap))}; use separate splits")


def test_directories(cfg):
    d = cfg["data"]
    sets = d.get("test_sets", [])
    if not sets:
        raise ValueError("data.test_sets is empty")
    mapping = sets if isinstance(sets, dict) else {name: name for name in sets}
    for name, rel in mapping.items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Unsafe test-set output name: {name}")
        yield name, str(Path(d["test_root"]) / rel)


def build_loaders(cfg: Dict, stage="all", include_val=False):
    if stage not in ("train", "eval", "all"):
        raise ValueError("Unknown loader stage")
    d, t = cfg["data"], cfg["training"]
    common = dict(workers=int(d.get("num_workers", 8)), load_size=int(d.get("load_size", 256)),
                  image_size=int(d.get("image_size", 224)), seed=int(t.get("seed", 0)))
    train = val = None
    tests = {}
    if stage in ("train", "all"):
        train = make_loader(d["train_root"], int(t["batch_size"]), train=True,
                            classes=d.get("train_classes"), **common)
    if stage in ("train", "all") or include_val:
        val = make_loader(d["val_root"], int(d.get("eval_batch_size", t["batch_size"])), train=False,
                          classes=d.get("train_classes"), **common)
        if set(y for _, y in val.dataset.samples) != {0, 1}:
            raise ValueError("Validation must contain both real and fake images")
    if train is not None:
        if set(y for _, y in train.dataset.samples) != {0, 1}:
            raise ValueError("Training must contain both real and fake images")
        assert_disjoint(train.dataset, val.dataset)
    if stage in ("eval", "all"):
        for name, root in test_directories(cfg):
            tests[name] = make_loader(root, int(d.get("eval_batch_size", t["batch_size"])), train=False, **common)
    return train, val, tests
