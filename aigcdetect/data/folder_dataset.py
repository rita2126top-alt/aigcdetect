from __future__ import annotations
import io
import os
from pathlib import Path
from typing import Dict
from PIL import Image, ImageFile, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = True
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

class BinaryImageFolder(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = str(root); self.transform = transform; self.samples = []
        for p in sorted(Path(root).rglob("*")):
            if not p.is_file() or p.suffix.lower() not in EXTS: continue
            parts = {x.lower() for x in p.parts}
            if "0_real" in parts or "real" in parts or "nature" in parts: label = 0
            elif "1_fake" in parts or "fake" in parts or "ai" in parts: label = 1
            else: continue
            self.samples.append((str(p), label))
        if not self.samples: raise RuntimeError(f"No labeled images found under {root}; expected folders 0_real/1_fake or real/fake.")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as im: image = im.convert("RGB")
        if self.transform: image = self.transform(image)
        return image, int(label), path

def _ensure_min_size(img, load_size):
    if min(img.size) < load_size: return transforms.Resize((load_size, load_size), interpolation=InterpolationMode.BILINEAR)(img)
    return img

def build_transform(train: bool, load_size: int = 256, image_size: int = 224, jpeg_quality: int | None = None, blur_radius: float | None = None):
    ops = [transforms.Lambda(lambda x: _ensure_min_size(x, load_size))]
    if jpeg_quality is not None:
        def jpeg(x):
            buf = io.BytesIO(); x.save(buf, format="JPEG", quality=int(jpeg_quality)); buf.seek(0); return Image.open(buf).convert("RGB")
        ops.append(transforms.Lambda(jpeg))
    if blur_radius is not None: ops.append(transforms.Lambda(lambda x: x.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))))
    ops += [transforms.RandomCrop(image_size), transforms.RandomHorizontalFlip()] if train else [transforms.CenterCrop(image_size)]
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)

def make_loader(root: str, batch_size: int, workers: int, train: bool, load_size: int, image_size: int, jpeg_quality=None, blur_radius=None):
    ds = BinaryImageFolder(root, build_transform(train, load_size, image_size, jpeg_quality, blur_radius))
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=workers, pin_memory=True, persistent_workers=workers > 0, drop_last=train)

def build_loaders(cfg: Dict):
    dcfg, tcfg = cfg["data"], cfg["training"]
    common = dict(batch_size=int(tcfg["batch_size"]), workers=int(dcfg.get("num_workers", 8)), load_size=int(dcfg.get("load_size", 256)), image_size=int(dcfg.get("image_size", 224)))
    train = make_loader(dcfg["train_root"], train=True, **common); val = make_loader(dcfg["val_root"], train=False, **common)
    tests = {name: make_loader(os.path.join(dcfg["test_root"], name), train=False, **common) for name in dcfg.get("test_sets", [])}
    return train, val, tests
