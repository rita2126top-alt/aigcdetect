"""Local CSV datasets, explicit provenance, group-aware splits and corruptions."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset
from .backbone import file_sha256

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
LABEL_DIRS = {'0_real': 0, 'real': 0, 'nature': 0, '1_fake': 1, 'fake': 1, 'ai': 1}
FIELDS = ['path', 'label', 'source', 'split', 'group', 'sha256']
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]


def write_rows(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError(f'Refusing to write an empty manifest: {p}')
    with p.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: r.get(key, '') for key in FIELDS} for r in rows)


def read_rows(path, data_root=None, require_files=True):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'Manifest missing: {p}. Run scan/split or prepare-protocol first.')
    with p.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if not {'path', 'label'}.issubset(reader.fieldnames or []):
            raise ValueError(f'{p} must have path,label columns')
        rows = list(reader)
    if not rows:
        raise ValueError(f'Manifest is empty: {p}')
    seen = set()
    for i, row in enumerate(rows, 2):
        try:
            label = int(row['label'])
            if label not in (0, 1):
                raise ValueError()
        except (ValueError, TypeError):
            raise ValueError(f'{p}:{i}: label must be 0=real or 1=fake') from None
        image = Path(row['path']).expanduser()
        if not image.is_absolute():
            if data_root is None:
                raise ValueError(f'{p}:{i}: relative image path requires data_root')
            image = Path(data_root) / image
        image = image.resolve()
        if require_files and not image.is_file():
            raise FileNotFoundError(f'{p}:{i}: missing image {image}')
        if str(image) in seen:
            raise ValueError(f'Duplicate image path inside {p}: {image}')
        seen.add(str(image))
        row.update(path=str(image), label=label, source=row.get('source') or 'unspecified',
                   split=row.get('split') or '', group=row.get('group') or '', sha256=row.get('sha256') or '')
    return rows


def scan(root, output, split, source=None, relative_to=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset folder not found: {root}')
    rows, missing = [], []
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in EXTENSIONS:
            continue
        labels = {LABEL_DIRS[part.lower()] for part in p.relative_to(root).parts[:-1] if part.lower() in LABEL_DIRS}
        if not labels and root.name.lower() in LABEL_DIRS:
            labels = {LABEL_DIRS[root.name.lower()]}
        if len(labels) != 1:
            missing.append(str(p))
            continue
        digest = file_sha256(p)
        path = str(p.relative_to(Path(relative_to).resolve())) if relative_to else str(p)
        rows.append({'path': path, 'label': labels.pop(), 'source': source or root.name,
                     'split': split, 'group': digest, 'sha256': digest})
    if missing:
        raise ValueError(f'{len(missing)} images lack an unambiguous real/fake folder. Examples: {missing[:3]}')
    write_rows(output, rows)
    return {'manifest': str(output), 'images': len(rows), 'labels': dict(Counter(r['label'] for r in rows))}


def split_manifest(input_path, train_path, val_path, fraction=0.1, seed=42, data_root=None):
    if not 0 < fraction < 1:
        raise ValueError('Validation fraction must lie in (0,1)')
    rows = read_rows(input_path, data_root)
    groups = defaultdict(list)
    content_labels = {}
    for row in rows:
        digest = row['sha256'] or file_sha256(row['path'])
        if digest in content_labels and content_labels[digest] != row['label']:
            raise ValueError('Identical image bytes carry conflicting real/fake labels')
        content_labels[digest] = row['label']
        row['sha256'] = digest
        groups[row['group'] or digest].append(row)
    # If a duplicate has different user-provided groups, fail rather than leak it.
    sha_groups = defaultdict(set)
    for group, entries in groups.items():
        for row in entries:
            sha_groups[row['sha256']].add(group)
    if any(len(g) > 1 for g in sha_groups.values()):
        raise ValueError('Duplicate image content was assigned inconsistent group IDs')
    strata = defaultdict(list)
    for group, entries in groups.items():
        signature = tuple(sorted({(r['source'], r['label']) for r in entries}))
        strata[signature].append(group)
    rng = random.Random(seed)
    validation = set()
    for signature, members in sorted(strata.items()):
        members.sort()
        if len(members) < 2:
            raise ValueError(f'At least two independent groups needed in stratum {signature}')
        rng.shuffle(members)
        n = min(len(members)-1, max(1, round(len(members)*fraction)))
        validation.update(members[:n])
    train, val = [], []
    for group, entries in groups.items():
        for row in entries:
            copied = {**row, 'split': 'val' if group in validation else 'train'}
            (val if group in validation else train).append(copied)
    write_rows(train_path, train)
    write_rows(val_path, val)
    return {'train_images': len(train), 'val_images': len(val), 'group_disjoint': True}


def audit(manifests, data_root, *, rehash=False, verify_images=False, require_hash=True):
    """Disallow train/val/test leakage; shared real samples across test domains are allowed.

    `manifests` is [(role, name, path), ...]. Hashes are mandatory by default;
    rehash=True additionally checks current image bytes against stored hashes.
    """
    sets, reports = [], []
    for role, name, path in manifests:
        rows = read_rows(path, data_root)
        digests, groups, paths = set(), set(), set()
        content_labels = {}
        for row in rows:
            if row['split'] and row['split'] != role:
                raise ValueError(f'{path} declares split={row["split"]}, expected {role}')
            old = row['sha256']
            if rehash:
                digest = file_sha256(row['path'])
                if old and digest != old:
                    raise ValueError(f'Image content changed since manifest creation: {row["path"]}')
            else:
                digest = old
            if not digest and require_hash:
                raise ValueError(f'Missing sha256 for {row["path"]}; generate manifests with scan')
            if digest:
                if digest in content_labels and content_labels[digest] != row['label']:
                    raise ValueError(f'Conflicting labels for identical content in {path}')
                content_labels[digest] = row['label']
                digests.add(digest)
            if row['group']:
                groups.add(row['group'])
            paths.add(row['path'])
            if verify_images:
                with Image.open(row['path']) as image:
                    image.verify()
        if role in ('train', 'val') and {r['label'] for r in rows} != {0, 1}:
            raise ValueError(f'{role} manifest must contain both classes')
        sets.append((role, name, digests, groups, paths))
        reports.append({'name': name, 'role': role, 'images': len(rows), 'unique_contents': len(digests),
                        'labels': dict(Counter(r['label'] for r in rows)), 'manifest_sha256': file_sha256(path)})
    for i, a in enumerate(sets):
        for b in sets[i+1:]:
            if a[0] == b[0] == 'test':
                continue
            overlap = (a[2] & b[2]) or (a[3] & b[3]) or (a[4] & b[4])
            if overlap:
                raise ValueError(f'Data leakage between {a[1]} and {b[1]}: {len(overlap)} overlapping hashes/groups/paths')
    return {'passed': True, 'rehash': rehash, 'verify_images': verify_images, 'manifests': reports}


def parse_corruption(spec):
    if spec == 'clean':
        return 'clean', 0.0
    try:
        name, value = spec.split(':')
        value = float(value)
    except (ValueError, AttributeError):
        raise ValueError(f'Invalid corruption {spec}; use clean/jpeg:75/blur:1/resize:0.5/noise:0.01') from None
    limits = {'jpeg': (1, 100), 'blur': (0, 20), 'resize': (0.01, 1), 'noise': (0, 1)}
    if name not in limits or not np.isfinite(value) or not limits[name][0] <= value <= limits[name][1]:
        raise ValueError(f'Unsupported corruption/value: {spec}')
    return name, value


def preprocess(image, image_size=224, min_size=256, training=False, corruption='clean', identity=''):
    image = ImageOps.exif_transpose(image).convert('RGB')
    # Matches upstream judge_img followed by crop; EXIF orientation is explicitly normalized.
    if min(image.size) < max(min_size, image_size):
        size = max(min_size, image_size)
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    width, height = image.size
    if training:
        left, top = random.randint(0, width-image_size), random.randint(0, height-image_size)
    else:
        left, top = round((width-image_size)/2), round((height-image_size)/2)
    image = image.crop((left, top, left+image_size, top+image_size))
    if training and random.random() < 0.5:
        image = ImageOps.mirror(image)
    name, value = parse_corruption(corruption)
    if name == 'jpeg':
        with io.BytesIO() as buffer:
            image.save(buffer, format='JPEG', quality=int(value))
            buffer.seek(0)
            image = Image.open(buffer).convert('RGB')
    elif name == 'blur':
        image = image.filter(ImageFilter.GaussianBlur(value))
    elif name == 'resize':
        small = max(1, round(image_size * value))
        image = image.resize((small, small), Image.Resampling.BILINEAR).resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    if name == 'noise':
        seed = int(hashlib.sha256((identity + corruption).encode()).hexdigest()[:16], 16)
        array = np.clip(array + np.random.default_rng(seed).normal(0, value, array.shape), 0, 1).astype(np.float32)
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
    return (tensor - MEAN) / STD


class ManifestDataset(Dataset):
    def __init__(self, manifest, data_root, image_size=224, min_size=256, training=False, corruption='clean'):
        self.rows = read_rows(manifest, data_root)
        self.image_size, self.min_size = image_size, min_size
        self.training, self.corruption = training, corruption
        parse_corruption(corruption)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        try:
            with Image.open(row['path']) as image:
                x = preprocess(image, self.image_size, self.min_size, self.training, self.corruption,
                               row['sha256'] or row['path'])
        except Exception as e:
            # Never silently replace a corrupt image with a black tensor and its old label.
            raise RuntimeError(f'Cannot decode/preprocess image: {row["path"]}') from e
        return x, row['label'], index
