"""Export locally downloaded HF Arrow datasets without re-encoding image bytes.

Each leaf must be a datasets.save_to_disk directory. No network access occurs.
The caller explicitly declares which binary label is fake; we never infer it
from a generator name. PNG/JPEG/WebP bytes retain their original codec.
"""
from __future__ import annotations
import io
import json
import os
from pathlib import Path
from PIL import Image
from .backbone import file_sha256


def export_arrow(snapshot, output, split='test', generator=None, fake_label=1):
    from datasets import Dataset, Image as HFImage, load_from_disk
    if fake_label not in (0, 1):
        raise ValueError('fake_label must be 0 or 1')
    base, out = Path(snapshot).resolve(), Path(output).resolve()
    source = base/'data'/split
    if not source.is_dir():
        source = base/split
    if not source.is_dir():
        raise FileNotFoundError(f'Expected local snapshot data/{split} or {split}: {base}')
    leaves = [source] if (source/'state.json').exists() else sorted(p for p in source.iterdir() if p.is_dir())
    if generator:
        leaves = [p for p in leaves if p.name == generator]
    if not leaves:
        raise ValueError(f'No Arrow leaf matches generator={generator!r} under {source}')
    reports = []
    for leaf in leaves:
        ds = load_from_disk(str(leaf))
        if not isinstance(ds, Dataset) or not {'image', 'label'}.issubset(ds.column_names):
            raise ValueError(f'{leaf} must contain one Dataset with image,label columns')
        ds = ds.cast_column('image', HFImage(decode=False))
        counts = {0: 0, 1: 0}
        for index, row in enumerate(ds):
            if row['label'] not in (0, 1):
                raise ValueError(f'{leaf} row {index}: nonbinary label {row["label"]!r}')
            image = row['image']
            raw = image.get('bytes')
            if raw is None:
                path = Path(image.get('path') or '')
                if not path.is_absolute():
                    path = leaf/path
                if not path.is_file():
                    raise FileNotFoundError(f'No local embedded bytes or image for {leaf} row {index}')
                raw = path.read_bytes()
            with Image.open(io.BytesIO(raw)) as im:
                fmt = im.format
                im.verify()
            ext = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp', 'BMP': '.bmp', 'TIFF': '.tiff'}.get(fmt)
            if ext is None:
                raise ValueError(f'Unsupported image codec {fmt!r}; refusing silent re-encoding')
            label = int(row['label'] == fake_label)
            dest = out/leaf.name/('1_fake' if label else '0_real')/f'{index:09d}{ext}'
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                import hashlib
                if file_sha256(dest) != hashlib.sha256(raw).hexdigest():
                    raise FileExistsError(f'Existing converted image differs: {dest}')
            else:
                tmp = dest.with_suffix(dest.suffix+'.part')
                tmp.write_bytes(raw); os.replace(tmp, dest)
            counts[label] += 1
        if not len(ds):
            raise ValueError(f'Empty Arrow dataset: {leaf}')
        reports.append({'generator': leaf.name, 'rows': len(ds), 'labels': counts,
                        'original_features': str(ds.features), 'source': str(leaf)})
    out.mkdir(parents=True, exist_ok=True)
    result = {'split': split, 'fake_label_in_source': fake_label, 'reencoded': False, 'datasets': reports}
    (out/'export_report.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
