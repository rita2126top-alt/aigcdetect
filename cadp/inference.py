"""Image/folder prediction, validation calibration, attention export and timing."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .backbone import file_sha256
from .config import manifest_path
from .data import EXTENSIONS, preprocess
from .engine import (autocast_context, audit_config, load_model, make_loader, predict_loader,
                     save_json, write_predictions)
from .metrics import choose_threshold


@torch.no_grad()
def predict(cfg, checkpoint, input_path, output, attention=False):
    model, device = load_model(cfg, checkpoint); model.eval()
    root = Path(input_path)
    files = [root] if root.is_file() else sorted(p for p in root.rglob('*') if p.suffix.lower() in EXTENSIONS)
    if not files:
        raise FileNotFoundError(f'No supported images found at {root}')
    rows = []
    for i, path in enumerate(files):
        digest = file_sha256(path)
        with Image.open(path) as image:
            x = preprocess(image, model.backbone.image_size, cfg['data']['min_size'],
                           corruption=cfg['eval']['corruption'], identity=digest).unsqueeze(0).to(device)
        with autocast_context(cfg, device):
            result = model(x, return_attention=attention)
        p0, p1 = result['probabilities'][0].cpu().tolist()
        rows.append({'path': str(path.resolve()), 'p_real': p0, 'p_fake': p1,
                     'prediction': int(p1 >= cfg['eval']['threshold']),
                     'private_length': int(result['private_lengths'][0]),
                     'threshold': cfg['eval']['threshold'], 'sha256': digest})
        if attention:
            arrays = {f'private_{r}': torch.stack(v['attention'], dim=1).float().cpu().numpy()
                      for r,v in result['attention'].items() if v['attention']}
            dest = Path(output).parent/'attention'; dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest/f'{i:06d}_{path.stem}.npz', **arrays)
    write_predictions(output, rows)
    print(json.dumps({'predictions': str(output), 'images': len(rows)}))
    return rows


def calibrate(cfg, checkpoint, output):
    # No arbitrary CSV argument: only configured validation data may select threshold.
    audit_config(cfg, include_tests=False)
    model, device = load_model(cfg, checkpoint)
    loader = make_loader(cfg, manifest_path(cfg, cfg['data']['val']), model.backbone.image_size,
                         corruption=cfg['eval']['corruption'])
    rows = predict_loader(model, loader, cfg, device)
    threshold = choose_threshold([r['label'] for r in rows], [r['p_fake'] for r in rows])
    value = {'threshold': threshold, 'criterion': 'validation-only Youden J',
             'checkpoint_sha256': file_sha256(checkpoint),
             'validation_manifest_sha256': file_sha256(manifest_path(cfg, cfg['data']['val'])),
             'samples': cfg['model']['eval_samples'], 'corruption': cfg['eval']['corruption']}
    save_json(output, value); print(json.dumps(value)); return value


@torch.no_grad()
def benchmark(cfg, checkpoint, output, warmup=3, repeats=20):
    if warmup < 0 or repeats < 1:
        raise ValueError('warmup >= 0 and repeats >= 1 required')
    model, device = load_model(cfg, checkpoint); model.eval()
    loader = make_loader(cfg, manifest_path(cfg, cfg['data']['val']), model.backbone.image_size)
    x = next(iter(loader))[0].to(device)
    def synchronize():
        if device.type=='cuda':
            torch.cuda.synchronize(device)
    times=[]
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for index in range(warmup+repeats):
        synchronize(); start=time.perf_counter()
        with autocast_context(cfg, device):
            model(x)
        synchronize()
        if index>=warmup:
            times.append(time.perf_counter()-start)
    result = {'schema': 'cadp-benchmark-v1', 'seed': cfg['seed'],
              'method': cfg['model']['method'], 'checkpoint_sha256': file_sha256(checkpoint),
              'device': str(device), 'batch_size': len(x), 'samples': cfg['model']['eval_samples'],
              'repositories': cfg['model']['repositories'], 'includes': 'full visual + routed text + MC + cross-attention forward',
              'excludes': 'disk reads, preprocessing, host-to-device transfer, checkpoint load, training',
              'batch_latency_ms_p50': float(np.median(times)*1000),
              'batch_latency_ms_p95': float(np.quantile(times,.95)*1000),
              'throughput_images_s': float(len(x)/np.mean(times)),
              'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
              'tiny_random_backbone': cfg['model']['tiny']}
    save_json(output, result); print(json.dumps(result)); return result
