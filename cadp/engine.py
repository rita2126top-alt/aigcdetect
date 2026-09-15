"""Single-device training, reproducible evaluation and strict adapter checkpoints."""
from __future__ import annotations
import contextlib
import csv
import json
import math
import os
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .backbone import file_sha256
from .config import dump_config, manifest_path
from .data import ManifestDataset, audit
from .metrics import binary_metrics
from .model import Detector, detection_loss


def save_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)


def device_for(cfg):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This release uses one device per experiment, not DDP. Run independent jobs with CUDA_VISIBLE_DEVICES.')
    d = torch.device(cfg['train']['device'])
    if d.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices: cpu or cuda[:index]')
    if d.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; check the PyTorch CUDA wheel and NVIDIA driver. No silent CPU fallback.')
        d = torch.device('cuda', torch.cuda.current_device() if d.index is None else d.index)
        torch.cuda.set_device(d)
        if cfg['train']['amp'] == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('GPU does not support bfloat16; select train.amp=fp16 or off.')
    if d.type == 'cpu' and cfg['train']['amp'] == 'fp16':
        raise ValueError('CPU fp16 training is not supported; select off or bf16.')
    return d


def autocast_context(cfg, device):
    mode = cfg['train']['amp']
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16 if mode == 'bf16' else torch.float16,
                          enabled=mode != 'off')


def make_loader(cfg, manifest, image_size, training=False, corruption='clean'):
    ds = ManifestDataset(manifest, cfg['paths']['data_root'], image_size,
                         cfg['data']['min_size'], training, corruption)
    generator = torch.Generator().manual_seed(cfg['seed'])
    return DataLoader(ds, batch_size=cfg['train']['batch_size'] if training else cfg['eval']['batch_size'],
                      shuffle=training, num_workers=cfg['data']['workers'],
                      pin_memory=cfg['data']['pin_memory'] and str(cfg['train']['device']).startswith('cuda'),
                      drop_last=False, worker_init_fn=worker_seed, generator=generator,
                      persistent_workers=False)


def manifest_records(cfg, include_tests=True):
    result = [('train', 'train', manifest_path(cfg, cfg['data']['train'])),
              ('val', 'val', manifest_path(cfg, cfg['data']['val']))]
    if include_tests:
        result += [('test', x['name'], manifest_path(cfg, x['manifest'])) for x in cfg['data']['tests']]
    return result


def audit_config(cfg, *, include_tests=True, rehash=None, verify_images=None):
    return audit(manifest_records(cfg, include_tests), cfg['paths']['data_root'],
                 rehash=cfg['data']['hash_check'] if rehash is None else rehash,
                 verify_images=cfg['data']['verify_images'] if verify_images is None else verify_images,
                 require_hash=True)


def architecture(cfg):
    # Throughput/evaluation settings do not change learned parameter semantics.
    ignored = {'eval_samples', 'checkpoint_blocks', 'text_chunk_size'}
    return {k: v for k, v in cfg['model'].items() if k not in ignored}


def delta_keys(model):
    return {k for k in model.state_dict() if not k.startswith('backbone.clip.') or 'lora_' in k}


def checkpoint_payload(model, cfg, epoch=0, optimizer=None, scheduler=None, scaler=None,
                       best=-1.0, manifests=None):
    state = model.state_dict()
    payload = {'format': 'cadp-v1', 'epoch': epoch, 'best_val_auroc': best,
               'config': cfg, 'architecture': architecture(cfg), 'base_sha256': model.backbone.fingerprint,
               'model': {k: state[k].detach().cpu() for k in sorted(delta_keys(model))},
               'manifests': manifests or {},
               'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                               'cuda_runtime': torch.version.cuda}}
    for name, obj in [('optimizer', optimizer), ('scheduler', scheduler), ('scaler', scaler)]:
        if obj is not None:
            payload[name] = obj.state_dict()
    return payload


def atomic_checkpoint(path, payload):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    torch.save(payload, temp); os.replace(temp, p)


def read_checkpoint(path):
    value = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(value, dict) or value.get('format') != 'cadp-v1':
        raise ValueError('Not a CADP v1 adapter checkpoint; original PPM checkpoints are not interchangeable.')
    return value


def apply_checkpoint(model, cfg, payload):
    if architecture(cfg) != payload['architecture']:
        raise ValueError('Checkpoint model configuration differs. Use its saved config.yaml, not another ablation.')
    if payload['base_sha256'] != model.backbone.fingerprint:
        raise ValueError('Pretrained CLIP SHA256/tiny seed mismatch; refusing to silently change the frozen backbone.')
    expected = delta_keys(model)
    if set(payload['model']) != expected:
        raise ValueError(f'Checkpoint key mismatch: missing={expected-set(payload["model"])}, extra={set(payload["model"])-expected}')
    model.load_state_dict(payload['model'], strict=False)  # Exact key set checked above; only frozen CLIP keys omitted.


def load_model(cfg, checkpoint=None):
    seed_all(cfg['seed']); device = device_for(cfg)
    if cfg['model']['method'] == 'ppm_baseline':
        from .legacy import LegacyDetector
        model = LegacyDetector(cfg).to(device)
    else:
        model = Detector(cfg).to(device)
    if checkpoint:
        apply_checkpoint(model, cfg, read_checkpoint(checkpoint))
    return model, device


@torch.no_grad()
def predict_loader(model, loader, cfg, device):
    was_training = model.training; model.eval()
    output_rows = []
    try:
        for x, y, indices in loader:
            with autocast_context(cfg, device):
                result = model(x.to(device, non_blocking=True))
            probabilities = result['probabilities'].float().cpu().numpy()
            lengths = result['private_lengths'].cpu().tolist()
            uncertainty = (result['pair_probabilities'][..., 1].var(dim=(1,2), unbiased=False).cpu().tolist()
                           if result.get('uncertainty_supported', True) else [None]*len(y))
            for j, index in enumerate(indices.tolist()):
                source = loader.dataset.rows[index]
                p = float(probabilities[j, 1])
                output_rows.append({'index': index, 'path': source['path'], 'label': int(y[j]),
                                    'source': source['source'], 'group': source['group'], 'sha256': source['sha256'],
                                    'p_real': float(probabilities[j, 0]), 'p_fake': p,
                                    'prediction': int(p >= cfg['eval']['threshold']),
                                    'private_length': lengths[j], 'prompt_probability_variance': uncertainty[j]})
    finally:
        model.train(was_training)
    return output_rows


def metrics_for_rows(rows, cfg):
    return binary_metrics([r['label'] for r in rows], [r['p_fake'] for r in rows],
                          threshold=cfg['eval']['threshold'], bins=cfg['eval']['calibration_bins'],
                          bootstrap=cfg['eval']['bootstrap'], seed=cfg['seed'])


def write_predictions(path, rows):
    if not rows:
        raise ValueError('No predictions to write')
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def evaluate(cfg, checkpoint, destination=None):
    if not cfg['data']['tests']:
        raise ValueError('data.tests is empty; register held-out test manifests in YAML.')
    # Enforce train/validation/test separation even when evaluating a pretrained adapter.
    report = audit_config(cfg)
    model, device = load_model(cfg, checkpoint)
    destination = Path(destination or Path(cfg['paths']['output_dir'])/'evaluation')
    checkpoint_digest = file_sha256(checkpoint)
    results = {}
    for entry in cfg['data']['tests']:
        name = entry['name']
        if '/' in name or '\\' in name or name in ('.', '..'):
            raise ValueError('Test domain names must be simple filenames')
        loader = make_loader(cfg, manifest_path(cfg, entry['manifest']), model.backbone.image_size,
                             corruption=cfg['eval']['corruption'])
        rows = predict_loader(model, loader, cfg, device)
        metrics = metrics_for_rows(rows, cfg)
        metrics['length_histogram'] = {str(r): sum(x['private_length']==r for x in rows) for r in model.lengths}
        metrics['by_source'] = {s: metrics_for_rows([r for r in rows if r['source']==s], cfg)
                                for s in sorted({r['source'] for r in rows})}
        write_predictions(destination / (name + '.predictions.csv'), rows)
        save_json(destination / (name + '.metrics.json'), metrics); results[name] = metrics
    keys = ('accuracy', 'balanced_accuracy', 'auroc', 'ap', 'f1', 'ece', 'brier')
    macro = {k: (float(np.mean([m[k] for m in results.values() if m.get(k) is not None]))
                 if any(m.get(k) is not None for m in results.values()) else None) for k in keys}
    summary = {'schema': 'cadp-evaluation-v1', 'seed': cfg['seed'], 'checkpoint_sha256': checkpoint_digest,
               'base_sha256': model.backbone.fingerprint, 'samples': cfg['model']['eval_samples'],
               'corruption': cfg['eval']['corruption'], 'threshold': cfg['eval']['threshold'],
               'method': cfg['model']['method'], 'domains': results, 'macro': macro, 'audit': report, 'tiny_random_backbone': cfg['model']['tiny']}
    save_json(destination/'summary.json', summary); dump_config(cfg, destination/'config.yaml')
    print(json.dumps({'evaluation': str(destination), 'macro': macro}, ensure_ascii=False))
    return summary


def make_scaler(cfg, device):
    return torch.amp.GradScaler(device.type,
        enabled=device.type == 'cuda' and cfg['train']['amp'] == 'fp16',
        init_scale=cfg['train']['grad_scaler_init_scale'])


def optimizer_step(model, optimizer, scaler, max_norm):
    """Let GradScaler skip nonfinite scaled gradients, but fail on unscaled NaNs."""
    scaler.unscale_(optimizer)
    parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not parameters:
        raise RuntimeError('No gradients reached trainable parameters')
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm or float('inf'),
                                         error_if_nonfinite=not scaler.is_enabled())
    finite = bool(torch.isfinite(norm))
    # For enabled scaling, step inspects the overflow recorded during unscale_.
    # Clip must NOT be allowed to hide that overflow by sanitizing gradients.
    scaler.step(optimizer)
    scaler.update()
    return finite, float(norm)


def train(cfg, *, stop_after_epoch=None):
    out = Path(cfg['paths']['output_dir']); out.mkdir(parents=True, exist_ok=True)
    if (out/'last.pt').exists() and not cfg['train']['resume']:
        raise FileExistsError(f'{out}/last.pt already exists; use train.resume or a new output directory.')
    report = audit_config(cfg, include_tests=False); save_json(out/'data_audit.json', report)
    signatures = {name: file_sha256(path) for _, name, path in manifest_records(cfg, include_tests=False)}
    model, device = load_model(cfg)
    lora, other = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (lora if 'lora_' in name else other).append(parameter)
    groups = [{'params': other, 'lr': cfg['train']['lr'], 'group_name': 'new_modules'}]
    if lora:
        groups.append({'params': lora, 'lr': cfg['train']['lora_lr'], 'group_name': 'vision_lora'})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg['train']['weight_decay'])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])
                 if cfg['train']['scheduler'] == 'cosine' else None)
    scaler = make_scaler(cfg, device)
    start, best = 0, -1.0
    if cfg['train']['resume']:
        if Path(cfg['train']['resume']).resolve().parent != out.resolve():
            raise ValueError('Resume must use the same run directory. Move the entire run directory, including best.pt and history, when relocating.')
        value = read_checkpoint(cfg['train']['resume']); apply_checkpoint(model, cfg, value)
        if value['manifests'] != signatures:
            raise ValueError('Resume manifests differ from the checkpoint.')
        for key in ('epochs', 'batch_size', 'accumulation_steps', 'lr', 'lora_lr', 'weight_decay', 'amp', 'scheduler', 'grad_clip', 'grad_scaler_init_scale'):
            if value['config']['train'][key] != cfg['train'][key]:
                raise ValueError(f'Resume setting differs: train.{key}. Start a new run for a different protocol.')
        if value['config']['seed'] != cfg['seed'] or value['config']['loss'] != cfg['loss'] or value['config']['data']['min_size'] != cfg['data']['min_size']:
            raise ValueError('Resume seed, loss weights or preprocessing differ.')
        optimizer.load_state_dict(value['optimizer']); scaler.load_state_dict(value['scaler'])
        if scheduler:
            scheduler.load_state_dict(value['scheduler'])
        start, best = value['epoch'], value['best_val_auroc']
    dump_config(cfg, out/'config.yaml')
    loader = make_loader(cfg, manifest_path(cfg, cfg['data']['train']), model.backbone.image_size, training=True)
    validation = make_loader(cfg, manifest_path(cfg, cfg['data']['val']), model.backbone.image_size)
    save_json(out/'parameters.json', {'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                                    'total': sum(p.numel() for p in model.parameters()),
                                    'effective_batch_size': cfg['train']['batch_size']*cfg['train']['accumulation_steps'],
                                    'device': str(device), 'tiny_random_backbone': cfg['model']['tiny']})
    for epoch in range(start, cfg['train']['epochs']):
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            break
        seed_all(cfg['seed'] + 100003*epoch); loader.generator.manual_seed(cfg['seed']+epoch)
        model.train(); optimizer.zero_grad(set_to_none=True)
        totals, count, begin = {}, 0, time.perf_counter()
        updates, skipped = 0, 0
        batch_size, accumulation = cfg['train']['batch_size'], cfg['train']['accumulation_steps']
        for step, (x, y, _) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # Sample-count normalization is correct even for an incomplete final accumulation window.
            window_first = (step//accumulation)*accumulation*batch_size
            window_count = min(accumulation*batch_size, len(loader.dataset)-window_first)
            with autocast_context(cfg, device):
                output = model(x, epoch=epoch+1)
                loss, terms = detection_loss(output, y, cfg['loss'])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'Nonfinite loss at epoch={epoch+1}, batch={step}; no checkpoint overwritten.')
            scaler.scale(loss * len(x)/window_count).backward()
            if (step+1)%accumulation == 0 or step+1 == len(loader):
                updated, _ = optimizer_step(model, optimizer, scaler, cfg['train']['grad_clip'])
                updates += int(updated); skipped += int(not updated)
                optimizer.zero_grad(set_to_none=True)
            for key, value in {'total': loss, **terms}.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach())*len(x)
            count += len(x)
        if not updates:
            raise FloatingPointError('Every optimizer step overflowed; reduce AMP scale, select bf16/off, or inspect losses.')
        if scheduler:
            scheduler.step()
        val_rows = predict_loader(model, validation, cfg, device)
        metrics = metrics_for_rows(val_rows, cfg); score = metrics['auroc']
        if score is None:
            raise ValueError('Validation must contain both classes for checkpoint selection.')
        improved = score > best; best = max(best, score)
        payload = checkpoint_payload(model, cfg, epoch+1, optimizer, scheduler, scaler, best, signatures)
        atomic_checkpoint(out/'last.pt', payload)
        if improved:
            atomic_checkpoint(out/'best.pt', payload)
            write_predictions(out/'best_validation.predictions.csv', val_rows)
            save_json(out/'best_validation.metrics.json', metrics)
        log = {'epoch': epoch+1, 'train': {k:v/count for k,v in totals.items()}, 'validation': metrics,
               'optimizer_steps': updates, 'skipped_amp_steps': skipped,
               'seconds': time.perf_counter()-begin, 'learning_rates': [g['lr'] for g in optimizer.param_groups]}
        with (out/'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(log, allow_nan=False)+'\n')
        print(json.dumps(log, ensure_ascii=False), flush=True)
    return str(out/'last.pt')


def preflight(cfg):
    """Real model/data forward + backward on requested device; no fake GPU success."""
    report = audit_config(cfg, include_tests=False)
    model, device = load_model(cfg); model.train()
    loader = make_loader(cfg, manifest_path(cfg, cfg['data']['train']), model.backbone.image_size)
    x,y,_ = next(iter(loader))
    with autocast_context(cfg, device):
        result = model(x.to(device)); loss, terms = detection_loss(result, y.to(device), cfg['loss'])
    scaler = make_scaler(cfg, device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['train']['lr'])
    scaler.scale(loss).backward()
    updated, _ = optimizer_step(model, optimizer, scaler, cfg['train']['grad_clip'])
    if not updated:
        raise FloatingPointError('Preflight AMP gradients overflowed; reduce train.grad_scaler_init_scale or use bf16/off.')
    grads = {n: float(p.grad.detach().float().norm()) for n,p in model.named_parameters() if p.grad is not None}
    if not all(math.isfinite(v) for v in grads.values()) or not math.isfinite(float(loss.detach())):
        raise FloatingPointError('Preflight found nonfinite loss/gradients')
    value = {'passed': True, 'device': str(device), 'tiny_random_backbone': cfg['model']['tiny'],
             'optimizer_step_executed': True, 'amp': cfg['train']['amp'],
             'base_sha256': model.backbone.fingerprint, 'losses': {k:float(v.detach()) for k,v in terms.items()},
             'gradient_norms': grads, 'audit': report}
    save_json(Path(cfg['paths']['output_dir'])/'preflight.json', value)
    print(json.dumps({'passed': True, 'device': str(device), 'loss': float(loss.detach()), 'tiny': cfg['model']['tiny']}))
    return value
