"""Strict YAML configuration; all paths resolve without network access."""
from __future__ import annotations
import copy
import os
import re
import math
from pathlib import Path
from typing import Any
import yaml

class ConfigLoader(yaml.SafeLoader):
    # YAML 1.2 boolean semantics: the string "off" must not become False.
    yaml_implicit_resolvers = copy.deepcopy(yaml.SafeLoader.yaml_implicit_resolvers)

for _key, _rules in list(ConfigLoader.yaml_implicit_resolvers.items()):
    ConfigLoader.yaml_implicit_resolvers[_key] = [(tag, regex) for tag, regex in _rules if tag != 'tag:yaml.org,2002:bool']
ConfigLoader.add_implicit_resolver('tag:yaml.org,2002:bool', re.compile(r'^(?:true|false|True|False|TRUE|FALSE)$'), list('tTfF'))

def yaml_value(value):
    return yaml.load(value, Loader=ConfigLoader)

DEFAULTS = {
    'seed': 42,
    'paths': {'clip_checkpoint': '/data/aigcdetect/models/ViT-L-14.pt',
              'data_root': '/data/aigcdetect/datasets',
              'manifest_dir': '/data/aigcdetect/manifests',
              'output_dir': '/data/aigcdetect/runs/main_seed42'},
    'model': {'method': 'cadp', 'legacy_orthogonal_weight': 1.0, 'backbone': 'ViT-L/14', 'tiny': False, 'tiny_seed': 123,
              'repositories': 2, 'shared_tokens': 3, 'private_tokens': 7,
              'lengths': [1, 3, 5, 7], 'fixed_length': None,
              'flows': 10, 'hidden_dim': 256, 'rho_shared': 0.1, 'rho_private': 0.1,
              'cross_attention': True, 'attention_dim': 256, 'attention_heads': 4,
              'attention_dropout': 0.1, 'anchor_trainable': True,
              'alpha_max': 0.5, 'alpha_init': 0.1,
              'stochastic': True, 'eval_samples': 10, 'noise_capacity': 50,
              'noise_seed': 2026, 'lora_rank': 4, 'lora_alpha': 0.5,
              'lora_dropout': 0.0, 'dct_rate': 0.5,
              'checkpoint_blocks': True, 'text_chunk_size': 8},
    'loss': {'rec': 0.5, 'kl': 0.001, 'patch': 0.5, 'anchor': 0.001, 'length': 0.001},
    'data': {'train': 'train.csv', 'val': 'val.csv', 'tests': [],
             'workers': 4, 'pin_memory': True, 'min_size': 256,
             'verify_images': True, 'hash_check': True},
    'train': {'epochs': 100, 'batch_size': 2, 'accumulation_steps': 24,
              'lr': 0.0001, 'lora_lr': 0.00001, 'weight_decay': 0.0001,
              'amp': 'bf16', 'scheduler': 'cosine', 'grad_clip': 1.0,
              'device': 'cuda', 'resume': None, 'grad_scaler_init_scale': 256.0},
    'eval': {'batch_size': 2, 'threshold': 0.5, 'corruption': 'clean',
             'calibration_bins': 15, 'bootstrap': 0},
}

def _merge(dst: dict, src: dict, prefix: str = '') -> dict:
    for k, v in src.items():
        if k not in dst:
            raise ValueError(f'Unknown configuration key: {prefix}{k}')
        if isinstance(dst[k], dict):
            if not isinstance(v, dict):
                raise ValueError(f'{prefix}{k} must be a mapping')
            _merge(dst[k], v, prefix + k + '.')
        else:
            dst[k] = v
    return dst

def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, encoding='utf-8') as f:
            raw = yaml_value(f) or {}
        _merge(cfg, raw)
    for item in overrides or []:
        if '=' not in item:
            raise ValueError(f'Override needs key=value: {item}')
        key, value = item.split('=', 1)
        cursor = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                raise ValueError(f'Unknown override: {key}')
            cursor = cursor[part]
        if parts[-1] not in cursor:
            raise ValueError(f'Unknown override: {key}')
        cursor[parts[-1]] = yaml_value(value)
    for key, value in cfg['paths'].items():
        cfg['paths'][key] = str(Path(os.path.expandvars(str(value))).expanduser().resolve())
    validate_config(cfg)
    return cfg

def validate_config(c: dict) -> None:
    # Normalize scientific-notation scalars and validate primitive types against the schema.
    for section, defaults in DEFAULTS.items():
        if not isinstance(defaults, dict):
            continue
        for key, default in defaults.items():
            value = c[section][key]
            if default is None:
                continue
            if isinstance(default, float):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    raise ValueError(f'{section}.{key} must be numeric') from None
                if not math.isfinite(value):
                    raise ValueError(f'{section}.{key} must be finite')
                c[section][key] = value
            elif isinstance(default, bool) and type(value) is not bool:
                raise ValueError(f'{section}.{key} must be true or false')
            elif type(default) is int and type(value) is not int:
                raise ValueError(f'{section}.{key} must be an integer')
            elif isinstance(default, str) and not isinstance(value, str):
                raise ValueError(f'{section}.{key} must be a string')
            elif isinstance(default, list) and not isinstance(value, list):
                raise ValueError(f'{section}.{key} must be a list')
    if type(c['seed']) is not int:
        raise ValueError('seed must be an integer')
    m, t, d = c['model'], c['train'], c['data']
    if m['method'] not in ('cadp', 'ppm_baseline'):
        raise ValueError('model.method must be cadp or ppm_baseline')
    for k in ('repositories', 'private_tokens', 'hidden_dim', 'attention_dim',
              'attention_heads', 'eval_samples', 'noise_capacity', 'text_chunk_size'):
        if not isinstance(m[k], int) or m[k] < 1:
            raise ValueError(f'model.{k} must be a positive integer')
    if m['shared_tokens'] < 0 or m['flows'] < 0 or m['lora_rank'] < 0:
        raise ValueError('Token, flow and rank counts must be nonnegative')
    lengths = m['lengths']
    if not lengths or lengths != sorted(set(lengths)) or any(not isinstance(r, int) or r < 0 or r > m['private_tokens'] for r in lengths):
        raise ValueError('lengths must be sorted unique integers in [0, private_tokens]')
    if min(lengths) + m['shared_tokens'] < 1 or max(lengths) + m['shared_tokens'] + 6 > 77:
        raise ValueError('Prompts must contain at least one context token and fit 77 positions')
    if m['fixed_length'] is not None and m['fixed_length'] not in lengths:
        raise ValueError('fixed_length must occur in lengths')
    if not 0 < m['alpha_init'] < m['alpha_max']:
        raise ValueError('Require 0 < alpha_init < alpha_max')
    if m['attention_dim'] % m['attention_heads']:
        raise ValueError('attention_dim must be divisible by attention_heads')
    if m['eval_samples'] > m['noise_capacity']:
        raise ValueError('eval_samples exceeds the fixed noise bank capacity')
    if not 0 < m['dct_rate'] < 1:
        raise ValueError('dct_rate must lie in (0,1)')
    if m['lora_dropout'] != 0:
        raise ValueError('The weight-space qkv LoRA implementation requires dropout=0 (the supplied method)')
    if m['backbone'] != 'ViT-L/14' and not m['tiny']:
        raise ValueError('This release validates the specified ViT-L/14 architecture only')
    for k in ('epochs', 'batch_size', 'accumulation_steps'):
        if not isinstance(t[k], int) or t[k] < 1:
            raise ValueError(f'train.{k} must be a positive integer')
    if t['grad_scaler_init_scale'] <= 0:
        raise ValueError('train.grad_scaler_init_scale must be positive')
    if t['lr'] <= 0 or t['lora_lr'] <= 0 or t['weight_decay'] < 0 or t['grad_clip'] < 0:
        raise ValueError('Learning rates must be positive; decay and gradient clipping nonnegative')
    if not 0 <= m['attention_dropout'] < 1 or m['legacy_orthogonal_weight'] < 0:
        raise ValueError('Invalid attention dropout or legacy orthogonal weight')
    if t['amp'] not in ('off', 'fp16', 'bf16') or t['scheduler'] not in ('none', 'cosine'):
        raise ValueError('Unsupported AMP precision or scheduler')
    if any(float(v) < 0 for v in c['loss'].values()):
        raise ValueError('Loss weights must be nonnegative')
    if d['workers'] < 0 or c['eval']['batch_size'] < 1 or not 0 <= c['eval']['threshold'] <= 1:
        raise ValueError('Invalid loader or evaluation settings')
    if c['eval']['calibration_bins'] < 1 or c['eval']['bootstrap'] < 0:
        raise ValueError('Invalid metric settings')
    if not isinstance(d['tests'], list) or any(set(x) != {'name','manifest'} for x in d['tests']):
        raise ValueError('data.tests must be [{name: domain, manifest: test.csv}, ...]')
    if any(x['name'] in ('train', 'val') for x in d['tests']):
        raise ValueError('Test domain names train/val are reserved')
    if len({x['name'] for x in d['tests']}) != len(d['tests']):
        raise ValueError('Duplicate test domain names')

def manifest_path(c: dict, value: str) -> Path:
    p = Path(os.path.expandvars(value)).expanduser()
    return p if p.is_absolute() else Path(c['paths']['manifest_dir']) / p

def dump_config(c: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(c, allow_unicode=True, sort_keys=False), encoding='utf-8')
