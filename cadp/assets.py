"""Explicit downloads only; training/evaluation never invoke these functions."""
from __future__ import annotations
import json
import os
import shutil
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
import yaml
from .backbone import file_sha256
from .config import dump_config
from .data import scan, split_manifest
from .engine import save_json

CLIP_SHA256 = 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
CLIP_URL = f'https://openaipublic.azureedge.net/clip/models/{CLIP_SHA256}/ViT-L-14.pt'


def download_file(url, destination, sha256=None, retries=3):
    """Resume into .part, verify whole-file SHA256, then rename atomically."""
    if not url.startswith(('https://', 'http://')):
        raise ValueError('Only HTTP(S) downloads are supported')
    p = Path(destination); p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        if sha256 and file_sha256(p) == sha256.lower():
            return str(p)
        raise FileExistsError(f'{p} exists but is unverified or has a wrong checksum. Move it aside explicitly.')
    part = p.with_suffix(p.suffix + '.part')
    for attempt in range(retries):
        try:
            offset = part.stat().st_size if part.exists() else 0
            request = urllib.request.Request(url, headers={'User-Agent': 'cadp-research/0.1',
                                                          **({'Range': f'bytes={offset}-'} if offset else {})})
            with urllib.request.urlopen(request, timeout=120) as response:
                resumed = offset and response.status == 206
                if resumed and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise IOError('Server returned an inconsistent Content-Range')
                with part.open('ab' if resumed else 'wb') as f:
                    shutil.copyfileobj(response, f, length=4*1024*1024)
            if sha256 and file_sha256(part) != sha256.lower():
                raise ValueError(f'SHA256 mismatch for {part}; remove .part and retry from a clean download.')
            os.replace(part, p)
            save_json(str(p)+'.download.json', {'url': url, 'sha256': file_sha256(p), 'bytes': p.stat().st_size})
            return str(p)
        except (OSError, TimeoutError):
            if attempt+1 == retries:
                raise
            time.sleep(2**attempt)
    raise RuntimeError('Download did not complete')


def download_model(cfg):
    return download_file(CLIP_URL, cfg['paths']['clip_checkpoint'], CLIP_SHA256)


def safe_extract(archive, destination):
    """No archive symlinks/devices, path traversal or overwriting existing files."""
    root = Path(destination).resolve(); root.mkdir(parents=True, exist_ok=True)
    def target(name):
        if '\\' in name:
            raise ValueError('Backslashes in archive member paths are unsupported')
        p = (root/name).resolve()
        if p != root and root not in p.parents:
            raise ValueError(f'Archive path traversal: {name}')
        return p
    def copy(stream, name):
        p = target(name); p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('xb') as f:
            shutil.copyfileobj(stream, f)
    a = Path(archive)
    if zipfile.is_zipfile(a):
        with zipfile.ZipFile(a) as z:
            for info in z.infolist():
                target(info.filename)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Archive symlinks are forbidden')
                if not info.is_dir() and target(info.filename).exists():
                    raise FileExistsError(target(info.filename))
            for info in z.infolist():
                if info.is_dir():
                    target(info.filename).mkdir(parents=True, exist_ok=True)
                else:
                    with z.open(info) as f:
                        copy(f, info.filename)
    elif tarfile.is_tarfile(a):
        with tarfile.open(a) as t:
            members = t.getmembers()
            for info in members:
                target(info.name)
                if not info.isdir() and not info.isfile():
                    raise ValueError('Only regular files/directories are allowed in tar archives')
                if info.isfile() and target(info.name).exists():
                    raise FileExistsError(target(info.name))
            for info in members:
                if info.isdir():
                    target(info.name).mkdir(parents=True, exist_ok=True)
                else:
                    with t.extractfile(info) as f:
                        copy(f, info.name)
    else:
        raise ValueError(f'Unsupported archive {a}; multipart archives require a compatible external unpacker.')
    return str(root)


def download_data(cfg, asset_yaml, execute=False):
    value = yaml.safe_load(Path(asset_yaml).read_text())
    result = []
    for asset in value['assets']:
        dest = Path(os.path.expandvars(asset['destination'])).expanduser()
        if not dest.is_absolute():
            dest = Path(cfg['paths']['data_root'])/dest
        item = {**asset, 'resolved_destination': str(dest), 'executed': execute}
        if execute:
            if asset['type'] == 'http':
                if not asset.get('sha256'):
                    raise ValueError('HTTP dataset assets require a publisher-provided SHA256')
                item['file'] = download_file(asset['url'], dest, asset['sha256'])
            elif asset['type'] == 'gdrive_folder':
                import gdown
                dest.mkdir(parents=True, exist_ok=True)
                # Drive quotas, permissions and large-folder limits are external; never label partial returns as success.
                files = gdown.download_folder(url=asset['url'], output=str(dest), quiet=False, remaining_ok=False)
                if not files:
                    raise RuntimeError('No dataset files were downloaded; check Drive quota/permissions.')
                item['downloaded_files'] = [str(x) for x in files]
                item['note'] = 'Archives are not automatically unpacked; verify actual dataset completeness before training.'
            elif asset['type'] == 'huggingface':
                from huggingface_hub import snapshot_download
                if not asset.get('revision'):
                    raise ValueError('Pin a Hugging Face revision for reproducible downloads')
                item['snapshot'] = snapshot_download(repo_id=asset['repo_id'], repo_type='dataset',
                                                      revision=asset['revision'], local_dir=str(dest),
                                                      allow_patterns=asset.get('allow_patterns'))
            else:
                raise ValueError(f'Unsupported download type {asset["type"]}')
        result.append(item)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def prepare_protocol(cfg, protocol_yaml, source=None, all_sources=False):
    """GenImage official val is TEST; model selection uses a split of source train."""
    import copy
    protocol = yaml.safe_load(Path(protocol_yaml).read_text())
    root = Path(protocol['root']).expanduser()
    if not root.is_absolute():
        root = Path(cfg['paths']['data_root'])/root
    domains = protocol['domains']
    chosen = source or protocol['source']
    if chosen not in domains:
        raise ValueError(f'Unknown source {chosen}; expected one of {list(domains)}')
    manifest_dir = Path(cfg['paths']['manifest_dir']); manifest_dir.mkdir(parents=True, exist_ok=True)
    tests = []
    for name, relative in domains.items():
        target = f'tests/{name}.csv'
        scan(root/relative/protocol.get('official_test_subdir', 'val'), manifest_dir/target,
             split='test', source=name)
        tests.append({'name': name, 'manifest': target})
    configs = []
    for name in (list(domains) if all_sources else [chosen]):
        sub = manifest_dir / f'source_{name}'; sub.mkdir(parents=True, exist_ok=True)
        scan(root/domains[name]/protocol.get('official_train_subdir', 'train'), sub/'all_train.csv', 'train', name)
        split_manifest(sub/'all_train.csv', sub/'train.csv', sub/'val.csv',
                       protocol.get('validation_fraction', 0.1), protocol.get('split_seed', 42), cfg['paths']['data_root'])
        c = copy.deepcopy(cfg)
        c['data'].update(train=f'source_{name}/train.csv', val=f'source_{name}/val.csv', tests=tests)
        c['paths']['output_dir'] = str(Path(cfg['paths']['output_dir']).parent/f'{name}_seed{c["seed"]}')
        target = manifest_dir/f'{name}.yaml'; dump_config(c, target); configs.append(str(target))
    save_json(manifest_dir/'protocol.json', {'protocol': protocol, 'configs': configs,
              'note': 'Held-out official val sets are not used for checkpoint or threshold selection.'})
    print(json.dumps({'ready_configs': configs}, ensure_ascii=False))
    return configs
