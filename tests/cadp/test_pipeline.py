"""Download boundary, protocol isolation and aggregation regression tests."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
import yaml
from cadp.assets import download_file, prepare_protocol
from cadp.config import load_config
from cadp.data import read_rows
from cadp.engine import audit_config, save_json
from cadp.experiments import aggregate
from cadp.model import Detector


def test_router_receives_pure_classification_gradient(cfg):
    model=Detector(cfg).train()
    out=model(torch.randn(2,3,28,28))
    p=out['pair_probabilities']
    loss=-(p[0,...,0].log().mean()+p[1,...,1].log().mean())/2
    loss.backward()
    grad=model.length_router[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0


@pytest.mark.parametrize('offset,range_supported', [(0,False),(3,True),(3,False)])
def test_http_download_resume_checks_sha_atomic(tmp_path,monkeypatch,offset,range_supported):
    content=b'abcdefghij0123456789';target=tmp_path/'weights.bin'
    if offset: target.with_suffix('.bin.part').write_bytes(content[:offset])
    class Response(io.BytesIO):
        status=206 if offset and range_supported else 200
        headers={'Content-Range': f'bytes {offset}-{len(content)-1}/{len(content)}'}
    def open_url(request,timeout):
        assert timeout>0
        assert request.get_header('Range')==(f'bytes={offset}-' if offset else None)
        return Response(content[offset:] if offset and range_supported else content)
    monkeypatch.setattr('urllib.request.urlopen',open_url)
    sha=hashlib.sha256(content).hexdigest()
    assert download_file('https://example.invalid/fixture',target,sha)==str(target)
    assert target.read_bytes()==content and not target.with_suffix('.bin.part').exists()
    # This is a mocked network boundary test, not a live publisher download.
    assert download_file('https://example.invalid/fixture',target,sha)==str(target)
    with pytest.raises(FileExistsError): download_file('https://example.invalid/fixture',target,'0'*64)


def test_prepare_protocol_validation_is_from_source_train(tmp_path,cfg):
    domains={'source':'Generator A','other':'Generator B'}; index=0
    for folder in domains.values():
        for split,n in [('train',5),('val',2)]:
            for label in ('nature','ai'):
                d=tmp_path/'GenImage'/folder/split/label;d.mkdir(parents=True)
                for k in range(n):
                    index+=1
                    Image.fromarray(np.random.default_rng(index).integers(0,256,(32,32,3),dtype=np.uint8)).save(d/f'{k}.png')
    protocol={'root':'GenImage','domains':domains,'source':'source','validation_fraction':.2,'split_seed':42}
    path=tmp_path/'protocol.yaml';path.write_text(yaml.safe_dump(protocol))
    cfg['paths'].update(data_root=str(tmp_path),manifest_dir=str(tmp_path/'manifests'))
    configs=prepare_protocol(cfg,path,all_sources=True)
    assert len(configs)==2
    for conf in configs:
        ready=load_config(conf);assert audit_config(ready,rehash=True)['passed']
        rows=read_rows(Path(ready['paths']['manifest_dir'])/ready['data']['val'])
        assert all('/train/' in row['path'] for row in rows)
        assert all('/val/' in row['path'] for item in ready['data']['tests'] for row in read_rows(Path(ready['paths']['manifest_dir'])/item['manifest']))


def test_efficiency_only_aggregate_without_fake_accuracy(tmp_path):
    save_json(tmp_path/'runs/full/seed42/benchmark.json',{
        'schema':'cadp-benchmark-v1','seed':42,'device':'cpu','samples':10,'batch_size':2,
        'batch_latency_ms_p50':1.,'batch_latency_ms_p95':2.,'throughput_images_s':1000.,
        'cuda_peak_allocated_bytes':None,'tiny_random_backbone':True})
    assert aggregate(tmp_path,tmp_path/'aggregate')==[]
    assert (tmp_path/'aggregate/benchmarks.csv').is_file()
    assert json.loads((tmp_path/'aggregate/summary.json').read_text())['results']==[]
