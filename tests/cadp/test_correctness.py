from __future__ import annotations
import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import pytest
import torch
from cadp.config import load_config, dump_config
from cadp.cli import verify_upstream, toy_data
from cadp.model import Detector, detection_loss
from cadp.backbone import QKVLoRA
from cadp.probabilistic import planar_step
from cadp.data import scan, read_rows, write_rows, split_manifest, audit, preprocess
from cadp.engine import (seed_all, train, read_checkpoint, checkpoint_payload, apply_checkpoint,
    evaluate, optimizer_step, preflight)
from cadp.metrics import binary_metrics, choose_threshold
from cadp.experiments import plan_suite, ABLATIONS, set_value


def test_upstream_unchanged():
    assert verify_upstream()['files'] == 47

@pytest.mark.parametrize('setting', ['bad=1','model.flowz=3','model.lengths=[3,1]',
    'model.attention_dim=31','train.amp=fp32','train.batch_size=0','model.eval_samples=999',
    'model.lora_dropout=0.1','train.grad_scaler_init_scale=0','loss.patch=-1'])
def test_strict_config(setting):
    with pytest.raises(ValueError): load_config(overrides=[setting])


def test_config_off_and_scientific(tmp_path):
    c=load_config(overrides=['train.amp=off','train.lr=1e-4'])
    assert c['train']['amp']=='off' and c['train']['lr']==1e-4
    dump_config(c,tmp_path/'x.yaml')
    assert load_config(tmp_path/'x.yaml')==c


def test_planar_exact_jacobian():
    torch.manual_seed(12)
    z=torch.randn(5,dtype=torch.float64,requires_grad=True)
    u,w=torch.randn(2,5,dtype=torch.float64);b=torch.randn(1,dtype=torch.float64)
    _,ld=planar_step(z,u,w,b)
    jac=torch.autograd.functional.jacobian(lambda x: planar_step(x,u,w,b)[0],z)
    torch.testing.assert_close(ld,torch.linalg.slogdet(jac).logabsdet,rtol=1e-9,atol=1e-9)
    assert torch.linalg.det(jac)>0


def test_zero_lora_matches_original():
    base=torch.nn.MultiheadAttention(32,4,dropout=0)
    lora=QKVLoRA(copy.deepcopy(base),4,.5)
    x=torch.randn(5,2,32)
    torch.testing.assert_close(base(x,x,x,need_weights=False)[0],lora(x,x,x)[0])
    before={k:v.clone() for k,v in lora.base.state_dict().items()}
    for _ in range(3): lora.train();lora.eval();lora(x,x,x)
    for k,v in lora.base.state_dict().items(): assert torch.equal(v,before[k])


def test_prompt_pair_and_causal_context(cfg):
    model=Detector(cfg).eval(); b=model.backbone
    bs,bp,_=model.probabilistic(torch.randn(2,b.joint_dim),2)
    for r in cfg['model']['lengths']:
        p=model.build_prompts(bs,bp,r)
        assert p.shape==(2,2,2,2,77,32)
        assert torch.equal(p[:,:,:,0,5:5+3+r],p[:,:,:,1,5:5+3+r])
        torch.testing.assert_close(p[0,0,0,0,5+3+r],model.word_embeddings[6])
        assert not torch.equal(p[:,:,:,0,2],p[:,:,:,1,2])
        _,ctx=b.encode_text_embeddings(p.reshape(-1,77,32),3+r)
        assert not torch.equal(ctx[0],ctx[1])


def test_fixed_noise_batch_and_pair_average(cfg):
    model=Detector(cfg).eval(); x=torch.randn(3,3,28,28)
    with torch.no_grad():
        a=model(x);torch.randn(100)
        b=model(x); singles=torch.cat([model(t[None])['probabilities'] for t in x])
    torch.testing.assert_close(a['probabilities'],b['probabilities'],atol=0,rtol=0)
    torch.testing.assert_close(a['probabilities'],singles,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(a['probabilities'],a['pair_probabilities'].mean((1,2)))
    torch.testing.assert_close(a['probabilities'].sum(-1),torch.ones(3))
    assert set(a['private_lengths'].tolist())<={1,3,5,7}


def test_gradients_and_labels_only_in_loss(cfg):
    seed_all(1);model=Detector(cfg).train();x=torch.randn(2,3,28,28)
    output=model(x);loss,terms=detection_loss(output,torch.tensor([0,1]),cfg['loss']);loss.backward()
    assert torch.isfinite(loss) and set(terms)=={'cls','rec','kl','patch','anchor','length'}
    for name in ['class_residual','prompt_shared','prompt_private','length_router.0.weight',
                 'cross_attention.q.weight','probabilistic.private_stats.0.weight']:
        grad=dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,name
    assert any('lora_B' in n and p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters())
    assert all(p.grad is None for n,p in model.backbone.clip.named_parameters() if 'lora_' not in n)
    with pytest.raises(TypeError): model(x,labels=torch.tensor([0,1]))


@pytest.mark.parametrize('name', list(ABLATIONS))
def test_every_ablation_forward_backward(cfg,name):
    for key,value in ABLATIONS[name].items(): set_value(cfg,key,value)
    model=Detector(cfg).train();o=model(torch.randn(2,3,28,28))
    loss,_=detection_loss(o,torch.tensor([0,1]),cfg['loss']);loss.backward()
    assert torch.isfinite(loss),name
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad(): assert model(torch.randn(1,3,28,28))['probabilities'].shape==(1,2)


def test_attention_export(cfg):
    model=Detector(cfg).eval()
    with torch.no_grad(): out=model(torch.randn(1,3,28,28),samples=1,return_attention=True)
    for value in out['attention'].values():
        a=value['attention'][0]
        assert a.shape[-1]==16 # 28 / 7, squared
        torch.testing.assert_close(a.sum(-1),torch.ones_like(a.sum(-1)),atol=1e-6,rtol=1e-6)


def test_checkpoint_contract(cfg):
    m=Detector(cfg);p=checkpoint_payload(m,cfg)
    assert not any(k.startswith('backbone.clip.') and 'lora_' not in k for k in p['model'])
    apply_checkpoint(Detector(cfg),cfg,p)
    q=copy.deepcopy(p);q['base_sha256']='wrong'
    with pytest.raises(ValueError,match='mismatch'): apply_checkpoint(m,cfg,q)
    q=copy.deepcopy(p);q['model'].pop(next(iter(q['model'])))
    with pytest.raises(ValueError,match='key mismatch'): apply_checkpoint(m,cfg,q)


def test_gradscaler_overflow_is_skipped_then_recovers():
    model=torch.nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters(),lr=.01)
    scaler=torch.amp.GradScaler('cpu',init_scale=256)
    initial=model.weight.detach().clone()
    scaler.scale(model(torch.ones(1,2)).sum()).backward()
    model.weight.grad.fill_(float('inf'))
    success,_=optimizer_step(model,opt,scaler,1.)
    assert not success and scaler.get_scale()==128
    assert torch.equal(initial,model.weight)
    opt.zero_grad(set_to_none=True)
    scaler.scale(model(torch.ones(1,2)).sum()).backward()
    success,_=optimizer_step(model,opt,scaler,1.)
    assert success and not torch.equal(initial,model.weight)


def test_fp16_autocast_real_model_cpu(cfg):
    # Kernel-level CPU simulation; not a substitute for a CUDA test.
    m=Detector(cfg).train();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad])
    scaler=torch.amp.GradScaler('cpu',init_scale=8.)
    with torch.autocast('cpu',dtype=torch.float16):
        out=m(torch.randn(2,3,28,28));loss,_=detection_loss(out,torch.tensor([0,1]),cfg['loss'])
    scaler.scale(loss).backward();success,_=optimizer_step(m,opt,scaler,1.)
    assert success


def create_image(path,seed):
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.random.default_rng(seed).integers(0,256,(32,32,3),dtype=np.uint8)).save(path)


def test_label_detection_ignores_parent_and_leakage(tmp_path):
    root=tmp_path/'real'/'dataset'
    for label in (0,1):
        for i in range(5): create_image(root/('0_real' if label==0 else '1_fake')/f'{i}.png',10*label+i)
    scan(root,tmp_path/'all.csv','train','s')
    rows=read_rows(tmp_path/'all.csv');assert sum(r['label'] for r in rows)==5
    split_manifest(tmp_path/'all.csv',tmp_path/'tr.csv',tmp_path/'v.csv',.2)
    assert audit([('train','train',tmp_path/'tr.csv'),('val','val',tmp_path/'v.csv')],root,rehash=True)['passed']
    wrong=read_rows(tmp_path/'tr.csv');wrong[0]['split']='val';wrong[1]['split']='val'
    write_rows(tmp_path/'bad.csv',[{**r,'split':'val'} for r in wrong])
    with pytest.raises(ValueError,match='leakage'): audit([('train','train',tmp_path/'tr.csv'),('val','val',tmp_path/'bad.csv')],root)


@pytest.mark.parametrize('kind',['clean','jpeg:75','blur:1','resize:0.5','noise:0.01'])
def test_corruption_shape_determinism(kind):
    image=Image.fromarray(np.random.default_rng(1).integers(0,256,(40,50,3),dtype=np.uint8))
    a=preprocess(image,28,28,False,kind,'id');b=preprocess(image,28,28,False,kind,'id')
    assert a.shape==(3,28,28) and torch.isfinite(a).all() and torch.equal(a,b)


def test_metrics_and_invalid_values():
    x=binary_metrics([0,0,1,1],[.1,.2,.8,.9]);assert x['accuracy']==1 and x['auroc']==1
    assert binary_metrics([0,0],[.2,.3])['auroc'] is None
    for p in [[.2,float('nan')],[.2,1.1]]:
        with pytest.raises(ValueError): binary_metrics([0,1],p)
    assert 0<=choose_threshold([0,0,1,1],[.1,.2,.8,.9])<=1


def test_archive_path_traversal_rejected(tmp_path):
    from cadp.assets import safe_extract
    with zipfile.ZipFile(tmp_path/'evil.zip','w') as z: z.writestr('../escape.txt','x')
    with pytest.raises(ValueError,match='traversal'): safe_extract(tmp_path/'evil.zip',tmp_path/'out')
    assert not (tmp_path/'escape.txt').exists()


def test_arrow_original_bytes_preserved(tmp_path):
    from datasets import Dataset, Image as HFImage
    from cadp.arrow import export_arrow
    raw=[]
    for i in range(4):
        buffer=io.BytesIO();Image.new('RGB',(32,32),(20*i,30,100)).save(buffer,format='JPEG',quality=73)
        raw.append(buffer.getvalue())
    ds=Dataset.from_dict({'image':[{'bytes':x,'path':None} for x in raw],'label':[0,1,0,1]}).cast_column('image',HFImage())
    ds.save_to_disk(str(tmp_path/'snapshot/data/test/sd14'))
    result=export_arrow(tmp_path/'snapshot',tmp_path/'export',fake_label=1)
    got={hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path/'export').rglob('*.jpg')}
    assert got=={hashlib.sha256(x).hexdigest() for x in raw} and result['reencoded'] is False
    assert export_arrow(tmp_path/'snapshot',tmp_path/'export',fake_label=1)==result


@pytest.mark.integration
def test_train_resume_equals_uninterrupted_and_eval(tmp_path):
    c=toy_data(tmp_path/'toy'); c['data']['hash_check']=False
    # Train and preflight must not load a missing test manifest.
    tests=c['data']['tests'];c['data']['tests']=[{'name':'not_installed','manifest':'absent.csv'}]
    preflight(c)
    train(c,stop_after_epoch=1)
    c['train']['resume']=str(Path(c['paths']['output_dir'])/'last.pt');train(c)
    resumed=read_checkpoint(c['train']['resume'])
    full=copy.deepcopy(c);full['train']['resume']=None;full['paths']['output_dir']=str(tmp_path/'full');train(full)
    uninterrupted=read_checkpoint(Path(full['paths']['output_dir'])/'last.pt')
    assert resumed['scheduler']==uninterrupted['scheduler']
    for key in resumed['model']: assert torch.equal(resumed['model'][key],uninterrupted['model'][key]),key
    c['data']['tests']=tests
    summary=evaluate(c,Path(c['paths']['output_dir'])/'best.pt')
    assert summary['tiny_random_backbone'] and summary['domains']['toy']['n']==6


def test_bf16_training(cfg,tmp_path):
    c=toy_data(tmp_path/'bf16');c['train']['amp']='bf16';c['train']['epochs']=1
    train(c);assert (Path(c['paths']['output_dir'])/'best.pt').exists()


def test_legacy_baseline_train_eval_checkpoint(cfg):
    from cadp.legacy import LegacyDetector
    cfg['model']['method']='ppm_baseline'
    m=LegacyDetector(cfg).train();x=torch.randn(2,3,28,28)
    o=m(x);loss,_=detection_loss(o,torch.tensor([0,1]),cfg['loss']);loss.backward()
    assert torch.isfinite(loss)
    m.eval()
    with torch.no_grad(): a=m(x)['probabilities'];b=m(x)['probabilities']
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    apply_checkpoint(m,cfg,checkpoint_payload(m,cfg))


def test_suite_full_job_counts(cfg,tmp_path):
    cfg['model']['tiny']=False
    jobs=plan_suite(cfg,Path(__file__).resolve().parents[2]/'configs/cadp/suite.yaml','all',tmp_path/'suite')
    assert len(jobs)==126 and sum(x['kind']=='train' for x in jobs)==45
    assert len({x['name'] for x in jobs})==126
    for j in jobs:
        c=load_config(j['config']);assert c['model']['tiny'] is False
