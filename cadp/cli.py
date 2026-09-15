"""Command-line entry point. `python -m cadp.cli --help` lists every command."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .config import load_config, dump_config


def parser():
    p=argparse.ArgumentParser(description='Class-Anchored Dynamic Probabilistic CLIP (CADP-CLIP)')
    subs=p.add_subparsers(dest='command',required=True)
    names=['write-config','verify-upstream','doctor','download-model','download-data','extract','scan','split','prepare-protocol',
           'export-arrow','audit','preflight','train','evaluate','predict','calibrate','benchmark','suite','aggregate','toy-data']
    for name in names:
        s=subs.add_parser(name)
        s.add_argument('--config',type=Path)
        s.add_argument('--set',dest='overrides',action='append',default=[],metavar='KEY=VALUE',help='Strict YAML dotted override; repeat for more settings')
        if name in ('write-config','scan','predict','calibrate','benchmark','aggregate','toy-data','export-arrow'):
            s.add_argument('--output',required=True,type=Path)
        elif name in ('evaluate','suite'):
            s.add_argument('--output',type=Path,required=name=='suite')
        if name in ('evaluate','predict','calibrate','benchmark'):
            s.add_argument('--checkpoint',required=True,type=Path)
        if name in ('predict','extract'):
            s.add_argument('--input',required=True,type=Path)
        if name=='predict': s.add_argument('--attention',action='store_true')
        if name=='extract': s.add_argument('--destination',required=True,type=Path)
        if name=='download-data':
            s.add_argument('--assets',type=Path,default=Path('configs/cadp/downloads.yaml'))
            s.add_argument('--execute',action='store_true',help='Without this flag, prints a download plan only')
        if name=='scan':
            s.add_argument('--root',required=True,type=Path); s.add_argument('--source',required=True)
            s.add_argument('--split',required=True,choices=['train','val','test'])
            s.add_argument('--relative-to',type=Path)
        if name=='split':
            s.add_argument('--input',required=True,type=Path)
            s.add_argument('--train-output',required=True,type=Path);s.add_argument('--val-output',required=True,type=Path)
            s.add_argument('--fraction',type=float,default=.1)
        if name=='prepare-protocol':
            s.add_argument('--protocol',type=Path,default=Path('configs/cadp/genimage.yaml'))
            s.add_argument('--source');s.add_argument('--all-sources',action='store_true')
        if name=='audit': s.add_argument('--rehash',action='store_true')
        if name=='train': s.add_argument('--stop-after-epoch',type=int,help='Intentional epoch-boundary interruption for resume tests; not used by full experiments')
        if name=='benchmark':
            s.add_argument('--warmup',type=int,default=3);s.add_argument('--repeats',type=int,default=20)
        if name=='suite':
            s.add_argument('--suite',type=Path,default=Path('configs/cadp/suite.yaml'))
            s.add_argument('--group',choices=['main','ablations','robustness','mc','efficiency','cross-source','all'],default='main')
            s.add_argument('--execute',action='store_true');s.add_argument('--resume',action='store_true')
        if name=='aggregate': s.add_argument('--root',required=True,type=Path)
        if name=='export-arrow':
            s.add_argument('--snapshot',required=True,type=Path)
            s.add_argument('--split',choices=['train','val','test'],default='test')
            s.add_argument('--generator')
            s.add_argument('--fake-label',type=int,choices=[0,1],required=True)

    return p


def verify_upstream():
    from .backbone import file_sha256
    root=Path(__file__).resolve().parents[1]
    provenance=json.loads((root/'provenance/upstream.json').read_text())
    upstream = root/'PPM_CLIP'
    bad=[p for p,h in provenance['sha256'].items() if not (upstream/p).is_file() or file_sha256(upstream/p)!=h]
    if bad: raise ValueError(f'Original PPM-CLIP files changed: {bad}')
    result={'passed':True,'files':len(provenance['sha256']),'upstream_commit':provenance['commit'], 'checked_root':str(upstream)}
    print(json.dumps(result));return result


def doctor(cfg):
    import importlib
    import platform
    import torch
    import torchvision
    modules={}
    for name in ('yaml','PIL','numpy','sklearn','ftfy','regex','clip'):
        module=importlib.import_module(name);modules[name]=str(getattr(module,'__version__','import OK'))
    value={'python':platform.python_version(),'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),
           'cuda_available':torch.cuda.is_available(),'cuda_runtime':torch.version.cuda,
           'gpu_count':torch.cuda.device_count(),'requested_device':cfg['train']['device'],
           'checkpoint_exists':Path(cfg['paths']['clip_checkpoint']).is_file(),'imports':modules}
    print(json.dumps(value,indent=2));return value


def toy_data(destination):
    """Synthetic random images exercise infrastructure, not detection accuracy."""
    import numpy as np
    from PIL import Image
    from .data import scan
    out=Path(destination).resolve();rng=np.random.default_rng(12345)
    for split,count in [('train',6),('val',3),('test',3)]:
        for label in (0,1):
            d=out/'images'/split/('0_real' if label==0 else '1_fake');d.mkdir(parents=True,exist_ok=True)
            for i in range(count):
                Image.fromarray(rng.integers(0,256,(32,32,3),dtype=np.uint8)).save(d/f'{i}.png')
        scan(out/'images'/split,out/'manifests'/f'{split}.csv',split,'toy-random')
    c=load_config(overrides=['model.tiny=true','model.attention_dim=32','model.hidden_dim=32',
                            'model.text_chunk_size=8','train.device=cpu','train.amp=off','data.workers=0',
                            'data.min_size=28','train.batch_size=2','train.accumulation_steps=3','train.epochs=2'])
    c['paths'].update(data_root=str(out/'images'),manifest_dir=str(out/'manifests'),output_dir=str(out/'run'))
    c['data']['tests']=[{'name':'toy','manifest':'test.csv'}]
    dump_config(c,out/'toy.yaml');print(out/'toy.yaml');return c


def main(argv=None):
    a=parser().parse_args(argv);c=load_config(a.config,a.overrides)
    cmd=a.command
    if cmd=='write-config': dump_config(c,a.output)
    elif cmd=='verify-upstream': verify_upstream()
    elif cmd=='doctor': doctor(c)
    elif cmd=='toy-data': toy_data(a.output)
    elif cmd=='export-arrow':
        from .arrow import export_arrow
        print(json.dumps(export_arrow(a.snapshot,a.output,a.split,a.generator,a.fake_label)))
    elif cmd in ('download-model','download-data','extract','prepare-protocol'):
        from . import assets
        if cmd=='download-model': print(assets.download_model(c))
        elif cmd=='download-data': assets.download_data(c,a.assets,a.execute)
        elif cmd=='extract': print(assets.safe_extract(a.input,a.destination))
        else: assets.prepare_protocol(c,a.protocol,a.source,a.all_sources)
    elif cmd in ('scan','split'):
        from .data import scan,split_manifest
        if cmd=='scan': result=scan(a.root,a.output,a.split,a.source,a.relative_to)
        else: result=split_manifest(a.input,a.train_output,a.val_output,a.fraction,c['seed'],c['paths']['data_root'])
        print(json.dumps(result))
    elif cmd in ('audit','preflight','train','evaluate'):
        from . import engine
        if cmd=='audit': print(json.dumps(engine.audit_config(c,rehash=a.rehash),ensure_ascii=False,indent=2))
        elif cmd=='preflight': engine.preflight(c)
        elif cmd=='train': engine.train(c,stop_after_epoch=a.stop_after_epoch)
        else: engine.evaluate(c,a.checkpoint,a.output)
    elif cmd in ('predict','calibrate','benchmark'):
        from . import inference
        if cmd=='predict': inference.predict(c,a.checkpoint,a.input,a.output,a.attention)
        elif cmd=='calibrate': inference.calibrate(c,a.checkpoint,a.output)
        else: inference.benchmark(c,a.checkpoint,a.output,a.warmup,a.repeats)
    elif cmd in ('suite','aggregate'):
        from . import experiments
        if cmd=='aggregate': experiments.aggregate(a.root,a.output)
        else:
            jobs=experiments.plan_suite(c,a.suite,a.group,a.output)
            if a.execute: experiments.execute_suite(jobs,a.output,a.resume)

if __name__=='__main__':
    main()
