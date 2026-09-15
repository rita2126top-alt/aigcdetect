"""Full (not smoke) experiment plans and seed-wise summaries; never fabricate metrics."""
from __future__ import annotations
import copy
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import yaml
from .config import dump_config, load_config
from .engine import save_json, audit_config, manifest_records
from .backbone import file_sha256

ABLATIONS = {
    'full': {},
    'no_cross_attention': {'model.cross_attention': False},
    'fixed_private_1': {'model.fixed_length': 1},
    'fixed_private_3': {'model.fixed_length': 3},
    'fixed_private_5': {'model.fixed_length': 5},
    'fixed_private_7': {'model.fixed_length': 7},
    'no_flow': {'model.flows': 0},
    'no_patch_loss': {'loss.patch': 0.0},
    'fixed_class_anchors': {'model.anchor_trainable': False, 'loss.anchor': 0.0},
    'no_length_penalty': {'loss.length': 0.0},
    'one_repository': {'model.repositories': 1},
    'no_vision_lora': {'model.lora_rank': 0},
    'deterministic_latent': {'model.stochastic': False},
    'no_reconstruction': {'loss.rec': 0.0},
    'no_kl': {'loss.kl': 0.0},
}


def set_value(cfg, dotted, value):
    section, name = dotted.split('.')
    cfg[section][name] = value


def plan_suite(base, suite_yaml, group, destination):
    if base['model']['method'] != 'cadp':
        raise ValueError('Ablation/MC suites target CADP; run the original PPM baseline separately with the matched data YAML.')
    suite = yaml.safe_load(Path(suite_yaml).read_text())
    root = Path(destination).resolve(); root.mkdir(parents=True, exist_ok=True)
    seeds = suite.get('seeds', [42,43,44])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Experiment seeds must be nonempty and unique')
    jobs=[]
    def add(kind, cfg, name, checkpoint=None, output=None):
        conf = root/'configs'/f'{name}.yaml'; dump_config(cfg, conf)
        command=[sys.executable, '-m', 'cadp.cli', kind, '--config', str(conf)]
        if checkpoint:
            command += ['--checkpoint', str(checkpoint)]
        if output:
            command += ['--output', str(output)]
        identity = hashlib.sha256(json.dumps({'kind':kind,'config':cfg,'checkpoint':str(checkpoint)}, sort_keys=True).encode()).hexdigest()
        jobs.append({'name': name, 'kind': kind, 'config':str(conf), 'command':command,
                     'identity':identity, 'output':str(output or cfg['paths']['output_dir'])})
    variants = suite.get('ablations', list(ABLATIONS)) if group in ('all','ablations') else ['full']
    if any(v not in ABLATIONS for v in variants):
        raise ValueError('Unrecognized ablation name in suite YAML')
    if group in ('all','main','ablations'):
        for variant in variants:
            for seed in seeds:
                c=copy.deepcopy(base); c['seed']=int(seed); c['train']['resume']=None
                for k,v in ABLATIONS[variant].items(): set_value(c,k,v)
                run=root/'runs'/variant/f'seed{seed}'; c['paths']['output_dir']=str(run)
                add('train',c,f'{variant}_seed{seed}_train')
                add('evaluate',c,f'{variant}_seed{seed}_clean',run/'best.pt',run/'eval_clean')
    if group in ('all','robustness','mc','efficiency'):
        for seed in seeds:
            c=copy.deepcopy(base); c['seed']=int(seed)
            run=root/'runs'/'full'/f'seed{seed}'; c['paths']['output_dir']=str(run)
            if group in ('all','robustness'):
                for corruption in suite.get('corruptions',['jpeg:65','jpeg:30','blur:3','blur:5','resize:0.5','resize:0.285714','noise:0.01']):
                    v=copy.deepcopy(c); v['eval']['corruption']=corruption
                    name=corruption.replace(':','_')
                    add('evaluate',v,f'full_seed{seed}_{name}',run/'best.pt',run/f'eval_{name}')
            if group in ('all','mc'):
                for samples in suite.get('mc_samples',[1,5,10,20]):
                    v=copy.deepcopy(c); v['model']['eval_samples']=int(samples)
                    add('evaluate',v,f'full_seed{seed}_mc{samples}',run/'best.pt',run/f'eval_mc{samples}')
            if group in ('all','efficiency'):
                add('benchmark',c,f'full_seed{seed}_benchmark',run/'best.pt',run/'benchmark.json')
    if group == 'cross-source':
        configs=suite.get('cross_source_configs',[])
        if not configs:
            raise ValueError('Set suite.cross_source_configs to YAML files produced by prepare-protocol --all-sources.')
        for source_path in configs:
            source_path=Path(source_path).expanduser(); source=source_path.stem
            for seed in seeds:
                c=load_config(source_path); c['seed']=int(seed); c['train']['resume']=None
                run=root/'cross_source'/source/f'seed{seed}'; c['paths']['output_dir']=str(run)
                add('train',c,f'cross_{source}_seed{seed}_train')
                add('evaluate',c,f'cross_{source}_seed{seed}_test',run/'best.pt',run/'evaluation')
    save_json(root/f'plan_{group}.json',{'group':group,'jobs':jobs,
              'training_epochs_per_job':base['train']['epochs'], 'seeds':seeds,
              'note':'all excludes the optional cross-source matrix; all runs use full manifests, no sample caps.'})
    print(json.dumps({'plan':str(root/f'plan_{group}.json'),'jobs':len(jobs),
                      'training_jobs':sum(j['kind']=='train' for j in jobs)},ensure_ascii=False))
    return jobs


def execute_suite(jobs, destination, resume=False):
    root=Path(destination).resolve(); status_dir=root/'status'; status_dir.mkdir(parents=True,exist_ok=True)
    source_digest=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest()
    audited = set()
    for index, job in enumerate(jobs):
        c=load_config(job['config'])
        audit_key=json.dumps({'data':c['data'], 'root':c['paths']['data_root'], 'manifests':c['paths']['manifest_dir']},sort_keys=True)
        if audit_key not in audited:
            audit_config(c);audited.add(audit_key)
        data_digest = {name:file_sha256(path) for _,name,path in manifest_records(c)}
        dependency = None
        if '--checkpoint' in job['command']:
            dependency = file_sha256(job['command'][job['command'].index('--checkpoint')+1])
        status=status_dir/f'{job["name"]}.json'
        old=json.loads(status.read_text()) if status.exists() else {}
        output=Path(job['output'])
        expected = output/'best.pt' if job['kind']=='train' else (output/'summary.json' if job['kind']=='evaluate' else output)
        if resume and old.get('completed') and old.get('identity')==job['identity'] and old.get('source_digest')==source_digest and old.get('dependency_sha256') == dependency and old.get('data_digest') == data_digest and expected.exists():
            print('SKIP completed',job['name'],flush=True); continue
        if job['kind']=='train' and old.get('source_digest') and old['source_digest'] != source_digest and expected.exists():
            raise ValueError('Training source code changed since this experiment; use a new experiment directory.')
        if job['kind']=='train' and (Path(c['paths']['output_dir'])/'last.pt').exists():
            if not resume:
                raise FileExistsError('Existing training output; use suite --resume or a new destination.')
            c['train']['resume']=str(Path(c['paths']['output_dir'])/'last.pt'); dump_config(c,job['config'])
        print(f'[{index+1}/{len(jobs)}] {job["name"]}',flush=True)
        log=status_dir/f'{job["name"]}.log'
        with log.open('a',encoding='utf-8') as f:
            result=subprocess.run(job['command'],stdout=f,stderr=subprocess.STDOUT)
        save_json(status,{'completed':result.returncode==0,'returncode':result.returncode,
                          'identity':job['identity'],'source_digest':source_digest,'dependency_sha256':dependency,'data_digest':data_digest,'command':job['command'],'log':str(log)})
        if result.returncode:
            raise RuntimeError(f'Experiment failed; remaining jobs were not run. Inspect {log}')
    aggregate(destination,root/'aggregate')


def aggregate_benchmarks(root, output):
    records = []
    for path in sorted(Path(root).rglob('benchmark.json')):
        value = json.loads(path.read_text())
        if value.get('schema') != 'cadp-benchmark-v1':
            continue
        records.append({'variant': path.parent.parent.name,
                        'seed': value['seed'], 'device': value['device'],
                        'samples': value['samples'], 'batch_size': value['batch_size'],
                        'latency_ms_p50': value['batch_latency_ms_p50'],
                        'latency_ms_p95': value['batch_latency_ms_p95'],
                        'throughput_images_s': value['throughput_images_s'],
                        'cuda_peak_allocated_bytes': value['cuda_peak_allocated_bytes'],
                        'tiny_random_backbone': value['tiny_random_backbone'],
                        'source': str(path)})
    if records:
        out = Path(output); out.mkdir(parents=True, exist_ok=True)
        with (out/'benchmarks.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    return records


def aggregate(root, output):
    # Domains are not mixed into pooled micro accuracy, because real samples may repeat across test domains.
    records=[]; seen=set()
    for path in sorted(Path(root).rglob('summary.json')):
        value=json.loads(path.read_text())
        if value.get('schema')!='cadp-evaluation-v1': continue
        key=(value['checkpoint_sha256'],value['corruption'],value['samples'],value['threshold'],
             json.dumps(value['audit']['manifests'],sort_keys=True))
        if key in seen: continue
        seen.add(key)
        # Layout: .../runs/<variant>/seed<N>/eval.../summary.json (cross-source is similarly organized).
        variant=path.parent.parent.parent.name
        for domain,metrics in value['domains'].items():
            records.append({'variant':variant,'seed':value['seed'],'domain':domain,'corruption':value['corruption'],
                            'samples':value['samples'],'threshold':value['threshold'],
                            **{k:metrics.get(k) for k in ('accuracy','balanced_accuracy','auroc','ap','f1','ece','brier')},
                            'summary_path':str(path)})
    benchmarks = aggregate_benchmarks(root, output)
    if not records:
        if benchmarks:
            save_json(Path(output)/'summary.json', {'results': [], 'benchmarks': benchmarks})
            return []
        raise ValueError('No completed evaluation summary.json or benchmark.json files; no results to aggregate.')
    groups={}
    for r in records:
        key=tuple(r[k] for k in ('variant','domain','corruption','samples','threshold'))
        groups.setdefault(key,[]).append(r)
    summary=[]
    for key,rows in groups.items():
        if len({r['seed'] for r in rows}) != len(rows):
            raise ValueError(f'Duplicate results for a training seed in {key}; select a clean results root.')
        item=dict(zip(('variant','domain','corruption','samples','threshold'),key))
        item['seeds']=len(rows)
        for metric in ('accuracy','balanced_accuracy','auroc','ap','f1','ece','brier'):
            vals=[r[metric] for r in rows if r[metric] is not None]
            item[metric+'_mean']=float(np.mean(vals)) if vals else None
            item[metric+'_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        summary.append(item)
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('per_seed.csv',records),('mean_std.csv',summary)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    save_json(out/'summary.json',{'results':summary,'standard_deviation':'sample std (ddof=1); null with one seed'})
    return summary
