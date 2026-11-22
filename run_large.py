"""Resumable GPU suite. No training occurs until this script is executed."""
import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,'reconfigure'): stream.reconfigure(encoding='utf-8',errors='backslashreplace')
BASE=Path(__file__).resolve().parent


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize(root,spec):
    raw=[]; runs=[]
    for fraction in spec['train_fractions']:
        for model in spec['models']:
            for seed in spec['seeds']:
                folder=root/run_name(model,seed,fraction)
                meta=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
                with (folder/'results.csv').open(encoding='utf-8-sig') as f: part=list(csv.DictReader(f))
                assert len(part)==1+(len(spec['rates'])-1)*spec['repeats']
                cfg=meta['config']
                assert (cfg['dataset'],cfg['model'],cfg['seed'],cfg['train_fraction'])==(spec['dataset'],model,seed,fraction)
                assert meta['source_sha256']==spec['source_sha256']
                runs.append(meta);raw.extend(part)
    assert len({(r['torch'],r['pyg'],r['device'],r['gpu']) for r in runs})==1
    assert len({(r['data']['split_hashes']['val'],r['data']['split_hashes']['test']) for r in runs})==1
    # Masks must match within a label budget across all models and initializations.
    for fraction in spec['train_fractions']:
        assert len({r['data']['split_hashes']['train'] for r in runs if r['config']['train_fraction']==fraction})==1
    seed_rows=[]
    for fraction in spec['train_fractions']:
        for model in spec['models']:
            for seed in spec['seeds']:
                for rate in spec['rates']:
                    part=[r for r in raw if float(r['train_fraction'])==fraction and r['model']==model and int(r['seed'])==seed and float(r['drop_rate'])==rate]
                    assert len(part)==(1 if rate==0 else spec['repeats'])
                    if model=='mlp': assert all(float(r['test_accuracy'])==float(r['clean_accuracy']) for r in part)
                    seed_rows.append(dict(dataset=spec['dataset'],train_fraction=fraction,train_nodes=int(part[0]['train_nodes']),model=model,
                        seed=seed,drop_rate=rate,accuracy_pct=statistics.mean(float(r['test_accuracy'])*100 for r in part),
                        macro_f1_pct=statistics.mean(float(r['macro_f1'])*100 for r in part),
                        drop_pp=statistics.mean(float(r['drop_pp']) for r in part),
                        relative_drop_pct=statistics.mean(float(r['relative_drop_pct']) for r in part) if all(r['relative_drop_pct'] for r in part) else None))
    aggregate=[]
    for fraction in spec['train_fractions']:
        for model in spec['models']:
            for rate in spec['rates']:
                part=[r for r in seed_rows if r['train_fraction']==fraction and r['model']==model and r['drop_rate']==rate]
                row=dict(dataset=spec['dataset'],train_fraction=fraction,train_nodes=part[0]['train_nodes'],model=model,drop_rate=rate,seeds=len(part))
                for key in ['accuracy_pct','macro_f1_pct','drop_pp','relative_drop_pct']:
                    vals=[r[key] for r in part]
                    row[key+'_mean']=statistics.mean(vals) if all(v is not None for v in vals) else None
                    row[key+'_sd']=statistics.stdev(vals) if len(vals)>1 and all(v is not None for v in vals) else None
                aggregate.append(row)
    write_csv(root/'all_results.csv',raw);write_csv(root/'seed_means.csv',seed_rows);write_csv(root/'aggregate.csv',aggregate)
    paired=[]
    if 'gated' in spec['models'] and 'fixed' in spec['models']:
        for fraction in spec['train_fractions']:
            for rate in spec['rates']:
                diffs=[]
                for seed in spec['seeds']:
                    p=[r for r in seed_rows if r['train_fraction']==fraction and r['drop_rate']==rate and r['seed']==seed]
                    diffs.append(next(r['accuracy_pct'] for r in p if r['model']=='gated')-next(r['accuracy_pct'] for r in p if r['model']=='fixed'))
                paired.append(dict(train_fraction=fraction,drop_rate=rate,gated_minus_fixed_pp_mean=statistics.mean(diffs),
                    paired_seed_difference_sd=statistics.stdev(diffs) if len(diffs)>1 else None,seeds=len(diffs)))
        write_csv(root/'gating_paired_differences.csv',paired)
    lines=[f'# {spec["dataset"]}: test-time edge deletion','',
        'Mean accuracy (%) +/- sample SD across training initialization seeds.',
        'Perturbed-graph evaluations are first averaged within each trained model.',
        'Graph draws are shared across models, label budgets and training seeds.',
        'The SD describes training-seed variability conditional on ONE fixed split and these graph draws; it is not a confidence interval.','']
    for fraction in spec['train_fractions']:
        n=next(r['train_nodes'] for r in aggregate if r['train_fraction']==fraction)
        lines += [f'## Training label budget: {fraction:g} of full training pool ({n} nodes)','',
                  '| Model | '+' | '.join(f'{r*100:g}% edges deleted' for r in spec['rates'])+' |',
                  '|---|'+'---:|'*len(spec['rates'])]
        for model in spec['models']:
            vals=[]
            for rate in spec['rates']:
                r=next(r for r in aggregate if r['train_fraction']==fraction and r['model']==model and r['drop_rate']==rate)
                sd=r['accuracy_pct_sd'];vals.append(f'{r["accuracy_pct_mean"]:.2f} +/- {sd:.2f}' if sd is not None else f'{r["accuracy_pct_mean"]:.2f} (n=1)')
            lines.append('| '+model+' | '+' | '.join(vals)+' |')
        lines.append('')
    lines += ['All test scores are exploratory; checkpoint selection uses validation accuracy only.',
        'CoraFull has 70 classes; do not interpret changes from 7-class Cora as pure sample-size effects.',
        'Within this suite label-budget comparisons hold the graph, validation/test nodes, and model architecture fixed.',
        'Unused training labels are excluded from loss, but their node features/edges remain visible (transductive setting).',
        'Hidden output dimension is 64; GAT uses 8 heads x 8 features. Learning rate/dropout follow the previous baseline defaults.',
        'Gated/fixed have additional self-feature branches; gated adds gate parameters. Models are not exactly parameter matched.',
        'More epochs/seeds/graph copies do not create additional independent documents.',
        'No result is supplied by the assistant: these files are generated by your local run.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


def run_name(model,seed,fraction):
    return f'{model}_seed{seed}_fraction{fraction:g}'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['CoraFull','Cora'],default='CoraFull')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--models',nargs='+',choices=['mlp','gcn','gat','gated','fixed'],default=['mlp','gcn','gat','gated','fixed'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--split-seed',type=int,default=2026)
    p.add_argument('--train-fractions',nargs='+',type=float,default=[1.])
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--rates',nargs='+',type=float,default=[0,.1,.3,.5])
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--out',type=Path,default=BASE/'large_results')
    a=p.parse_args()
    if a.epochs<1 or a.repeats<1 or any(not 0<f<=1 for f in a.train_fractions) or any(not 0<=r<=1 for r in a.rates):p.error('Invalid parameters')
    import torch,torch_geometric
    if a.device=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA unavailable. Use .venv-gpu/Scripts/python.exe')
    spec=vars(a).copy();spec.pop('out')
    for key in ['train_fractions','rates','seeds']:spec[key]=sorted(set(spec[key]))
    spec['rates']=sorted(set([0.]+spec['rates']));spec['models']=list(dict.fromkeys(spec['models']))
    spec['source_sha256']=hashlib.sha256(b''.join((BASE/n).read_bytes() for n in ['train_large.py','large_data.py','train.py'])).hexdigest()
    spec['environment']=dict(python=platform.python_version(),torch=torch.__version__,pyg=torch_geometric.__version__,
        gpu=torch.cuda.get_device_name(0) if a.device=='cuda' else None)
    root=a.out.resolve();root.mkdir(parents=True,exist_ok=True)
    plan=root/'suite_spec.json'
    if plan.exists():
        if json.loads(plan.read_text(encoding='utf-8'))!=spec:raise SystemExit('This output folder has another configuration/environment/source. Choose a new --out folder.')
    else:plan.write_text(json.dumps(spec,indent=2),encoding='utf-8')
    tasks=[(f,m,s) for f in spec['train_fractions'] for m in spec['models'] for s in spec['seeds']]
    print(f'{len(tasks)} training runs. Output: {root}',flush=True)
    env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'}
    for i,(fraction,model,seed) in enumerate(tasks,1):
        dest=root/run_name(model,seed,fraction)
        marker=dest/'summary.json'
        if marker.exists() and all((dest/n).exists() for n in ['results.csv','best_model.pt','history.csv','split_indices.pt']):
            print(f'[{i}/{len(tasks)}] SKIP completed: {dest.name}',flush=True);continue
        dest.mkdir(exist_ok=True)
        print(f'[{i}/{len(tasks)}] RUN {dest.name}',flush=True)
        cmd=[sys.executable,'-X','utf8',str(BASE/'train_large.py'),'--dataset',a.dataset,'--model',model,'--seed',str(seed),
            '--split-seed',str(a.split_seed),'--train-fraction',str(fraction),'--epochs',str(a.epochs),'--device',a.device,
            '--rates',*[str(r) for r in spec['rates']],'--repeats',str(a.repeats),'--out',str(dest)]
        with (dest/'console.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(cmd,cwd=BASE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            try:
                for line in proc.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                code=proc.wait()
            except KeyboardInterrupt:
                proc.terminate();proc.wait();print('Interrupted. Re-run the same command to resume completed runs.');raise SystemExit(130)
        if code:raise SystemExit(f'Run failed ({code}). Log: {dest/"console.log"}. Fix the cause and repeat the same command.')
    summarize(root,spec)
    print(f'ALL FINISHED. Read: {root/"RESULTS.md"}',flush=True)


if __name__=='__main__':main()
