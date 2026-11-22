"""Run all 5 models x 3 seeds, then summarize within seed before across seeds.
No extra dependencies. Use the same interpreter/environment as train.py.
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='backslashreplace')
BASE = Path(__file__).resolve().parent


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(root):
    raw, summaries = [], []
    for path in sorted(root.glob('*/results.csv')):
        with path.open(encoding='utf-8') as f:
            raw.extend(list(csv.DictReader(f)))
        summaries.append(json.loads(path.with_name('summary.json').read_text(encoding='utf-8')))
    assert len(summaries) == 15, f'Expected 15 completed runs, found {len(summaries)}'
    assert len(raw) == 150
    configs = {(s['config']['model'], s['config']['seed']) for s in summaries}
    assert len(configs) == 15
    assert len({(s['torch'], s['pyg'], s['device'], s['script_sha256']) for s in summaries}) == 1, 'Mixed environment or source code'
    seed_rows = []
    for model in ['mlp', 'gcn', 'gat', 'gated', 'fixed']:
        for seed in [42, 43, 44]:
            for rate in [0.0, .1, .3, .5]:
                part = [r for r in raw if r['model'] == model and int(r['seed']) == seed and float(r['drop_rate']) == rate]
                assert len(part) == (1 if rate == 0 else 3)
                if model == 'mlp':
                    assert all(float(r['test_accuracy']) == float(r['clean_test_accuracy']) for r in part)
                seed_rows.append(dict(model=model, seed=seed, drop_rate=rate,
                    accuracy_pct=statistics.mean(float(r['test_accuracy'])*100 for r in part),
                    drop_pp=statistics.mean(float(r['drop_percentage_points']) for r in part),
                    relative_drop_pct=statistics.mean(float(r['relative_drop'])*100 for r in part)))
    aggregate = []
    for model in ['mlp', 'gcn', 'gat', 'gated', 'fixed']:
        for rate in [0.0, .1, .3, .5]:
            part = [r for r in seed_rows if r['model'] == model and r['drop_rate'] == rate]
            row = dict(model=model, drop_rate=rate, training_seeds=3)
            for key in ['accuracy_pct', 'drop_pp', 'relative_drop_pct']:
                row[key+'_mean'] = statistics.mean(r[key] for r in part)
                row[key+'_sd'] = statistics.stdev(r[key] for r in part)
            aggregate.append(row)
    write_csv(root/'all_results.csv', raw)
    write_csv(root/'seed_means.csv', seed_rows)
    write_csv(root/'aggregate.csv', aggregate)
    lines = ['# Cora results', '', 'Mean accuracy (%) +/- sample SD across 3 training seeds.',
             'For each seed and nonzero deletion rate, first average 3 perturbed-graph evaluations.',
             'Same public split; clean training/validation; checkpoint chosen by validation accuracy.', '',
             '| Model | 0% deleted | 10% deleted | 30% deleted | 50% deleted |',
             '|---|---:|---:|---:|---:|']
    for model in ['mlp','gcn','gat','gated','fixed']:
        part = [r for r in aggregate if r['model']==model]
        vals = [f'{r["accuracy_pct_mean"]:.2f} +/- {r["accuracy_pct_sd"]:.2f}' for r in part]
        lines.append('| '+model+' | '+' | '.join(vals)+' |')
    lines += ['', 'These are exploratory comparisons, not statistical significance claims.',
              'GCN/MLP hidden size 16; GAT 8 heads x 8 dimensions. Models are not parameter matched.',
              'Gated vs fixed tests adaptive mixing; gated also adds gate parameters.',
              'Do not mix CPU/GPU or different library versions into one aggregate.', '',
              '| Model | Parameters |', '|---|---:|']
    for model in ['mlp','gcn','gat','gated','fixed']:
        s = next(s for s in summaries if s['config']['model']==model)
        lines.append(f'| {model} | {s["parameters"]} |')
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args()
    root=BASE / ('experiment_'+args.device+'_'+time.strftime('%Y%m%d_%H%M%S'))
    root.mkdir(exist_ok=False)
    start=time.perf_counter()
    for model in ['mlp','gcn','gat','gated','fixed']:
        for seed in [42,43,44]:
            print(f'\nRUN {model} seed={seed}',flush=True)
            cmd=[sys.executable,'-X','utf8',str(BASE/'train.py'),'--model',model,'--seed',str(seed),'--epochs','200',
                 '--device',args.device,'--drop-rates','0','0.1','0.3','0.5','--repeats','3','--out',str(root)]
            with (root/f'{model}_seed{seed}.log').open('w',encoding='utf-8') as log:
                result=subprocess.run(cmd,cwd=BASE,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
            if result.returncode:
                print((root/f'{model}_seed{seed}.log').read_text(encoding='utf-8'))
                raise SystemExit(result.returncode)
            print((root/f'{model}_seed{seed}.log').read_text(encoding='utf-8').splitlines()[-2],flush=True)
    summarize(root)
    print(f'ALL FINISHED in {time.perf_counter()-start:.1f}s. Output: {root}',flush=True)


if __name__=='__main__':
    main()
