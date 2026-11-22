"""Real-data training. CPU execution is deliberately opt-in; default is CUDA."""
import argparse
import copy
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import torch_geometric
from train import Model, seed_everything, delete_edges
from large_data import load_data, make_split, mask_hash
BASE=Path(__file__).resolve().parent


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


@torch.no_grad()
def evaluate(model,data,edges,mask,classes):
    model.eval()
    pred=model(data.x,edges).argmax(-1)[mask]
    truth=data.y[mask]
    cm=torch.bincount(truth*classes+pred,minlength=classes*classes).reshape(classes,classes).float()
    denom=cm.sum(0)+cm.sum(1)
    f1=torch.where(denom>0,2*cm.diag()/denom,torch.zeros_like(denom)).mean()
    return float((pred==truth).float().mean()),float(f1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['CoraFull','Cora'],default='CoraFull')
    p.add_argument('--model',choices=['mlp','gcn','gat','gated','fixed'],default='gcn')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--split-seed',type=int,default=2026)
    p.add_argument('--train-fraction',type=float,default=1.0)
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--rates',type=float,nargs='+',default=[0,.1,.3,.5])
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.epochs<1 or a.repeats<1 or not 0<a.train_fraction<=1 or any(not 0<=r<=1 for r in a.rates):
        p.error('Invalid epochs, repeats, fraction or rates')
    torch.set_num_threads(2)
    if a.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: use your .venv-gpu interpreter.')
    # Avoid TF32 as an additional experiment-level difference.
    if a.device=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.cuda.reset_peak_memory_stats()
    seed_everything(a.seed)
    data,classes=load_data(a.dataset)
    tr,va,te,counts=make_split(data.y,a.split_seed,a.train_fraction)
    data.train_mask,data.val_mask,data.test_mask=tr,va,te
    split_hashes={k:mask_hash(v) for k,v in [('train',tr),('val',va),('test',te)]}
    a.out.mkdir(parents=True,exist_ok=True)
    # Explicit split indices make the protocol independently inspectable.
    write_csv(a.out/'class_counts.csv',counts)
    torch.save({'train':tr.nonzero().view(-1),'validation':va.nonzero().view(-1),'test':te.nonzero().view(-1)},a.out/'split_indices.pt')
    metadata=dict(dataset=a.dataset,nodes=data.num_nodes,edge_entries=data.num_edges,features=data.num_features,
                  classes=classes,train_nodes=int(tr.sum()),validation_nodes=int(va.sum()),test_nodes=int(te.sum()),split_hashes=split_hashes)
    print(json.dumps(metadata),flush=True)
    data=data.to(a.device)
    # All models now have 64 hidden output features; GAT retains 8 heads x 8.
    model=Model(a.model,data.num_features,classes,hidden=64).to(a.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.005 if a.model=='gat' else .01,weight_decay=5e-4)
    history=[]; best=-1.; best_epoch=0; state=None
    if a.device=='cuda': torch.cuda.synchronize()
    start=time.perf_counter()
    for epoch in range(1,a.epochs+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits=model(data.x,data.edge_index)
        loss=F.cross_entropy(logits[data.train_mask],data.y[data.train_mask])
        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
        loss.backward(); optimizer.step()
        # Test labels are never queried for checkpoint selection.
        val,_=evaluate(model,data,data.edge_index,data.val_mask,classes)
        if val>best:
            best=val; best_epoch=epoch; state=copy.deepcopy(model.state_dict())
        history.append(dict(epoch=epoch,train_loss=loss.item(),validation_accuracy=val))
        if epoch==1 or epoch%20==0 or epoch==a.epochs:
            print(f'Epoch {epoch:03d} | loss={loss.item():.4f} | val={val:.4f}',flush=True)
    model.load_state_dict(state)
    clean,clean_f1=evaluate(model,data,data.edge_index,data.test_mask,classes)
    # Delete on CPU once per repeat/rate; use identical graph draws for every
    # model, label budget and training seed. The perturbation factor is crossed.
    original=data.edge_index.cpu()
    rows=[]
    for rate in sorted(set([0.]+a.rates)):
        for repeat in range(1 if rate==0 else a.repeats):
            perturb_seed=700000+repeat
            edges=data.edge_index if rate==0 else delete_edges(original,rate,perturb_seed).to(a.device)
            acc,f1=(clean,clean_f1) if rate==0 else evaluate(model,data,edges,data.test_mask,classes)
            rows.append(dict(dataset=a.dataset,model=a.model,seed=a.seed,split_seed=a.split_seed,
                train_fraction=a.train_fraction,train_nodes=metadata['train_nodes'],drop_rate=rate,repeat=repeat,
                perturb_seed=perturb_seed,edge_entries=edges.shape[1],test_accuracy=acc,macro_f1=f1,
                clean_accuracy=clean,drop_pp=(clean-acc)*100,relative_drop_pct=(clean-acc)/clean*100 if clean else None))
    if a.device=='cuda': torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    write_csv(a.out/'history.csv',history)
    write_csv(a.out/'results.csv',rows)
    torch.save({k:v.cpu() for k,v in state.items()},a.out/'best_model.pt')
    config=vars(a).copy(); config['out']=str(a.out.resolve())
    source_hash=hashlib.sha256(b''.join((BASE/n).read_bytes() for n in ['train_large.py','large_data.py','train.py'])).hexdigest()
    summary=dict(config=config,data=metadata,python=sys.version,platform=platform.platform(),torch=torch.__version__,
        pyg=torch_geometric.__version__,device=a.device,gpu=torch.cuda.get_device_name(0) if a.device=='cuda' else None,
        hidden_output_features=64,parameters=sum(v.numel() for v in model.parameters()),best_epoch=best_epoch,
        best_validation_accuracy=best,clean_test_accuracy=clean,clean_macro_f1=clean_f1,seconds=elapsed,
        peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9 if a.device=='cuda' else None,source_sha256=source_hash,
        protocol='stratified 60/20/20, fixed split seed; nested training label budgets; full-graph transductive; clean train/validation; paired test-time edge deletion')
    # This file is the completion marker, written only after all artifacts.
    tmp=a.out/'summary.json.tmp'
    tmp.write_text(json.dumps(summary,indent=2),encoding='utf-8'); tmp.replace(a.out/'summary.json')
    print(f'FINISHED | best_epoch={best_epoch} | clean_accuracy={clean:.4f} | macro_f1={clean_f1:.4f} | seconds={elapsed:.1f}',flush=True)
    print(f'Output: {a.out.resolve()}',flush=True)


if __name__=='__main__': main()
