r"""CoraFull validation-only diagnostic; run with the existing GPU environment.
Place beside train.py and large_data.py, then:
.\.venv-gpu\Scripts\python.exe diagnose_convergence.py
No test-label evaluation. No modification of earlier results or source files.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import torch_geometric
from train import Model, seed_everything
from large_data import load_data, make_split, mask_hash

BASE=Path(__file__).resolve().parent


@torch.no_grad()
def train_validation_stats(model,data,classes):
    model.eval()
    pred=model(data.x,data.edge_index).argmax(-1)
    result={}
    # Deliberately do not evaluate any test nodes or test labels.
    for name,mask in [('train',data.train_mask),('val',data.val_mask)]:
        truth=data.y[mask]; yhat=pred[mask]
        cm=torch.bincount(truth*classes+yhat,minlength=classes*classes).reshape(classes,classes).float()
        denom=cm.sum(0)+cm.sum(1)
        macro=torch.where(denom>0,2*cm.diag()/denom,torch.zeros_like(denom)).mean()
        result[name+'_accuracy']=float((truth==yhat).float().mean())
        result[name+'_macro_f1']=float(macro)
        result[name+'_predicted_classes']=int(yhat.unique().numel())
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=2000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--models',nargs='+',choices=['gcn','fixed'],default=['gcn','fixed'])
    p.add_argument('--weight-decays',type=float,nargs='+',default=[.0005,.00005])
    a=p.parse_args()
    if a.epochs<1 or any(w<0 for w in a.weight_decays):p.error('Invalid epochs or weight decay')
    torch.set_num_threads(2)
    if a.device=='cuda':
        if not torch.cuda.is_available():raise SystemExit('Use your .venv-gpu Python: CUDA unavailable.')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    data,classes=load_data('CoraFull')
    tr,va,te,counts=make_split(data.y,2026,1.)
    data.train_mask=tr;data.val_mask=va
    masks={'train':mask_hash(tr),'val':mask_hash(va),'test':mask_hash(te)}
    del te
    data=data.to(a.device)
    root=BASE/('convergence_diagnostics_'+time.strftime('%Y%m%d_%H%M%S'))
    root.mkdir(exist_ok=False)
    print(f'VALIDATION ONLY: {len(a.models)*len(a.weight_decays)} runs; no test evaluation.',flush=True)
    runs=[]
    for name in a.models:
        for wd in a.weight_decays:
            seed_everything(a.seed)
            model=Model(name,data.num_features,classes,hidden=64).to(a.device)
            optimizer=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=wd)
            best=-1.;best_epoch=0;best_stats=None;history=[]
            if a.device=='cuda':torch.cuda.synchronize()
            start=time.perf_counter()
            print(f'\nMODEL={name} WEIGHT_DECAY={wd} SEED={a.seed}',flush=True)
            for epoch in range(1,a.epochs+1):
                model.train();optimizer.zero_grad(set_to_none=True)
                logits=model(data.x,data.edge_index)
                loss=F.cross_entropy(logits[data.train_mask],data.y[data.train_mask])
                if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
                loss.backward();optimizer.step()
                # Evaluate validation each epoch, as in the original experiment.
                # Save compact traces every 20 epochs, plus first/last.
                stats=train_validation_stats(model,data,classes)
                if stats['val_accuracy']>best:
                    best=stats['val_accuracy'];best_epoch=epoch;best_stats=stats.copy()
                if epoch==1 or epoch%20==0 or epoch==a.epochs:
                    row=dict(epoch=epoch,training_dropout_loss=loss.item(),**stats)
                    history.append(row)
                    print(f'Epoch {epoch:04d} | loss={loss.item():.4f} | train={stats["train_accuracy"]:.4f} | val={stats["val_accuracy"]:.4f} | val_macro_f1={stats["val_macro_f1"]:.4f} | predicted_classes={stats["val_predicted_classes"]}/{classes}',flush=True)
            if a.device=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            trace=root/f'{name}_wd{wd:g}_history.csv'
            with trace.open('w',newline='',encoding='utf-8-sig') as f:
                writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
            runs.append(dict(model=name,weight_decay=wd,seed=a.seed,max_epochs=a.epochs,learning_rate=.01,
                best_epoch=best_epoch,best_validation_accuracy=best,best_checkpoint_stats=best_stats,
                last_checkpoint_stats=stats,seconds=elapsed,history=history))
            result=dict(purpose='validation-only convergence and regularization diagnostic; NOT final test results',
                dataset='CoraFull',split_seed=2026,train_fraction=1.,hidden_output_features=64,
                torch=torch.__version__,pyg=torch_geometric.__version__,python=sys.version,device=a.device,
                gpu=torch.cuda.get_device_name(0) if a.device=='cuda' else None,split_hashes=masks,
                completed_runs=runs,expected_runs=len(a.models)*len(a.weight_decays),test_evaluated=False,
                limitations='One initialization seed and one split; candidates prespecified. Validation selection is not proof of generalization.')
            (root/'diagnostic_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(f'COMPLETED: {name} wd={wd:g} best_val={best:.4f} at epoch {best_epoch}; {elapsed:.1f}s',flush=True)
            del optimizer,model,logits,loss
            if a.device=='cuda':torch.cuda.empty_cache()
    print(f'ALL FINISHED. Send this file: {root/"diagnostic_summary.json"}',flush=True)


if __name__=='__main__':main()
