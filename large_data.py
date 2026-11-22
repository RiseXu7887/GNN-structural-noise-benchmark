"""Shared, deterministic data protocol; no training occurs in this module."""
import hashlib
from pathlib import Path
import torch
from torch_geometric.datasets import CoraFull, Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import is_undirected

BASE = Path(__file__).resolve().parent


def load_data(name):
    if name == 'CoraFull':
        dataset = CoraFull(str(BASE/'data'/'CoraFull'), transform=NormalizeFeatures())
    else:
        dataset = Planetoid(str(BASE/'data'), name='Cora', split='public', transform=NormalizeFeatures())
    data = dataset[0]
    if not is_undirected(data.edge_index):
        raise ValueError('Expected undirected graph')
    return data, dataset.num_classes


def make_split(labels, seed=2026, train_fraction=1.0):
    """Fixed stratified ~60/20/20 split; fraction subsamples training labels only.
    Labels in unused training positions remain unavailable to the loss.
    Nested subsets are sampled independently per class with a local RNG.
    """
    if not 0 < train_fraction <= 1:
        raise ValueError('train_fraction must be in (0,1]')
    labels=labels.cpu()
    masks=[torch.zeros(labels.numel(),dtype=torch.bool) for _ in range(3)]
    rng=torch.Generator().manual_seed(seed)
    counts=[]
    for cls in labels.unique(sorted=True):
        ids=(labels==cls).nonzero(as_tuple=False).view(-1)
        n=ids.numel()
        if n<3:
            raise ValueError(f'Class {int(cls)} has fewer than 3 nodes')
        ids=ids[torch.randperm(n,generator=rng)]
        ntrain=min(n-2,max(1,int(n*.6)))
        nval=min(n-ntrain-1,max(1,int(n*.2)))
        used=max(1,int(ntrain*train_fraction))
        masks[0][ids[:used]]=True
        masks[1][ids[ntrain:ntrain+nval]]=True
        masks[2][ids[ntrain+nval:]]=True
        counts.append(dict(class_id=int(cls),total=n,full_train=ntrain,used_train=used,validation=nval,test=n-ntrain-nval))
    assert all(not (masks[i]&masks[j]).any() for i,j in [(0,1),(0,2),(1,2)])
    return (*masks,counts)


def mask_hash(mask):
    return hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest()
