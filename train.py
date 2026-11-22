"""Cora clean-graph training; optional test-time paired undirected edge deletion.
Run from the extracted folder: python train.py --model gcn --epochs 200
"""
import argparse
import copy
import csv
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import is_undirected

# Redirected Windows subprocess output otherwise defaults to cp1252.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='backslashreplace')

BASE = Path(__file__).resolve().parent


class Model(nn.Module):
    def __init__(self, name, inputs, classes, hidden=16):
        super().__init__()
        self.name = name
        if name == 'mlp':
            self.first = nn.Linear(inputs, hidden)
            self.second = nn.Linear(hidden, classes)
        elif name == 'gat':
            self.first = GATConv(inputs, 8, heads=8, dropout=0.6)
            self.second = GATConv(64, classes, heads=1, concat=False, dropout=0.6)
        else:
            # cached=False: subsequent evaluation graphs can change.
            self.first = GCNConv(inputs, hidden, cached=False)
            self.second = GCNConv(hidden, classes, cached=False)
            if name in ('gated', 'fixed'):
                self.self_branch = nn.Linear(inputs, hidden)
                if name == 'gated':
                    self.gate = nn.Linear(2 * hidden, 1)

    def forward(self, x, edges):
        p = 0.6 if self.name == 'gat' else 0.5
        x = F.dropout(x, p=p, training=self.training)
        if self.name == 'mlp':
            h = F.relu(self.first(x))
        else:
            h = self.first(x, edges)
            h = F.elu(h) if self.name == 'gat' else F.relu(h)
            if self.name in ('gated', 'fixed'):
                own = F.relu(self.self_branch(x))
                # A scalar gate per node, estimated from own and graph features.
                g = torch.sigmoid(self.gate(torch.cat([own, h], dim=-1))) if self.name == 'gated' else 0.5
                h = g * h + (1 - g) * own
        h = F.dropout(h, p=p, training=self.training)
        return self.second(h) if self.name == 'mlp' else self.second(h, edges)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def delete_edges(edges, fraction, seed):
    """Exact rounded fraction of unique pairs; never delete directions separately.
    Same seed -> nested deletion sets across fractions and shared across models.
    Self-loops are supplied by convolution layers and are not deletion targets.
    """
    pairs = edges[:, edges[0] < edges[1]].cpu()
    order = torch.randperm(pairs.size(1), generator=torch.Generator().manual_seed(seed))
    keep = order[round(fraction * pairs.size(1)):]
    pairs = pairs[:, keep]
    return torch.cat([pairs, pairs.flip(0)], dim=1).to(edges.device)


@torch.no_grad()
def accuracy(model, data, edges, mask):
    model.eval()
    pred = model(data.x, edges).argmax(dim=-1)
    return float((pred[mask] == data.y[mask]).float().mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp', 'gcn', 'gat', 'gated', 'fixed'], default='gcn')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    parser.add_argument('--drop-rates', nargs='+', type=float, default=[0.0])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--out', default='runs')
    args = parser.parse_args()
    if args.epochs < 1 or args.repeats < 1 or args.threads < 1 or any(not 0 <= p <= 1 for p in args.drop_rates):
        parser.error('epochs, repeats, threads must be positive; drop-rates must be in [0,1].')
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device == 'auto':
        device = 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use --device cpu for the first run.')
    print(f'Python={platform.python_version()} Torch={torch.__version__} PyG={torch_geometric.__version__} Device={device}', flush=True)
    dataset = Planetoid(str(BASE / 'data'), name='Cora', split='public', transform=NormalizeFeatures())
    data = dataset[0]
    assert is_undirected(data.edge_index)
    assert not (data.train_mask & data.val_mask).any()
    assert not (data.train_mask & data.test_mask).any()
    assert not (data.val_mask & data.test_mask).any()
    print(f'Nodes={data.num_nodes} Directed-edge entries={data.num_edges} Features={dataset.num_features} Classes={dataset.num_classes}', flush=True)
    print(f'Train/Val/Test={int(data.train_mask.sum())}/{int(data.val_mask.sum())}/{int(data.test_mask.sum())}', flush=True)
    data = data.to(device)
    model = Model(args.model, dataset.num_features, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005 if args.model == 'gat' else 0.01, weight_decay=5e-4)
    out = Path(args.out) / f'{args.model}_seed{args.seed}_{time.strftime("%Y%m%d_%H%M%S")}'
    out.mkdir(parents=True, exist_ok=False)
    history, best_val, best_epoch, best_state = [], -1.0, 0, None
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        # Only training labels contribute to the optimization objective.
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        loss.backward()
        optimizer.step()
        val_acc = accuracy(model, data, data.edge_index, data.val_mask)
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append({'epoch': epoch, 'train_loss': loss.item(), 'val_accuracy': val_acc})
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(f'Epoch {epoch:03d} | loss={loss.item():.4f} | val_acc={val_acc:.4f}', flush=True)
    model.load_state_dict(best_state)
    clean = accuracy(model, data, data.edge_index, data.test_mask)
    rows = []
    for fraction in sorted(set([0.0] + args.drop_rates)):
        for repeat in range(1 if fraction == 0 else args.repeats):
            perturb_seed = 100000 + args.seed * 100 + repeat
            edges = data.edge_index if fraction == 0 else delete_edges(data.edge_index, fraction, perturb_seed)
            acc = clean if fraction == 0 else accuracy(model, data, edges, data.test_mask)
            rows.append(dict(model=args.model, seed=args.seed, drop_rate=fraction, repeat=repeat, perturb_seed=perturb_seed,
                             directed_edges=edges.size(1), test_accuracy=acc, clean_test_accuracy=clean,
                             drop_percentage_points=(clean-acc)*100,
                             relative_drop=(clean-acc)/clean if clean else None))
    for filename, records in [('history.csv', history), ('results.csv', rows)]:
        with (out / filename).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    torch.save({k: v.cpu() for k, v in best_state.items()}, out / 'best_model.pt')
    manifest = dict(config=vars(args), python=sys.version, platform=platform.platform(), torch=torch.__version__,
                    pyg=torch_geometric.__version__, numpy=np.__version__, device=device, best_epoch=best_epoch,
                    best_val_accuracy=best_val, clean_test_accuracy=clean, seconds=time.perf_counter()-start,
                    parameters=sum(p.numel() for p in model.parameters()),
                    protocol='Cora public split; transductive; clean train/validation; validation-only checkpoint selection; test-time paired edge deletion',
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / 'summary.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'FINISHED | best_epoch={best_epoch} | best_val={best_val:.4f} | clean_test_acc={clean:.4f} | seconds={manifest["seconds"]:.1f}', flush=True)
    print(f'Output: {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
