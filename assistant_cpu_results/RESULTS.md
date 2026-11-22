# Cora results

Mean accuracy (%) +/- sample SD across 3 training seeds.
For each seed and nonzero deletion rate, first average 3 perturbed-graph evaluations.
Same public split; clean training/validation; checkpoint chosen by validation accuracy.

| Model | 0% deleted | 10% deleted | 30% deleted | 50% deleted |
|---|---:|---:|---:|---:|
| mlp | 56.10 +/- 1.21 | 56.10 +/- 1.21 | 56.10 +/- 1.21 | 56.10 +/- 1.21 |
| gcn | 81.37 +/- 0.57 | 80.51 +/- 0.82 | 79.19 +/- 0.92 | 77.42 +/- 0.83 |
| gat | 82.80 +/- 0.61 | 82.04 +/- 0.36 | 80.39 +/- 0.61 | 78.59 +/- 0.70 |
| gated | 80.03 +/- 1.18 | 79.37 +/- 1.13 | 78.27 +/- 0.98 | 76.46 +/- 0.68 |
| fixed | 80.10 +/- 0.78 | 79.67 +/- 0.95 | 78.56 +/- 0.92 | 76.91 +/- 0.68 |

These are exploratory comparisons, not statistical significance claims.
GCN/MLP hidden size 16; GAT 8 heads x 8 dimensions. Models are not parameter matched.
Gated vs fixed tests adaptive mixing; gated also adds gate parameters.
Do not mix CPU/GPU or different library versions into one aggregate.

| Model | Parameters |
|---|---:|
| mlp | 23063 |
| gcn | 23063 |
| gat | 92373 |
| gated | 46040 |
| fixed | 46007 |
