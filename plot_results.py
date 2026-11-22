"""Optional: python plot_results.py experiment_cpu_TIMESTAMP
Requires matplotlib (not needed for training or tabular summaries).
"""
import csv
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1])
with (root/'aggregate.csv').open(encoding='utf-8-sig') as f:
    rows=list(csv.DictReader(f))
fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
colors={'mlp':'#6b7280','gcn':'#2563eb','gat':'#d97706','gated':'#059669','fixed':'#9333ea'}
for ax,models,title in [(axes[0],['mlp','gcn','gat'],'Baselines'),(axes[1],['gcn','gated','fixed'],'Gating ablation')]:
    for model in models:
        part=[r for r in rows if r['model']==model]
        ax.errorbar([float(r['drop_rate'])*100 for r in part],
                    [float(r['accuracy_pct_mean']) for r in part],
                    yerr=[float(r['accuracy_pct_sd']) for r in part],
                    label=model.upper(),color=colors[model],marker='o',capsize=3,linewidth=2)
    ax.set(title=title,xlabel='Undirected edges deleted (%)',ylabel='Test accuracy (%)',xticks=[0,10,30,50],ylim=(45,90))
    ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False)
fig.suptitle('Cora | clean-graph training, test-time edge deletion',fontsize=14)
fig.supxlabel('Mean +/- sample SD across 3 training seeds; 3 perturbations averaged within each seed.',fontsize=9)
fig.savefig(root/'robustness.png',dpi=180)
fig.savefig(root/'robustness.pdf')
print(root/'robustness.png')
