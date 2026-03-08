"""
Plot 2 — Execution latency vs sample count, BFV vs CKKS.

Exec latency = time to perform all homomorphic operations: group-sum
accumulation + ciphertext subtraction across every pairwise cancer-type
comparison. Runtime scales with the NUMBER OF PAIRS, not sample count.

  Dataset 1: C(5,2) = 10 pairs  →  higher exec latency
  Dataset 2: C(2,2) =  1 pair   →  ~10× lower exec latency
             (despite Dataset 2 having MORE samples)

CKKS SIMD batching: packs up to N/2 values into one ciphertext; one HE
addition operates on the full batch simultaneously — amortising per-sample
cost. CKKS exec latency is therefore nearly flat with sample count.
BFV: no cross-sample SIMD packing; each ciphertext holds one sample.
BFV exec cost scales weakly with sample count (more group-sum terms) but
is still dominated by pair count.

Results averaged across 10 independent runs; ±1 std shown as shaded bands.

Hardware: Intel i7-10750H, 16 GB RAM, Ubuntu 22.04 (WSL2)
Software: Python 3.12, TenSEAL 0.3.14

HE parameters:
  CKKS — N=8192:  coeff_mod=[40,30,30,40], scale=2^30
  CKKS — N=16384: coeff_mod=[60,40,40,60], scale=2^40
  BFV  — N=4096/8192/16384: default SEAL integer params
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from plot_style import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

bfv1  = pd.read_csv('results/phase3_bfv_dataset1.csv')
bfv2  = pd.read_csv('results/phase3_bfv_dataset2.csv')
ckks1 = pd.read_csv('results/phase3_ckks_dataset1.csv')
ckks2 = pd.read_csv('results/phase3_ckks_dataset2.csv')
all_data = pd.concat([bfv1, bfv2, ckks1, ckks2], ignore_index=True)

stats = (all_data
    .groupby(['dataset', 'scheme', 'poly_mod_degree', 'samples'])['exec_latency_ms']
    .agg(mean='mean', std='std').reset_index())

datasets = [
    ('dataset1', 'Dataset 1 — UCI RNA-Seq\n10 pairwise comparisons, n ∈ {100, 400, 801}'),
    ('dataset2', 'Dataset 2 — TCGA LUSC+LUAD\n1 pairwise comparison, n ∈ {100, 400, 1129}'),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(wspace=0.32)

for ax, (ds, title) in zip(axes, datasets):
    sub = stats[stats['dataset'] == ds]

    for (scheme, pmd), grp in sub.groupby(['scheme', 'poly_mod_degree']):
        grp = grp.sort_values('samples')
        key = (scheme, pmd)
        ax.plot(grp['samples'], grp['mean'],
                color=COLORS[key], marker=MARKERS[key],
                linestyle=LINESTYLES[scheme], linewidth=LW, markersize=MS,
                label=scheme_label(scheme, pmd), zorder=3)
        ax.fill_between(grp['samples'],
                        grp['mean'] - grp['std'],
                        grp['mean'] + grp['std'],
                        color=COLORS[key], alpha=BAND_ALPHA, zorder=2)

    # log₁₀ scale on y-axis
    ax.set_yscale('log', base=10)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=9)
    ax.set_xlabel('Sample count (n)', fontsize=10)
    ax.set_ylabel('Exec latency (ms, log₁₀ scale)', fontsize=9.5)
    style_ax(ax)

    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: (
        0 if 'CKKS' in labels[i] else 1,
        -int(labels[i].split('=')[1].rstrip(')'))
    ))
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              fontsize=8.2, loc='lower right',
              title='Scheme (poly mod degree N)', title_fontsize=8,
              **LEGEND_KW)

# Caption text (copy into LaTeX):
#   "Execution latency (exec\_latency\_ms) = homomorphic group-sum accumulation
#    and ciphertext subtraction across all pairwise cancer-type comparisons.
#    Runtime is driven by pair count [C(5,2)=10 for D1; C(2,2)=1 for D2],
#    not sample count — Dataset 2 exec latency is ~10× lower despite having
#    more samples. CKKS SIMD batching amortises per-sample cost; BFV operates
#    per ciphertext without cross-sample packing. Shaded bands = ±1 std, 10 runs."

fig.suptitle('Homomorphic Exec Latency vs Sample Count — BFV vs CKKS',
             fontsize=12, fontweight='bold', y=1.01)
add_env_note(fig)
plt.tight_layout()
plt.savefig('figures/plot2_exec_latency.png', dpi=180, bbox_inches='tight')
plt.show()
print("Saved figures/plot2_exec_latency.png")
