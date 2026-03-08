"""
Plot 5 — MAE vs plaintext baseline: BFV (exact) vs CKKS (approximate).

MAE = mean absolute error between encrypted-computation output and
plaintext DE-score baseline, averaged across all gene features and pairs.

BFV uses exact integer arithmetic — MAE std = 0 across all 10 runs
(deterministic). BFV MAE is nonzero only due to fixed-point rounding in
the encode step (scale factor quantisation), typically ~10^-6.

CKKS uses floating-point approximation at scale 2^30 (N=8192) or 2^40
(N=16384). MAE is stable across runs but nonzero — typically ~10^-5.
Higher N gives slightly lower MAE because the larger coefficient modulus
allows a more precise floating-point representation.

CKKS SIMD batching does not affect MAE: approximation error arises from
the rescaling / modulus-switching chain, independent of batch size.

Results averaged across 10 independent runs; ±1 std shaded for CKKS
(BFV std = 0 exactly — band has zero width and is not shown).

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
    .groupby(['dataset', 'scheme', 'poly_mod_degree', 'samples'])['mae']
    .agg(mean='mean', std='std').reset_index())

datasets = [
    ('dataset1', 'Dataset 1 — UCI RNA-Seq\n10 cancer pairs'),
    ('dataset2', 'Dataset 2 — TCGA LUSC+LUAD\n1 cancer pair'),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
fig.subplots_adjust(wspace=0.12)

for ax, (ds, title) in zip(axes, datasets):
    sub = stats[stats['dataset'] == ds]

    for (scheme, pmd), grp in sub.groupby(['scheme', 'poly_mod_degree']):
        grp = grp.sort_values('samples')
        key = (scheme, pmd)
        ax.plot(grp['samples'], grp['mean'],
                color=COLORS[key], marker=MARKERS[key],
                linestyle=LINESTYLES[scheme], linewidth=LW, markersize=MS,
                label=scheme_label(scheme, pmd), zorder=3)
        # CKKS only — BFV std = 0 (deterministic exact arithmetic)
        if scheme == 'CKKS':
            ax.fill_between(grp['samples'],
                            grp['mean'] - grp['std'],
                            grp['mean'] + grp['std'],
                            color=COLORS[key], alpha=BAND_ALPHA, zorder=2)

    # log₁₀ scale on y-axis
    ax.set_yscale('log', base=10)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{x:.0e}'))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=9)
    ax.set_xlabel('Sample count (n)', fontsize=10)
    if ax == axes[0]:
        ax.set_ylabel('MAE vs plaintext baseline (log₁₀ scale)', fontsize=9.5)
    style_ax(ax)

    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: (
        0 if 'CKKS' in labels[i] else 1,
        -int(labels[i].split('=')[1].rstrip(')'))
    ))
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              fontsize=8.2, loc='upper right',
              title='Scheme (poly mod degree N)', title_fontsize=8,
              **LEGEND_KW)

    # Shaded region annotations — BFV and CKKS error bands
    if ds == 'dataset1':
        ax.axhspan(5e-7, 1e-5, alpha=0.06, color=COLORS[('BFV', 8192)],  zorder=0)
        ax.axhspan(1e-5, 3e-4, alpha=0.06, color=COLORS[('CKKS', 8192)], zorder=0)
        ax.text(150, 3e-6,  'BFV range\n(exact integer\narithmetic)',
                fontsize=7.5, color=COLORS[('BFV', 16384)], style='italic')
        ax.text(150, 1.5e-5, 'CKKS range\n(floating-point\napproximation,\nscale 2³⁰–2⁴⁰)',
                fontsize=7.5, color=COLORS[('CKKS', 16384)], style='italic')

# Caption text (copy into LaTeX):
#   "MAE = mean absolute error between FHE-computed DE scores and plaintext
#    baseline, log₁₀ scale. BFV MAE std = 0 across all 10 runs (deterministic
#    exact integer arithmetic); residual error ~10^{-6} arises from fixed-point
#    quantisation in the encode step. CKKS MAE is stable but nonzero due to
#    floating-point approximation at scale 2^{30} (N=8192) and 2^{40} (N=16384);
#    larger N yields slightly lower MAE. Shaded bands = ±1 std for CKKS only."

fig.suptitle('MAE vs Plaintext Baseline — BFV (Exact) vs CKKS (Approximate)',
             fontsize=12, fontweight='bold', y=1.01)
add_env_note(fig)
plt.tight_layout()
plt.savefig('figures/plot5_mae.png', dpi=180, bbox_inches='tight')
plt.show()
print("Saved figures/plot5_mae.png")
