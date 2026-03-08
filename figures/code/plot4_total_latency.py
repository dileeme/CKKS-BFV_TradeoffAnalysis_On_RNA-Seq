"""
Plot 4 — Total latency breakdown (Enc + Exec + Dec), BFV vs CKKS.

Total latency = enc_latency_ms + exec_latency_ms + dec_latency_ms.
  enc  : encode + encrypt all samples (scales linearly with n)
  exec : all HE operations — group sums + ciphertext subtraction per pair
         (scales with pair count, not sample count)
  dec  : decrypt + decode all result ciphertexts (negligible for both schemes)

Pareto finding: BFV total wall-clock time is substantially lower than CKKS
across all configurations. The gap widens at N=16384 and high pair count.
CKKS total cost is dominated by exec due to SIMD mean-computation overhead.

Dec latency is negligible for both schemes and is visible only as the
topmost (darkest) sliver on each bar.

Results averaged across 10 independent runs; error bars = ±1 std (propagated
in quadrature: total_std = sqrt(enc_std² + exec_std² + dec_std²)).

Hardware: Intel i7-10750H, 16 GB RAM, Ubuntu 22.04 (WSL2)
Software: Python 3.12, TenSEAL 0.3.14

HE parameters:
  CKKS — N=8192:  coeff_mod=[40,30,30,40], scale=2^30
  CKKS — N=16384: coeff_mod=[60,40,40,60], scale=2^40
  BFV  — N=8192/16384: default SEAL integer params
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from plot_style import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

bfv1  = pd.read_csv('results/phase3_bfv_dataset1.csv')
bfv2  = pd.read_csv('results/phase3_bfv_dataset2.csv')
ckks1 = pd.read_csv('results/phase3_ckks_dataset1.csv')
ckks2 = pd.read_csv('results/phase3_ckks_dataset2.csv')
all_data = pd.concat([bfv1, bfv2, ckks1, ckks2], ignore_index=True)

# Dataset 1 only; matched PMDs (8192 & 16384); all three batch sizes
comp = all_data[
    (all_data['dataset'] == 'dataset1') &
    (all_data['poly_mod_degree'].isin([8192, 16384]))
].groupby(['scheme', 'poly_mod_degree', 'samples']).agg(
    enc      =('enc_latency_ms',  'mean'),
    exec_lat =('exec_latency_ms', 'mean'),
    dec      =('dec_latency_ms',  'mean'),
    enc_std  =('enc_latency_ms',  'std'),
    exec_std =('exec_latency_ms', 'std'),
    dec_std  =('dec_latency_ms',  'std'),
).reset_index()
comp['total']     = comp['enc'] + comp['exec_lat'] + comp['dec']
comp['total_std'] = np.sqrt(comp['enc_std']**2 + comp['exec_std']**2 + comp['dec_std']**2)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
fig.subplots_adjust(wspace=0.35)

pmds         = [8192, 16384]
sample_labels = {100: 'n=100', 400: 'n=400', 801: 'n=801'}
samples_list  = [100, 400, 801]

ENC_ALPHA  = 0.45
EXEC_ALPHA = 0.80
DEC_ALPHA  = 1.00

for ax, pmd in zip(axes, pmds):
    x     = np.arange(len(samples_list))
    width = 0.32

    bfv_sub  = comp[(comp['scheme'] == 'BFV')  & (comp['poly_mod_degree'] == pmd)].sort_values('samples')
    ckks_sub = comp[(comp['scheme'] == 'CKKS') & (comp['poly_mod_degree'] == pmd)].sort_values('samples')

    for offset, scheme, sub in zip(
            [-width / 2 - 0.02, width / 2 + 0.02],
            ['BFV', 'CKKS'],
            [bfv_sub, ckks_sub]):

        key   = (scheme, pmd)
        color = COLORS[key]

        enc_vals  = sub['enc'].values
        exec_vals = sub['exec_lat'].values
        dec_vals  = sub['dec'].values
        tot_std   = sub['total_std'].values

        ax.bar(x + offset, enc_vals,  width, color=color, alpha=ENC_ALPHA)
        ax.bar(x + offset, exec_vals, width, bottom=enc_vals,
               color=color, alpha=EXEC_ALPHA)
        ax.bar(x + offset, dec_vals,  width, bottom=enc_vals + exec_vals,
               color=color, alpha=DEC_ALPHA)

        # Error bar on total
        totals = enc_vals + exec_vals + dec_vals
        ax.errorbar(x + offset, totals, yerr=tot_std,
                    fmt='none', color='#333333', capsize=3,
                    linewidth=1.2, zorder=5)

        # Speedup ratio above CKKS bars
        if scheme == 'CKKS':
            bfv_totals = bfv_sub['total'].values
            for xi, (ct, bt) in enumerate(zip(totals, bfv_totals)):
                speedup = ct / bt if bt > 0 else float('nan')
                ax.text(xi + offset,
                        ct + tot_std[xi] + ct * 0.04,
                        f'{speedup:.1f}×\nslower',
                        ha='center', fontsize=7.5,
                        color='#7B241C', fontweight='bold')

    # ── y-axis: log₁₀ scale ──────────────────────────────────────────────
    ax.set_yscale('log', base=10)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # ── Custom legend: component breakdown ───────────────────────────────
    legend_elements = [
        Patch(facecolor=COLORS[('BFV',  pmd)], alpha=ENC_ALPHA,  label='BFV — Enc'),
        Patch(facecolor=COLORS[('BFV',  pmd)], alpha=EXEC_ALPHA, label='BFV — Exec'),
        Patch(facecolor=COLORS[('BFV',  pmd)], alpha=DEC_ALPHA,  label='BFV — Dec'),
        Patch(facecolor=COLORS[('CKKS', pmd)], alpha=ENC_ALPHA,  label='CKKS — Enc'),
        Patch(facecolor=COLORS[('CKKS', pmd)], alpha=EXEC_ALPHA, label='CKKS — Exec'),
        Patch(facecolor=COLORS[('CKKS', pmd)], alpha=DEC_ALPHA,  label='CKKS — Dec'),
    ]
    ax.legend(handles=legend_elements, fontsize=7.8, loc='upper left',
              framealpha=0.92, edgecolor='#cccccc', ncol=2,
              title=f'N = {pmd}', title_fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([sample_labels[s] for s in samples_list], fontsize=9.5)
    ax.set_xlabel('Sample count (n)', fontsize=10)
    ax.set_ylabel('Total latency (ms, log₁₀ scale)', fontsize=10)
    ax.set_title(
        f'Dataset 1 — poly mod degree N = {pmd}\n'
        f'Stacked: Enc + Exec + Dec  |  Error bars = ±1 std, 10 runs',
        fontsize=10.5, fontweight='bold', pad=9)
    style_ax(ax)

# Caption text (copy into LaTeX):
#   "Total latency = enc + exec + dec, broken down as stacked bars.
#    Dec latency is negligible for both schemes (top sliver, barely visible).
#    CKKS total cost is dominated by exec due to SIMD mean-computation overhead.
#    Speedup labels above CKKS bars show the CKKS/BFV total latency ratio;
#    BFV is consistently faster. Error bars = ±1 std propagated in quadrature
#    across components, averaged over 10 runs. Log₁₀ y-axis."

fig.suptitle('Total Latency Breakdown — BFV vs CKKS (Dataset 1)',
             fontsize=12, fontweight='bold', y=1.01)
add_env_note(fig)
plt.tight_layout()
plt.savefig('figures/plot4_total_latency.png', dpi=180, bbox_inches='tight')
plt.show()
print("Saved figures/plot4_total_latency.png")
