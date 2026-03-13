"""
Models:
  M1 — Enc latency ~ sample count (linear, per scheme x PMD)
  M2 — CKKS exec latency ~ pair count (linear, per PMD, using D1/D2 means)
  M3 — BFV exec latency ~ sample count (linear, per PMD)
  M4 — CT size ~ PMD (power law via log-log, per scheme)
"""
import pandas as pd
import numpy as np
from scipy import stats
import os

bfv1  = pd.read_csv('results/phase3_bfv_dataset1.csv')
bfv2  = pd.read_csv('results/phase3_bfv_dataset2.csv')
ckks1 = pd.read_csv('results/phase3_ckks_dataset1.csv')
ckks2 = pd.read_csv('results/phase3_ckks_dataset2.csv')
all_data = pd.concat([bfv1, bfv2, ckks1, ckks2], ignore_index=True)

means = (all_data
    .groupby(['dataset','scheme','poly_mod_degree','samples'])
    .agg(enc=('enc_latency_ms','mean'),
         exec=('exec_latency_ms','mean'),
         dec=('dec_latency_ms','mean'),
         ct=('ct_size_kb','mean'))
    .reset_index())

results = []

def linear_fit(x, y, label, x_label, y_label, scheme, pmd=None):
    """Fit y = a*x + b, return result dict."""
    slope, intercept, r, p, se = stats.linregress(x, y)
    r2 = r**2
    n  = len(x)
    return {
        'model':       label,
        'scheme':      scheme,
        'pmd':         pmd if pmd else 'all',
        'x_var':       x_label,
        'y_var':       y_label,
        'slope':       round(slope, 6),
        'intercept':   round(intercept, 4),
        'R2':          round(r2, 6),
        'p_value':     round(p, 6),
        'n_points':    n,
        'equation':    f'y = {slope:.3f}·x + {intercept:.2f}',
    }

def power_fit(x, y, label, x_label, y_label, scheme):
    """Fit y = a * x^k via log-log regression."""
    log_x = np.log2(x)
    log_y = np.log2(y)
    slope, intercept, r, p, se = stats.linregress(log_x, log_y)
    r2 = r**2
    a  = 2**intercept
    return {
        'model':       label,
        'scheme':      scheme,
        'pmd':         'all',
        'x_var':       x_label,
        'y_var':       y_label,
        'slope':       round(slope, 6),   
        'intercept':   round(a, 4),       
        'R2':          round(r2, 6),
        'p_value':     round(p, 6),
        'n_points':    len(x),
        'equation':    f'y = {a:.2f} · x^{slope:.3f}',
    }

print("=== M1: Enc latency ~ sample count ===")
for scheme in ['BFV', 'CKKS']:
    pmds = means[means['scheme']==scheme]['poly_mod_degree'].unique()
    for pmd in sorted(pmds):
        # Use all datasets combined for more data points
        sub = means[(means['scheme']==scheme) & (means['poly_mod_degree']==pmd)]
        r = linear_fit(sub['samples'].values, sub['enc'].values,
                       'M1_enc_vs_samples', 'samples', 'enc_latency_ms',
                       scheme, pmd)
        results.append(r)
        print(f"  {scheme} PMD={pmd}: {r['equation']}  R²={r['R2']:.4f}")

# ── M2 ───────────────────────────────────

print("\n=== M2: CKKS exec ~ pair count ===")
pair_map = {'dataset1': 10, 'dataset2': 1}
for pmd in [8192, 16384]:
    sub = (means[(means['scheme']=='CKKS') & (means['poly_mod_degree']==pmd)]
           .groupby('dataset')['exec'].mean().reset_index())
    sub['pairs'] = sub['dataset'].map(pair_map)
    r = linear_fit(sub['pairs'].values, sub['exec'].values,
                   'M2_ckks_exec_vs_pairs', 'pairs', 'exec_latency_ms',
                   'CKKS', pmd)
    results.append(r)
    print(f"  CKKS PMD={pmd}: {r['equation']}  R²={r['R2']:.4f}  (only 2 pts — ratio confirms N3)")

# ── M3 ──────────────────────────────────
print("\n=== M3: BFV exec ~ sample count ===")
for pmd in [4096, 8192, 16384]:
    sub = means[(means['scheme']=='BFV') & (means['poly_mod_degree']==pmd)]
    r = linear_fit(sub['samples'].values, sub['exec'].values,
                   'M3_bfv_exec_vs_samples', 'samples', 'exec_latency_ms',
                   'BFV', pmd)
    results.append(r)
    print(f"  BFV PMD={pmd}: {r['equation']}  R²={r['R2']:.4f}")

# ── M4 ────────────────────────────────────────
print("\n=== M4: CT size ~ PMD (power law, log-log) ===")
ct_by_pmd = (all_data
    .groupby(['scheme','poly_mod_degree'])['ct_size_kb']
    .mean().reset_index())

for scheme in ['BFV', 'CKKS']:
    sub = ct_by_pmd[ct_by_pmd['scheme']==scheme].sort_values('poly_mod_degree')
    r = power_fit(sub['poly_mod_degree'].values, sub['ct_size_kb'].values,
                  'M4_ct_vs_pmd', 'poly_mod_degree', 'ct_size_kb', scheme)
    results.append(r)
    print(f"  {scheme}: {r['equation']}  R²={r['R2']:.4f}  (exponent={r['slope']:.3f})")

# ── M5 ────────────────────────────────────
print("\n=== M5: Enc latency ~ PMD (power law, per scheme, n=100 baseline) ===")
enc_pmd = means[means['samples']==100].groupby(['scheme','poly_mod_degree'])['enc'].mean().reset_index()
for scheme in ['BFV','CKKS']:
    sub = enc_pmd[enc_pmd['scheme']==scheme].sort_values('poly_mod_degree')
    r = power_fit(sub['poly_mod_degree'].values, sub['enc'].values,
                  'M5_enc_vs_pmd', 'poly_mod_degree', 'enc_latency_ms', scheme)
    results.append(r)
    print(f"  {scheme}: {r['equation']}  R²={r['R2']:.4f}  (exponent={r['slope']:.3f})")

df_results = pd.DataFrame(results)
os.makedirs('results', exist_ok=True)
df_results.to_csv('results/regression_results.csv', index=False)

print("\n" + "="*65)
print("SUMMARY TABLE (for paper)")
print("="*65)
print(f"{'Model':<35} {'Scheme':<6} {'PMD':<7} {'Equation':<35} {'R²':<8} {'n'}")
print("-"*65)
for _, row in df_results.iterrows():
    print(f"{row['model']:<35} {row['scheme']:<6} {str(row['pmd']):<7} {row['equation']:<35} {row['R2']:<8.4f} {row['n_points']}")

print("\nDone — saved to results/regression_results.csv")
