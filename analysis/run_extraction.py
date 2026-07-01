import pandas as pd
import numpy as np

TOP_K = 15

UCI_BATCHES = [
    "scoring/dataset1/de_baselines/de_baseline_batch_a.csv",
    "scoring/dataset1/de_baselines/de_baseline_batch_b.csv",
    "scoring/dataset1/de_baselines/de_baseline_batch_c.csv",
]
TCGA_BATCHES = [
    "scoring/dataset2/d2_de_baseline_batch_a.csv",
    "scoring/dataset2/d2_de_baseline_batch_b.csv",
    "scoring/dataset2/d2_de_baseline_batch_c.csv",
]

# UCI
uci = pd.concat([pd.read_csv(p) for p in UCI_BATCHES], ignore_index=True)
comp_cols = [c for c in uci.columns if c != 'gene']
print("="*60)
print("UCI DATASET 1")
print("="*60)
for comp in comp_cols:
    top = uci.nlargest(TOP_K, comp)[['gene', comp]]
    print(f"\n--- {comp} ---")
    for rank, row in enumerate(top.itertuples(), 1):
        print(f"  {rank:2d}. {row.gene:<25} {getattr(row, comp):.6f}")

# TCGA
tcga = pd.concat([pd.read_csv(p) for p in TCGA_BATCHES], ignore_index=True)
print("\n" + "="*60)
print("TCGA DATASET 2")
print("="*60)
comp_cols_t = [c for c in tcga.columns if c != 'gene']
for comp in comp_cols_t:
    top = tcga.nlargest(TOP_K, comp)[['gene', comp]]
    print(f"\n--- {comp} ---")
    for rank, row in enumerate(top.itertuples(), 1):
        print(f"  {rank:2d}. {row.gene:<25} {getattr(row, comp):.6f}")
