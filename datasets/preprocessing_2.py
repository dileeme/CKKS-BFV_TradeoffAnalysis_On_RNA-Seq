"""
Dataset 2 Preprocessing Script
TCGA LUSC + LUAD HiSeqV2 RNA-Seq Data

Mirrors Dataset 1 preprocessing exactly:
  - Top 500 features by variance
  - Min-max normalization to [0, 1]
  - random_state=42
  - Three batch splits (small, medium, full)
  - Plaintext DE baselines for LUSC vs LUAD

Run from your project root:
  python preprocess_dataset2.py

Input files expected in datasets/:
  datasets/LUSC_HiSeqV2
  datasets/LUAD_HiSeqV2
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

LUSC_FILE      = "datasets/LUSC_HiSeqV2"
LUAD_FILE      = "datasets/LUAD_HiSeqV2"
TOP_N_FEATURES = 500
RANDOM_STATE   = 42
OUTPUT_DIR     = "datasets"
BASELINE_DIR   = "de_baselines"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BASELINE_DIR, exist_ok=True)

# ── STEP 1: LOAD ──────────────────────────────────────────────────────────────

loaded = {}
for name, path in tqdm([("LUSC", LUSC_FILE), ("LUAD", LUAD_FILE)], desc="Step 1/7 Loading files", unit="file"):
    raw = pd.read_csv(path, sep="\t", index_col=0)
    loaded[name] = raw.T.copy()
    tqdm.write(f"  {name}: {loaded[name].shape[0]} samples x {loaded[name].shape[1]} genes")

lusc_expr = loaded["LUSC"]
luad_expr = loaded["LUAD"]

# ── STEP 2: MERGE ─────────────────────────────────────────────────────────────

with tqdm(total=3, desc="Step 2/7 Merging datasets", unit="op") as pbar:
    common_genes = lusc_expr.columns.intersection(luad_expr.columns)
    tqdm.write(f"  Common genes: {len(common_genes)}")
    pbar.update(1)

    labels = pd.concat([
        pd.Series(["LUSC"] * len(lusc_expr), index=lusc_expr.index),
        pd.Series(["LUAD"] * len(luad_expr), index=luad_expr.index)
    ], axis=0)
    labels.name = "cancer_type"
    pbar.update(1)

    merged = pd.concat([lusc_expr[common_genes], luad_expr[common_genes]], axis=0)
    tqdm.write(f"  Merged shape: {merged.shape}")
    pbar.update(1)

# ── STEP 3: DROP UNKNOWN GENE SYMBOLS ─────────────────────────────────────────

with tqdm(total=1, desc="Step 3/7 Dropping unknown gene symbols", unit="op") as pbar:
    before = merged.shape[1]
    merged = merged.loc[:, ~merged.columns.str.startswith("?")]
    dropped = before - merged.shape[1]
    tqdm.write(f"  Dropped {dropped} unknown genes, {merged.shape[1]} remaining")
    pbar.update(1)

# ── STEP 4: HANDLE MISSING VALUES ─────────────────────────────────────────────

with tqdm(total=1, desc="Step 4/7 Checking missing values", unit="op") as pbar:
    missing = merged.isnull().sum().sum()
    if missing > 0:
        tqdm.write(f"  Found {missing} missing values — filling with column mean")
        merged = merged.fillna(merged.mean())
    else:
        tqdm.write(f"  No missing values found")
    pbar.update(1)

# ── STEP 5: TOP 500 FEATURES BY VARIANCE ──────────────────────────────────────

with tqdm(total=2, desc="Step 5/7 Selecting top 500 features", unit="op") as pbar:
    variances = merged.var(axis=0)
    pbar.update(1)
    top500_genes = variances.nlargest(TOP_N_FEATURES).index
    merged = merged[top500_genes]
    tqdm.write(f"  Shape after feature selection: {merged.shape}")
    pbar.update(1)

# ── STEP 6: MIN-MAX NORMALIZATION ─────────────────────────────────────────────

with tqdm(total=1, desc="Step 6/7 Applying min-max normalization", unit="op") as pbar:
    min_vals = merged.min(axis=0)
    max_vals = merged.max(axis=0)
    denom = (max_vals - min_vals).replace(0, 1)
    merged_norm = (merged - min_vals) / denom
    tqdm.write(f"  Value range: [{merged_norm.min().min():.4f}, {merged_norm.max().max():.4f}]")
    pbar.update(1)

# ── STEP 7: SHUFFLE + SAVE ────────────────────────────────────────────────────

with tqdm(total=1, desc="Step 7/7 Shuffling", unit="op") as pbar:
    shuffle_idx = np.random.RandomState(RANDOM_STATE).permutation(len(merged_norm))
    merged_norm = merged_norm.iloc[shuffle_idx].reset_index(drop=True)
    labels_shuffled = labels.iloc[shuffle_idx].reset_index(drop=True)
    total_samples = len(merged_norm)
    tqdm.write(f"  Total: {total_samples} | LUSC: {(labels_shuffled == 'LUSC').sum()} | LUAD: {(labels_shuffled == 'LUAD').sum()}")
    pbar.update(1)

merged_final = merged_norm.copy()
merged_final.insert(0, "cancer_type", labels_shuffled.values)

# ── BATCH SPLITS + MASTER FILE ────────────────────────────────────────────────

batch_small  = 100
batch_medium = min(400, total_samples)
batch_full   = total_samples

batch_a = merged_final.iloc[:batch_small]
batch_b = merged_final.iloc[:batch_medium]
batch_c = merged_final

save_files = [
    (merged_final, os.path.join(OUTPUT_DIR, "d2_processed_dataset.csv"),          "master file"),
    (batch_a,      os.path.join(OUTPUT_DIR, "d2_batch_a_100.csv"),                 "batch_a (100)"),
    (batch_b,      os.path.join(OUTPUT_DIR, f"d2_batch_b_{batch_medium}.csv"),     f"batch_b ({batch_medium})"),
    (batch_c,      os.path.join(OUTPUT_DIR, f"d2_batch_c_{batch_full}.csv"),       f"batch_c ({batch_full})"),
]

for df, path, label in tqdm(save_files, desc="Saving dataset files", unit="file"):
    df.to_csv(path, index=False)
    tqdm.write(f"  Saved {label} → {path}")

# ── PLAINTEXT DE BASELINES ────────────────────────────────────────────────────

def compute_de_baseline(df):
    expr = df.drop(columns=["cancer_type"])
    lusc_mask = df["cancer_type"] == "LUSC"
    luad_mask = df["cancer_type"] == "LUAD"
    mean_lusc = expr[lusc_mask].mean(axis=0)
    mean_luad = expr[luad_mask].mean(axis=0)
    result = pd.DataFrame(index=expr.columns)
    result["mean_LUSC"]       = mean_lusc
    result["mean_LUAD"]       = mean_luad
    result["DE_LUSC_vs_LUAD"] = (mean_lusc - mean_luad).abs()
    return result

baseline_jobs = [
    (batch_a, "d2_de_baseline_batch_a.csv", "batch_a"),
    (batch_b, "d2_de_baseline_batch_b.csv", "batch_b"),
    (batch_c, "d2_de_baseline_batch_c.csv", "batch_c"),
]

for batch_df, out_name, label in tqdm(baseline_jobs, desc="Computing DE baselines", unit="batch"):
    de = compute_de_baseline(batch_df)
    out_path = os.path.join(BASELINE_DIR, out_name)
    de.to_csv(out_path)
    tqdm.write(f"  {label}: mean DE={de['DE_LUSC_vs_LUAD'].mean():.4f}, "
               f"max DE={de['DE_LUSC_vs_LUAD'].max():.4f}, "
               f"NaNs={de.isnull().sum().sum()} → {out_path}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("DATASET 2 PREPROCESSING COMPLETE")
print("="*60)
print(f"  Total samples : {total_samples}")
print(f"  Features      : {TOP_N_FEATURES}")
print(f"  Cancer types  : LUSC, LUAD")
print(f"  Normalization : min-max [0, 1]")
print(f"  random_state  : {RANDOM_STATE}")
print(f"\n  datasets/")
print(f"    d2_processed_dataset.csv")
print(f"    d2_batch_a_100.csv")
print(f"    d2_batch_b_{batch_medium}.csv")
print(f"    d2_batch_c_{batch_full}.csv")
print(f"\n  de_baselines/")
print(f"    d2_de_baseline_batch_a.csv")
print(f"    d2_de_baseline_batch_b.csv")
print(f"    d2_de_baseline_batch_c.csv")