"""
extract_top_genes.py
====================
Extracts top-ranked genes from plaintext DE scores.

Based on your actual directory structure:
  scoring/dataset1/de_baselines/de_baseline_batch_a.csv  (UCI batches a/b/c)
  scoring/dataset2/d2_de_baseline_batch_a.csv            (TCGA batches a/b/c)
  results/ckks_d2_plain_scores.npy                       (numpy plain scores)

Run from your project root. Paste output back to Claude.
"""

import pandas as pd
import numpy as np
import os
from itertools import combinations

# ─────────────────────────────────────────────────────────────
# PATHS — based on your directory structure
# ─────────────────────────────────────────────────────────────

# UCI Dataset 1 — three batch files
UCI_BATCHES = [
    "scoring/dataset1/de_baselines/de_baseline_batch_a.csv",
    "scoring/dataset1/de_baselines/de_baseline_batch_b.csv",
    "scoring/dataset1/de_baselines/de_baseline_batch_c.csv",
]

# TCGA Dataset 2 — three batch files
TCGA_BATCHES = [
    "scoring/dataset2/d2_de_baseline_batch_a.csv",
    "scoring/dataset2/d2_de_baseline_batch_b.csv",
    "scoring/dataset2/d2_de_baseline_batch_c.csv",
]

# NumPy plain scores (may also contain useful data)
NPY_SCORES = "results/ckks_d2_plain_scores.npy"

TOP_K = 15  # number of top genes to print per comparison

# ─────────────────────────────────────────────────────────────
# HELPER: peek at a CSV to understand its structure
# ─────────────────────────────────────────────────────────────

def peek(path):
    if not os.path.exists(path):
        print(f"  NOT FOUND: {path}")
        return None
    df = pd.read_csv(path)
    print(f"  Loaded: {path}")
    print(f"  Shape : {df.shape}")
    print(f"  Cols  : {list(df.columns[:8])} ...")
    print(f"  Dtypes: {dict(df.dtypes.value_counts())}")
    return df

# ─────────────────────────────────────────────────────────────
# STEP 1: Inspect the batch files to understand format
# ─────────────────────────────────────────────────────────────

print("="*60)
print("PEEKING AT FILE STRUCTURES")
print("="*60)

print("\n--- UCI batch_a ---")
uci_a = peek(UCI_BATCHES[0])

print("\n--- TCGA batch_a ---")
tcga_a = peek(TCGA_BATCHES[0])

print("\n--- NPY plain scores ---")
if os.path.exists(NPY_SCORES):
    npy = np.load(NPY_SCORES, allow_pickle=True)
    print(f"  Loaded: {NPY_SCORES}")
    print(f"  Shape : {npy.shape if hasattr(npy, 'shape') else 'dict/object'}")
    if isinstance(npy, np.ndarray) and npy.ndim <= 2:
        print(f"  Sample values: {npy.flat[:5]}")
    elif isinstance(npy, np.ndarray) and npy.dtype == object:
        # might be a dict saved as object array
        obj = npy.item()
        print(f"  Type inside: {type(obj)}")
        if isinstance(obj, dict):
            print(f"  Keys: {list(obj.keys())[:10]}")
else:
    print(f"  NOT FOUND: {NPY_SCORES}")

# ─────────────────────────────────────────────────────────────
# STEP 2: Detect score format and extract top genes
# The batch CSVs likely have one of these structures:
#   A) columns = [comparison, gene, score]  (long format)
#   B) columns = [gene, BRCA_vs_KIRC, LUAD_vs_PRAD, ...]  (wide format)
#   C) columns = [gene_id, plaintext_score] per comparison file
# ─────────────────────────────────────────────────────────────

def detect_and_extract(batch_paths, dataset_name):
    """Load all batch files, detect format, extract top genes."""
    print(f"\n{'='*60}")
    print(f"{dataset_name} — TOP {TOP_K} GENES PER COMPARISON")
    print(f"{'='*60}")

    # Load all batches and concatenate
    dfs = []
    for p in batch_paths:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
        else:
            print(f"  Skipping missing file: {p}")
    if not dfs:
        print("  No batch files found.")
        return

    df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())

    cols = list(df.columns)

    # ── Format A: long format with comparison/gene/score columns ──
    comp_col  = next((c for c in cols if "comp"  in c.lower() or "pair" in c.lower()), None)
    gene_col  = next((c for c in cols if "gene"  in c.lower() or "feat" in c.lower()), None)
    score_col = next((c for c in cols if "score" in c.lower() or "diff" in c.lower()
                      or "plain" in c.lower() or "base"  in c.lower()), None)

    if comp_col and gene_col and score_col:
        print(f"\nDetected FORMAT A (long): comp='{comp_col}' gene='{gene_col}' score='{score_col}'")
        for comp, grp in df.groupby(comp_col):
            grp_sorted = grp.nlargest(TOP_K, score_col)
            print(f"\n  --- {comp} ---")
            for rank, row in enumerate(grp_sorted.itertuples(), 1):
                g = getattr(row, gene_col)
                s = getattr(row, score_col)
                print(f"    {rank:2d}. {str(g):<25} {float(s):.6f}")
        return

    # ── Format B: wide format, genes as rows, comparisons as columns ──
    # First column is gene name, remaining columns are comparisons
    possible_gene_col = cols[0]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 1 and df[possible_gene_col].dtype == object:
        print(f"\nDetected FORMAT B (wide): gene_col='{possible_gene_col}', "
              f"comparison cols={numeric_cols[:5]}...")
        for comp in numeric_cols:
            top = df.nlargest(TOP_K, comp)[[possible_gene_col, comp]]
            print(f"\n  --- {comp} ---")
            for rank, row in enumerate(top.itertuples(), 1):
                g = getattr(row, possible_gene_col)
                s = getattr(row, comp)
                print(f"    {rank:2d}. {str(g):<25} {float(s):.6f}")
        return

    # ── Format C: all numeric, assume rows=genes, cols=comparisons ──
    if len(numeric_cols) == len(cols):
        print(f"\nDetected FORMAT C (all numeric): {len(cols)} comparisons, "
              f"{len(df)} genes")
        print("  NOTE: no gene name column found — will print row index as gene ID")
        print("  You may need to load gene names separately from your dataset CSV")
        for comp in cols:
            top_idx = df[comp].nlargest(TOP_K).index
            print(f"\n  --- {comp} ---")
            for rank, idx in enumerate(top_idx, 1):
                print(f"    {rank:2d}. row_{idx:<20} {df[comp][idx]:.6f}")
        return

    print("\n  Could not detect format automatically.")
    print("  Please share first 5 rows and column names with Claude.")


# ─────────────────────────────────────────────────────────────
# STEP 3: Run extraction
# ─────────────────────────────────────────────────────────────

detect_and_extract(UCI_BATCHES,  "UCI DATASET 1")
detect_and_extract(TCGA_BATCHES, "TCGA DATASET 2")

print("\n\nDONE — paste output above back to Claude.")