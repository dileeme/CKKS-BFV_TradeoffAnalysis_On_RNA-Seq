import pandas as pd
import numpy as np

SCALE_FACTOR = 10_000
cancer_d1 = ["BRCA", "KIRC", "LUAD", "PRAD", "COAD"]
cancer_d2 = ["LUSC", "LUAD"]

files = {
    'D1 batch_a': ('datasets/batch_a_100.csv',    cancer_d1),
    'D1 batch_b': ('datasets/batch_b_400.csv',    cancer_d1),
    'D1 batch_c': ('datasets/batch_c_801.csv',    cancer_d1),
    'D2 batch_a': ('datasets/d2_batch_a_100.csv', cancer_d2),
    'D2 batch_b': ('datasets/d2_batch_b_400.csv', cancer_d2),
    'D2 batch_c': ('datasets/d2_batch_c_1129.csv',cancer_d2),
}

for label, (path, ctypes) in files.items():
    df = pd.read_csv(path)
    gene_cols = [c for c in df.columns if c != 'cancer_type']
    max_sum = max(
        int(np.round(df[df['cancer_type']==ct][gene_cols].values * SCALE_FACTOR).sum(axis=0).max())
        for ct in ctypes if ct in df['cancer_type'].values
    )
    print(f"{label}: max col sum = {max_sum:,}")
