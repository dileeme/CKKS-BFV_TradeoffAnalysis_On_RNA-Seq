import pandas as pd

expr = pd.read_csv("datasets/data.csv", index_col=0)
labels = pd.read_csv("datasets/labels.csv", index_col=0)

print("Expression matrix shape:", expr.shape)
print("Expression matrix index (first 5):", expr.index[:5].tolist())
print("Expression matrix columns (first 5):", expr.columns[:5].tolist())
print()
print("Labels shape:", labels.shape)
print("Labels index (first 5):", labels.index[:5].tolist())
print("Labels columns:", labels.columns.tolist())