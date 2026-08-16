"""
Check how many factors are predicted per test row, using the CURRENT
(pre-floor-fix) submission, to see how systemic the over-firing is.

Usage: python check_factor_counts.py
"""
import ast
import pandas as pd

df = pd.read_csv("outputs/predictions/RayofHope.csv")
df["n_factors"] = df["factors"].apply(lambda s: len(ast.literal_eval(s)))

print("Distribution of predicted factor count per row:")
print(df["n_factors"].value_counts().sort_index())
print(f"\nMean factors/row: {df['n_factors'].mean():.2f}")
print(f"Rows with all 24 factors: {(df['n_factors'] == 24).sum()} / {len(df)}")