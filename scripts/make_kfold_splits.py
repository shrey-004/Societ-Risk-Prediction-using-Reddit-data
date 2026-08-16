"""
Phase 1 (addendum) — generate grouped, stratified K-fold splits from the
FULL labeled dataset (train + val combined, 1635 rows / 153 users).

Why this exists: the current fixed 85/15 split (1364/271 rows) is what
both (a) reports every metric in reports/phase*.md and (b) is what
tune_factor_thresholds.py tunes per-class decision thresholds against.
That's a single, small, noisy sample — it's the direct cause of the
"ensemble + retuned thresholds looked better on val (0.372) but scored
worse on the real leaderboard (0.2371 vs the reverted 0.3417)" incident
documented in generate_submission.py. Averaging out-of-fold results across
5 folds instead of trusting one 271-row slice gives much more reliable
signal for both performance estimates and any future threshold/ensemble
decisions.

Grouping: never split a user (anon_user_id) across folds — same rule the
original Phase 1 split used, still required (a user's posts are similar in
style/content; leaking a user across train/val inflates apparent
performance).
Stratification: approximately balance risk_label distribution across
folds via sklearn's StratifiedGroupKFold — see
`stratified_group_kfold_assignment` below.

Usage (from esrd_project/ root):
    python scripts/make_kfold_splits.py
Output:
    data/processed/folds/fold{i}_train.csv, fold{i}_val.csv  for i in 0..n_folds-1
    (each row appears in exactly one fold's val set and n_folds-1 train sets)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def stratified_group_kfold_assignment(df: pd.DataFrame, group_col: str, label_col: str,
                                       n_folds: int, seed: int) -> np.ndarray:
    """Thin wrapper around sklearn's StratifiedGroupKFold: never splits a
    group (user) across folds, while approximately balancing the label
    distribution across folds. Returns a fold id per row (same order as
    df). Requires scikit-learn >= 1.1 (requirements.txt pins 1.5.1)."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_ids = np.empty(len(df), dtype=int)
    X_dummy = np.zeros(len(df))
    for fold_i, (_, val_idx) in enumerate(
        sgkf.split(X_dummy, y=df[label_col].values, groups=df[group_col].values)
    ):
        fold_ids[val_idx] = fold_i
    return fold_ids


def main():
    root = Path(__file__).resolve().parents[1]
    with open(root / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    processed_dir = root / cfg["paths"]["processed_dir"]
    n_folds = cfg["cross_validation"]["n_folds"]
    seed = cfg["cross_validation"]["seed"]

    train_df = pd.read_csv(processed_dir / "train_clean.csv")
    val_df = pd.read_csv(processed_dir / "val_clean.csv")
    full = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Full labeled set: {len(full)} rows / {full['user_id'].nunique()} users")

    fold_ids = stratified_group_kfold_assignment(full, "user_id", "risk_label", n_folds, seed)
    full = full.assign(fold=fold_ids)

    out_dir = processed_dir / "folds"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Fold':>4} {'val_rows':>9} {'val_users':>10}  risk_label % (Indicator/Ideation/Behavior/Attempt)")
    for i in range(n_folds):
        fold_val = full[full["fold"] == i].drop(columns=["fold"])
        fold_train = full[full["fold"] != i].drop(columns=["fold"])

        # sanity: no user leakage
        assert set(fold_val["user_id"]) & set(fold_train["user_id"]) == set()

        fold_val.to_csv(out_dir / f"fold{i}_val.csv", index=False)
        fold_train.to_csv(out_dir / f"fold{i}_train.csv", index=False)

        pct = fold_val["risk_label"].value_counts(normalize=True).reindex(
            ["Indicator", "Ideation", "Behavior", "Attempt"]).fillna(0) * 100
        pct_str = "/".join(f"{p:.0f}%" for p in pct)
        print(f"{i:>4} {len(fold_val):>9} {fold_val['user_id'].nunique():>10}  {pct_str}")

    print(f"\nSaved {n_folds} folds to {out_dir}")
    print("\nFull-dataset risk_label % for reference: "
          + "/".join(f"{p:.0f}%" for p in full['risk_label'].value_counts(normalize=True)
                      .reindex(['Indicator','Ideation','Behavior','Attempt']) * 100))


if __name__ == "__main__":
    main()
