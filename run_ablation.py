"""
Ablation study — isolates the effect of target leakage and sentinel encoding.

Runs the same three classifiers over the same data and the same train/test split,
varying only the feature representation:

  full            Baseline. All seven features, including delta_days and
                  delta_valor_pct — the two quantities the labeler thresholds to
                  produce the target. Reproduces data/results/<scenario>/.

  no_leak         Drops delta_days and delta_valor_pct. Removes the direct
                  circularity between label and features.

  no_leak_strict  Also drops nfse_valor_iss and nfse_aliquota, which are zero for
                  every unmatched record and therefore proxy the existence of a
                  join. Leaves only genuinely independent signal.

  sentinel_fixed  Keeps the delta features but replaces the 9999 / 100 sentinels
                  with an explicit has_match indicator plus neutral imputation,
                  so StandardScaler is no longer dominated by the sentinel.

Outputs data/results/ablation/ablation_summary.csv plus per-run confusion
matrices. Does not touch the baseline artifacts in data/results/exact|fuzzy.

Usage:
    python run_ablation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from reconciliacao.models.features import build_features
from reconciliacao.models.trainer import train_all
from reconciliacao.utils.config import load_config

CONFIGS = ["full", "no_leak", "no_leak_strict", "sentinel_fixed"]
SCENARIOS = ["exact", "fuzzy"]
LEAKED = ["delta_days", "delta_valor_pct"]
ISS_PROXIES = ["nfse_valor_iss", "nfse_aliquota"]


def apply_config(X: pd.DataFrame, config: str) -> pd.DataFrame:
    """Derive a feature matrix variant from the baseline feature matrix."""
    X = X.copy()

    if config == "full":
        return X

    if config == "no_leak":
        return X.drop(columns=LEAKED)

    if config == "no_leak_strict":
        return X.drop(columns=LEAKED + ISS_PROXIES)

    if config == "sentinel_fixed":
        # 9999 days / 100 pct are sentinels meaning "no NFS-e joined to this payment".
        unmatched = X["delta_days"] >= 9999.0
        X["has_match"] = (~unmatched).astype(int)
        # Neutral imputation: median over records that actually joined.
        for col in LEAKED:
            median_matched = X.loc[~unmatched, col].median()
            X.loc[unmatched, col] = median_matched
        return X

    raise ValueError(f"unknown config: {config}")


def per_class_recall(y_true, y_pred) -> dict[str, float]:
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    return {f"recall_class_{i}": float(r) for i, r in enumerate(recalls)}


def main() -> None:
    cfg = load_config("config.yaml")
    out_root = Path("data/results/ablation")
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []

    for scenario in SCENARIOS:
        df = pd.read_csv(f"data/processed/{scenario}_reconciliado.csv")
        X_base, y = build_features(df)
        n_classes = cfg["scenarios"][scenario]["n_classes"]

        for config in CONFIGS:
            X = apply_config(X_base, config)
            out_dir = out_root / config / scenario
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{scenario}/{config}] features={list(X.columns)}")
            results = train_all(X, y, cfg, out_dir / "models")

            for name, r in results.items():
                est, X_test, y_test = r["estimator"], r["X_test"], r["y_test"]
                y_pred = est.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)

                row = {
                    "scenario": scenario,
                    "config": config,
                    "algorithm": name,
                    "n_features": X.shape[1],
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                    "precision_macro": precision_score(
                        y_test, y_pred, average="macro", zero_division=0
                    ),
                    "recall_macro": recall_score(
                        y_test, y_pred, average="macro", zero_division=0
                    ),
                    "cv_mean": r["cv_score_mean"],
                    "cv_std": r["cv_score_std"],
                }
                row.update(per_class_recall(y_test, y_pred))

                if n_classes == 2:
                    tn, fp, fn, tp = cm.ravel()
                    row.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

                rows.append(row)
                np.savetxt(out_dir / f"cm_{name}.txt", cm, fmt="%d")
                print(f"    {name:20s} f1_macro={row['f1_macro']:.4f} acc={row['accuracy']:.4f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / "ablation_summary.csv", index=False)
    print(f"\nWrote {out_root / 'ablation_summary.csv'} ({len(summary)} runs)")


if __name__ == "__main__":
    main()
