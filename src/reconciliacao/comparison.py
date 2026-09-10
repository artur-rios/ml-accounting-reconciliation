"""Answer the first research question: which algorithm best automates reconciliation.

Ten independent datasets, three algorithms, one decision criterion derived from
the control objective -- exception recall under a precision floor. The label
comes from generator ground truth, so the comparison measures the algorithms
rather than the labeling rule.
"""

from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
from sklearn.metrics import PrecisionRecallDisplay  # noqa: E402

from reconciliacao.etl.truth_labeler import label_from_truth
from reconciliacao.models.comparison_stats import paired_comparisons
from reconciliacao.models.comparison_trainer import fit_algorithms
from reconciliacao.models.decision import choose_threshold, exception_metrics
from reconciliacao.models.features_v2 import build_features_v2, fit_supplier_terms
from reconciliacao.models.leak_guard import assert_no_leak
from reconciliacao.simulation.truth_generator import generate_comparison_dataset

PRIMARY_METRIC = "recall_excecao"


def _split_three_ways(df: pd.DataFrame, cmp_cfg: dict, seed: int):
    fractions = cmp_cfg["split"]
    train_df, rest_df = train_test_split(
        df,
        train_size=fractions["train"],
        stratify=df["label"],
        random_state=seed,
    )
    val_fraction = fractions["val"] / (fractions["val"] + fractions["test"])
    val_df, test_df = train_test_split(
        rest_df,
        train_size=val_fraction,
        stratify=rest_df["label"],
        random_state=seed,
    )
    return train_df, val_df, test_df


def run_seed(cfg: dict, seed: int) -> tuple[list[dict], pd.Series, dict[str, tuple]]:
    """Run one seed end to end.

    Returns the per-algorithm metric rows, the leak-guard report, and the test
    partition predictions kept for plotting.
    """
    cmp_cfg = cfg["comparison"]

    df_payments, df_invoices, df_truth = generate_comparison_dataset(cmp_cfg, seed)
    pairs = label_from_truth(df_payments, df_invoices, df_truth)

    train_df, val_df, test_df = _split_three_ways(pairs, cmp_cfg, seed)
    supplier_terms, global_term = fit_supplier_terms(train_df)

    X_tr, y_tr = build_features_v2(train_df, cmp_cfg, supplier_terms, global_term)
    X_val, y_val = build_features_v2(val_df, cmp_cfg, supplier_terms, global_term)
    X_te, y_te = build_features_v2(test_df, cmp_cfg, supplier_terms, global_term)

    leak_report = assert_no_leak(
        X_tr, y_tr, max_accuracy=cmp_cfg["leak_guard_max_stump_accuracy"], seed=seed
    ).rename(seed)

    models = fit_algorithms(X_tr, y_tr, cfg, seed)

    rows, curves = [], {}
    for name, model in models.items():
        proba_val = model.predict_proba(X_val)[:, 0]
        proba_test = model.predict_proba(X_te)[:, 0]
        curves[name] = ((y_te.to_numpy() == 0).astype(int), proba_test)

        threshold = choose_threshold(y_val, proba_val, cmp_cfg["min_precision"])
        if threshold is None:
            rows.append({
                "seed": seed, "algorithm": name, "threshold": None,
                "precisao_validacao": None, "recall_excecao": 0.0,
                "precisao_excecao": 0.0, "taxa_encaminhamento": 0.0,
                "pr_auc_excecao": float("nan"), "f1_macro": float("nan"),
            })
            continue

        metrics_val = exception_metrics(y_val, proba_val, threshold)
        metrics_test = exception_metrics(y_te, proba_test, threshold)
        rows.append({
            "seed": seed,
            "algorithm": name,
            "threshold": threshold,
            "precisao_validacao": metrics_val["precisao_excecao"],
            **metrics_test,
        })
    return rows, leak_report, curves


def _plot_pr_curves(curves: dict[str, tuple], destination: Path) -> None:
    """Precision-recall curves for the exception class, from the first seed."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (is_exception, proba) in curves.items():
        PrecisionRecallDisplay.from_predictions(is_exception, proba, name=name, ax=ax)
    ax.set_title("Exception-class precision-recall — first seed", color="black")
    ax.set_xlabel("Exception recall", color="black")
    ax.set_ylabel("Exception precision", color="black")
    ax.tick_params(colors="black")
    fig.savefig(destination, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_recall_boxplot(per_seed: pd.DataFrame, destination: Path) -> None:
    """Spread of exception recall across seeds, one box per algorithm."""
    algorithms = sorted(per_seed["algorithm"].unique())
    data = [per_seed.loc[per_seed["algorithm"] == a, PRIMARY_METRIC] for a in algorithms]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data, tick_labels=algorithms)
    ax.set_ylabel("Exception recall (precision >= floor)", color="black")
    ax.set_title(f"Spread across {per_seed['seed'].nunique()} seeds", color="black")
    ax.tick_params(colors="black")
    fig.savefig(destination, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(cfg: dict, n_seeds: int, output_dir: Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, leak_reports, first_seed_curves = [], [], {}
    for seed in range(n_seeds):
        print(f"[seed {seed + 1}/{n_seeds}] generating, guarding, training...")
        seed_rows, leak_report, curves = run_seed(cfg, seed)
        rows.extend(seed_rows)
        leak_reports.append(leak_report)
        if seed == 0:
            first_seed_curves = curves

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    pd.concat(leak_reports, axis=1).to_csv(output_dir / "leak_guard_report.csv")

    wilcoxon = paired_comparisons(per_seed, metric=PRIMARY_METRIC)
    wilcoxon.to_csv(output_dir / "wilcoxon.csv", index=False)

    summary = (
        per_seed.groupby("algorithm")
        .agg(
            recall_excecao_medio=("recall_excecao", "mean"),
            recall_excecao_dp=("recall_excecao", "std"),
            precisao_excecao_media=("precisao_excecao", "mean"),
            taxa_encaminhamento_media=("taxa_encaminhamento", "mean"),
            pr_auc_media=("pr_auc_excecao", "mean"),
        )
        .sort_values("recall_excecao_medio", ascending=False)
    )
    summary.to_csv(output_dir / "decision_summary.csv")

    _plot_pr_curves(first_seed_curves, output_dir / "pr_curves.png")
    _plot_recall_boxplot(per_seed, output_dir / "recall_boxplot.png")

    print("\n=== Exception recall under precision >= "
          f"{cfg['comparison']['min_precision']} ===")
    print(summary.to_string())
    print("\n=== Paired Wilcoxon (Holm) ===")
    print(wilcoxon.to_string(index=False))

    return per_seed
