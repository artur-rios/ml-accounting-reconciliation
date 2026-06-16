from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_all(results: dict, n_classes: int, output_dir: Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    roc_fig, roc_ax = plt.subplots() if n_classes == 2 else (None, None)

    for name, r in results.items():
        est = r["estimator"]
        X_test, y_test = r["X_test"], r["y_test"]
        y_pred = est.predict(X_test)

        row = {
            "algorithm": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "cv_mean": r["cv_score_mean"],
            "cv_std": r["cv_score_std"],
        }

        if n_classes == 2 and hasattr(est, "predict_proba"):
            y_prob = est.predict_proba(X_test)[:, 1]
            try:
                row["roc_auc"] = roc_auc_score(y_test, y_prob)
                RocCurveDisplay.from_estimator(est, X_test, y_test, ax=roc_ax, name=name)
            except ValueError:
                row["roc_auc"] = float("nan")

        rows.append(row)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        fig, ax = plt.subplots()
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"Confusion Matrix — {name}")
        fig.savefig(output_dir / f"confusion_matrix_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Feature importances (Random Forest only)
        clf = getattr(est, "named_steps", {}).get("clf", est)
        if hasattr(clf, "feature_importances_") and hasattr(X_test, "columns"):
            imp = pd.Series(clf.feature_importances_, index=X_test.columns)
            fig, ax = plt.subplots()
            imp.sort_values().plot.barh(ax=ax)
            ax.set_title("Feature Importances — Random Forest")
            fig.savefig(output_dir / "feature_importance_rf.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    if roc_fig is not None:
        roc_ax.set_title("ROC Curve — Exact Scenario")
        roc_fig.savefig(output_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close(roc_fig)

    df_metrics = pd.DataFrame(rows).set_index("algorithm")
    df_metrics.to_csv(output_dir / "metrics_summary.csv")
    return df_metrics
