"""Decision rule derived from the control objective, not from convention.

The exception class is label 0 -- the payment whose pairing is not genuine.
Selecting a model by F1-macro treats a silenced divergence and a redundant
manual review as equally costly; in reconciliation they are not. The rule here
maximises exception recall subject to a precision floor, and reports the
referral rate that comes with it.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def choose_threshold(
    y_val: pd.Series, proba_excecao: np.ndarray, min_precision: float
) -> float | None:
    """Highest-recall threshold whose exception precision meets the floor.

    Returns None when no threshold reaches min_precision -- the caller must then
    report the algorithm as unable to satisfy the operating constraint.
    """
    is_exception = (np.asarray(y_val) == 0).astype(int)
    precision, recall, thresholds = precision_recall_curve(is_exception, proba_excecao)

    # precision_recall_curve returns one more point than thresholds
    best_threshold, best_recall = None, -1.0
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= min_precision and r > best_recall:
            best_threshold, best_recall = float(t), float(r)
    return best_threshold


def exception_metrics(
    y_true: pd.Series, proba_excecao: np.ndarray, threshold: float
) -> dict[str, float]:
    """Exception-class metrics at a fixed threshold, plus threshold-free context."""
    y_true = np.asarray(y_true)
    is_exception = (y_true == 0).astype(int)
    flagged = (np.asarray(proba_excecao) >= threshold).astype(int)

    y_pred = np.where(flagged == 1, 0, 1)

    return {
        "recall_excecao": float(recall_score(is_exception, flagged, zero_division=0)),
        "precisao_excecao": float(precision_score(is_exception, flagged, zero_division=0)),
        "taxa_encaminhamento": float(flagged.mean()),
        "pr_auc_excecao": float(average_precision_score(is_exception, proba_excecao)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
