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
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def choose_threshold(
    y_val: pd.Series,
    proba_excecao: np.ndarray,
    min_precision: float,
    precision_margin: float = 0.0,
) -> float | None:
    """Highest-recall threshold whose exception precision meets the floor.

    Returns None when no threshold reaches min_precision -- the caller must then
    report the algorithm as unable to satisfy the operating constraint.

    ``precision_margin`` exists because maximising recall over every candidate
    cut point is a selection procedure, not just an estimate: with a few
    thousand thresholds on the validation curve, the winning point is
    systematically the one whose validation precision clears the floor by
    luck, so the same threshold lands *below* the floor on test more often
    than sampling noise alone would explain. This is the winner's curse, and
    it is why the reported test precision sits just under 0.90 for two of the
    three algorithms. Requiring ``min_precision + precision_margin`` on
    validation buys back that optimism at the cost of some recall.

    The default is 0.0, which reproduces the behaviour under which every
    result in `data/results/comparison/` was produced. Raising it changes the
    operating point and therefore every reported metric -- do not change the
    default without regenerating the artifacts.
    """
    is_exception = (np.asarray(y_val) == 0).astype(int)
    precision, recall, thresholds = precision_recall_curve(is_exception, proba_excecao)

    required_precision = min_precision + precision_margin

    # precision_recall_curve returns one more point than thresholds
    best_threshold, best_recall = None, -1.0
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= required_precision and r > best_recall:
            best_threshold, best_recall = float(t), float(r)
    return best_threshold


def exception_metrics(
    y_true: pd.Series, proba_excecao: np.ndarray, threshold: float
) -> dict[str, float]:
    """Exception-class metrics at a fixed threshold, plus threshold-free context.

    The four confusion-matrix cells are returned alongside the aggregates
    because this study's own recommendation is to report the full matrix and
    not only the summary statistics -- it was the 130/0 asymmetry of the main
    experiment, visible only in the matrix, that revealed the models erring in
    the direction that silences divergences. The cells are counted with the
    *exception* (label 0) as the positive class, matching every other metric
    in this module: `vp_excecao` is a divergence correctly referred to manual
    review, `fn_excecao` is a divergence silently approved.
    """
    y_true = np.asarray(y_true)
    is_exception = (y_true == 0).astype(int)
    flagged = (np.asarray(proba_excecao) >= threshold).astype(int)

    y_pred = np.where(flagged == 1, 0, 1)

    vn, fp, fn, vp = confusion_matrix(is_exception, flagged, labels=[0, 1]).ravel()

    return {
        "recall_excecao": float(recall_score(is_exception, flagged, zero_division=0)),
        "precisao_excecao": float(precision_score(is_exception, flagged, zero_division=0)),
        "taxa_encaminhamento": float(flagged.mean()),
        "pr_auc_excecao": float(average_precision_score(is_exception, proba_excecao)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "vp_excecao": int(vp),
        "fp_excecao": int(fp),
        "fn_excecao": int(fn),
        "vn_excecao": int(vn),
    }
