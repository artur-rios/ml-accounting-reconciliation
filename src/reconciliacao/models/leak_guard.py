"""Fail the pipeline when any single feature separates the classes.

A label that is a deterministic function of one feature is recoverable by a
depth-1 tree. This guard is the regression test the original experiment lacked:
accuracy above the threshold means the label leaked into the features, not that
the model learned something.
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier


class LeakDetected(Exception):
    """Raised when a single feature predicts the label too well."""


def stump_accuracies(
    X: pd.DataFrame, y: pd.Series, cv: int = 3, seed: int = 42
) -> pd.Series:
    """Cross-validated accuracy of a depth-1 tree fitted on each feature alone."""
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    scores = {}
    for coluna in X.columns:
        stump = DecisionTreeClassifier(max_depth=1, random_state=seed)
        scores[coluna] = float(
            cross_val_score(stump, X[[coluna]], y, cv=folds, scoring="accuracy").mean()
        )
    return pd.Series(scores).sort_values(ascending=False)


def assert_no_leak(
    X: pd.DataFrame,
    y: pd.Series,
    max_accuracy: float,
    cv: int = 3,
    seed: int = 42,
) -> pd.Series:
    """Return the per-feature report, raising LeakDetected if any feature exceeds max_accuracy."""
    report = stump_accuracies(X, y, cv=cv, seed=seed)
    culpadas = report[report > max_accuracy]
    if not culpadas.empty:
        detalhe = ", ".join(f"{nome}={valor:.4f}" for nome, valor in culpadas.items())
        raise LeakDetected(
            f"single-feature accuracy above {max_accuracy}: {detalhe}. "
            "The label is recoverable from a feature in isolation."
        )
    return report
