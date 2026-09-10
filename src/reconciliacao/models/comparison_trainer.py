"""Fit the three algorithms on one training partition.

The pipelines and hyperparameter grids are reused verbatim from the original
experiment, so any difference in outcome is attributable to the data
representation rather than to a changed search space. Only the scoring function
and the seed differ: average_precision, to match the decision criterion.
"""

import copy

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from reconciliacao.models.trainer import get_pipelines


def fit_algorithms(
    X_train: pd.DataFrame, y_train: pd.Series, cfg: dict, seed: int
) -> dict[str, object]:
    """Return {algorithm_name: fitted best estimator} for one seed."""
    comparison_cfg = cfg["comparison"]

    seeded_cfg = copy.deepcopy(cfg)
    seeded_cfg["simulation"] = dict(seeded_cfg.get("simulation", {}))
    seeded_cfg["simulation"]["random_seed"] = seed

    folds = StratifiedKFold(
        n_splits=comparison_cfg["cv_folds"], shuffle=True, random_state=seed
    )

    models = {}
    for name, (estimator, param_grid) in get_pipelines(seeded_cfg).items():
        search = GridSearchCV(
            estimator,
            param_grid,
            cv=folds,
            scoring=comparison_cfg["scoring"],
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        models[name] = search.best_estimator_
    return models
