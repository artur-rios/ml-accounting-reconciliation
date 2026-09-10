"""Fit the three algorithms on one training partition.

The pipelines and hyperparameter grids are reused verbatim from the original
experiment, so any difference in outcome is attributable to the data
representation rather than to a changed search space. Only the scoring function
and the seed differ: average precision of the exception class (label 0), to
match the decision criterion -- every reported metric and the whole decision
rule are about label 0, so model selection must target it too, not sklearn's
default positive class (label 1, the genuine pair).
"""

import copy

import pandas as pd
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from reconciliacao.models.trainer import get_pipelines

_SCORERS = {
    "average_precision_excecao": make_scorer(
        average_precision_score, response_method="predict_proba", pos_label=0
    ),
}


def _resolve_scoring(name: str):
    """Map a config scoring name to a scorer, rejecting anything unrecognised.

    A bare sklearn scorer name (e.g. "average_precision") is deliberately not
    accepted here: it would score the genuine-pair class (label 1), silently
    optimising model selection for the wrong side of the decision rule.
    """
    try:
        return _SCORERS[name]
    except KeyError:
        raise ValueError(
            f"Unrecognised comparison.scoring {name!r}; expected one of "
            f"{sorted(_SCORERS)}."
        ) from None


def fit_algorithms(
    X_train: pd.DataFrame, y_train: pd.Series, cfg: dict, seed: int
) -> dict[str, object]:
    """Return {algorithm_name: fitted best estimator} for one seed."""
    comparison_cfg = cfg["comparison"]
    scoring = _resolve_scoring(comparison_cfg["scoring"])

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
            scoring=scoring,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        models[name] = search.best_estimator_
    return models
