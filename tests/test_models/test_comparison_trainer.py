import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from reconciliacao.models.comparison_trainer import _resolve_scoring, fit_algorithms


@pytest.fixture
def tiny_cfg(sample_config, comparison_config):
    cfg = dict(sample_config)
    cfg["comparison"] = comparison_config
    return cfg


def _xy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.normal(size=n),
    })
    y = pd.Series((X["a"] + X["b"] > 0).astype(int))
    return X, y


def test_fits_all_three_algorithms(tiny_cfg):
    X, y = _xy()
    models = fit_algorithms(X, y, tiny_cfg, seed=42)
    assert set(models) == {"random_forest", "svm", "logistic_regression"}


def test_every_estimator_can_produce_probabilities(tiny_cfg):
    X, y = _xy()
    models = fit_algorithms(X, y, tiny_cfg, seed=42)
    for name, model in models.items():
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2), name
        assert np.allclose(proba.sum(axis=1), 1.0), name


def test_seed_is_threaded_through_to_the_estimators(tiny_cfg):
    X, y = _xy()
    a = fit_algorithms(X, y, tiny_cfg, seed=1)["random_forest"]
    b = fit_algorithms(X, y, tiny_cfg, seed=1)["random_forest"]
    assert np.array_equal(a.predict(X), b.predict(X))


def test_does_not_mutate_the_caller_cfg(tiny_cfg):
    import copy

    X, y = _xy()
    cfg_before = copy.deepcopy(tiny_cfg)
    fit_algorithms(X, y, tiny_cfg, seed=7)
    assert tiny_cfg == cfg_before


class _StubClassifier(ClassifierMixin, BaseEstimator):
    """A fitted classifier stand-in with a fixed predict_proba output."""

    def __init__(self, proba=None):
        self.proba = proba

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.proba


def test_resolve_scoring_rejects_unknown_names():
    with pytest.raises(ValueError):
        _resolve_scoring("average_precision")  # the wrong-class sklearn scorer name


def test_scorer_targets_the_exception_class_not_the_genuine_pair():
    """A model that ranks exceptions (label 0) well must outscore one that
    ranks genuine pairs (label 1) well instead -- proving the configured
    scorer evaluates the exception column, not sklearn's default positive
    class."""
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    ranks_exceptions_well = np.array([
        [0.9, 0.1], [0.8, 0.2], [0.7, 0.3],
        [0.2, 0.8], [0.1, 0.9], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8],
    ])
    ranks_genuine_pairs_well = np.array([
        [0.3, 0.7], [0.4, 0.6], [0.2, 0.8],
        [0.1, 0.9], [0.05, 0.95], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3],
    ])

    scoring = _resolve_scoring("average_precision_excecao")
    good = _StubClassifier(ranks_exceptions_well).fit(None, y)
    bad = _StubClassifier(ranks_genuine_pairs_well).fit(None, y)

    assert scoring(good, None, y) > scoring(bad, None, y)


def test_scorer_agrees_with_manual_pos_label_zero_computation(tiny_cfg):
    """Empirical verification (per the task brief): the configured scorer's
    output must match average_precision_score computed by hand on column 0
    of predict_proba."""
    from sklearn.metrics import average_precision_score

    X, y = _xy()
    models = fit_algorithms(X, y, tiny_cfg, seed=3)
    scoring = _resolve_scoring(tiny_cfg["comparison"]["scoring"])

    for name, model in models.items():
        idx0 = list(model.classes_).index(0)
        manual = average_precision_score((np.asarray(y) == 0).astype(int),
                                          model.predict_proba(X)[:, idx0])
        scored = scoring(model, X, y)
        assert scored == pytest.approx(manual), name
