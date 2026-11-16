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


def test_given_a_training_partition_when_fitted_then_all_three_algorithms_are_returned(tiny_cfg, small_grids):
    X, y = _xy()
    models = fit_algorithms(X, y, tiny_cfg, seed=42)
    assert set(models) == {"random_forest", "svm", "logistic_regression"}


def test_given_fitted_algorithms_when_asked_for_probabilities_then_each_returns_a_distribution(tiny_cfg, small_grids):
    X, y = _xy()
    models = fit_algorithms(X, y, tiny_cfg, seed=42)
    for name, model in models.items():
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2), name
        assert np.allclose(proba.sum(axis=1), 1.0), name


def test_given_the_same_seed_when_fitted_twice_then_the_forests_agree(tiny_cfg, small_grids):
    X, y = _xy()
    a = fit_algorithms(X, y, tiny_cfg, seed=1)["random_forest"]
    b = fit_algorithms(X, y, tiny_cfg, seed=1)["random_forest"]
    assert np.array_equal(a.predict(X), b.predict(X))
    assert a.named_steps["clf"].random_state == 1


def test_given_a_config_when_fitted_then_the_callers_config_is_unchanged(tiny_cfg, small_grids):
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


def test_given_an_unknown_scoring_name_when_resolved_then_it_is_refused():
    with pytest.raises(ValueError):
        _resolve_scoring("average_precision")  # the wrong-class sklearn scorer name


def test_given_two_rankings_when_scored_then_the_exception_class_is_the_one_measured():
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


def test_given_fitted_models_when_scored_then_the_scorer_matches_a_manual_label_zero_computation(tiny_cfg, small_grids):
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
