import numpy as np
import pandas as pd
import pytest

from reconciliacao.models.comparison_trainer import fit_algorithms


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
