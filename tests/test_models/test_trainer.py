import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV

from reconciliacao.models.trainer import get_pipelines, train_all


@pytest.fixture
def small_dataset():
    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        "delta_days": rng.uniform(0, 30, 60).tolist(),
        "delta_valor_pct": rng.uniform(0, 20, 60).tolist(),
        "valor_pago": rng.uniform(500, 50000, 60).tolist(),
        "nfse_valor_iss": rng.uniform(10, 500, 60).tolist(),
        "nfse_aliquota": rng.uniform(2, 5, 60).tolist(),
        "cnpj_match": [1] * 40 + [0] * 20,
        "descricao_similarity": rng.uniform(0, 1, 60).tolist(),
    })
    y = pd.Series([0] * 20 + [1] * 20 + [2] * 20)
    return X, y


@pytest.fixture
def mini_config():
    return {
        "simulation": {"random_seed": 42},
        "models": {
            "random_forest": {"n_estimators": 10, "class_weight": "balanced_subsample"},
            "svm": {"kernel": "rbf", "class_weight": "balanced"},
            "logistic_regression": {"class_weight": "balanced"},
            "cv_folds": 2,
            "scoring": "f1_macro",
            "test_size": 0.25,
        },
    }


@pytest.fixture
def trained(small_dataset, mini_config, tmp_path, small_grids):
    X, y = small_dataset
    return train_all(X, y, mini_config, tmp_path / "models"), tmp_path / "models"


def test_given_a_config_when_pipelines_are_built_then_there_are_three_algorithms(mini_config):
    pipelines = get_pipelines(mini_config)
    assert set(pipelines.keys()) == {"random_forest", "svm", "logistic_regression"}


def test_given_a_config_when_pipelines_are_built_then_each_carries_the_published_search_grid(mini_config):
    """The grids other tests shrink (conftest.SMALL_GRIDS) are pinned here."""
    grids = {name: grid for name, (_estimator, grid) in get_pipelines(mini_config).items()}
    assert grids == {
        "random_forest": {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10]},
        "svm": {"clf__estimator__C": [0.1, 1.0, 10.0]},
        "logistic_regression": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    }


def test_given_a_config_when_pipelines_are_built_then_its_settings_and_seed_reach_the_estimators(mini_config):
    pipelines = {name: estimator for name, (estimator, _grid) in get_pipelines(mini_config).items()}
    forest = pipelines["random_forest"].named_steps["clf"]
    assert forest.n_estimators == 10
    assert forest.class_weight == "balanced_subsample"
    assert forest.random_state == 42

    svm = pipelines["svm"].named_steps["clf"]
    assert isinstance(svm, CalibratedClassifierCV)
    assert svm.estimator.kernel == "rbf"
    assert svm.estimator.random_state == 42
    assert list(pipelines["svm"].named_steps) == ["scaler", "clf"]

    logistic = pipelines["logistic_regression"].named_steps["clf"]
    assert logistic.class_weight == "balanced"
    assert logistic.max_iter == 1000


def test_given_a_dataset_when_trained_then_each_model_is_saved(trained):
    _results, models_dir = trained
    for name in ("random_forest", "svm", "logistic_regression"):
        assert (models_dir / f"{name}.joblib").exists()


def test_given_a_dataset_when_trained_then_the_results_carry_the_estimator_test_split_and_cv_scores(
        trained, small_dataset, mini_config):
    results, _models_dir = trained
    X, _y = small_dataset
    assert set(results.keys()) == {"random_forest", "svm", "logistic_regression"}
    for r in results.values():
        assert len(r["X_test"]) == round(len(X) * mini_config["models"]["test_size"])
        assert r["y_test"].index.equals(r["X_test"].index)
        assert 0.0 <= r["cv_score_mean"] <= 1.0
        assert r["cv_score_std"] >= 0.0


def test_given_a_dataset_when_trained_then_the_split_is_stratified(trained):
    results, _models_dir = trained
    counts = results["svm"]["y_test"].value_counts()
    assert sorted(counts.index) == [0, 1, 2]
    assert counts.max() - counts.min() <= 1


def test_given_a_trained_model_when_it_predicts_then_there_is_one_prediction_per_test_row(trained):
    results, _models_dir = trained
    for r in results.values():
        preds = r["estimator"].predict(r["X_test"])
        assert len(preds) == len(r["y_test"])
