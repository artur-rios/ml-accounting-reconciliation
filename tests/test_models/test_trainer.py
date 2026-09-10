import numpy as np
import pandas as pd
import pytest
from pathlib import Path
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
            "logistic_regression": {"l1_ratio": 0, "class_weight": "balanced"},
            "cv_folds": 2,
            "scoring": "f1_macro",
            "test_size": 0.25,
        },
    }


def test_get_pipelines_returns_three_algorithms(mini_config):
    pipelines = get_pipelines(mini_config)
    assert set(pipelines.keys()) == {"random_forest", "svm", "logistic_regression"}


def test_train_all_saves_model_files(small_dataset, mini_config, tmp_path):
    X, y = small_dataset
    train_all(X, y, mini_config, tmp_path / "models")
    assert (tmp_path / "models" / "random_forest.joblib").exists()
    assert (tmp_path / "models" / "svm.joblib").exists()
    assert (tmp_path / "models" / "logistic_regression.joblib").exists()


def test_train_all_returns_results_dict(small_dataset, mini_config, tmp_path):
    X, y = small_dataset
    results = train_all(X, y, mini_config, tmp_path / "models")
    assert set(results.keys()) == {"random_forest", "svm", "logistic_regression"}
    for r in results.values():
        assert "estimator" in r
        assert "X_test" in r
        assert "y_test" in r
        assert "cv_score_mean" in r


def test_train_all_estimator_can_predict(small_dataset, mini_config, tmp_path):
    X, y = small_dataset
    results = train_all(X, y, mini_config, tmp_path / "models")
    for r in results.values():
        preds = r["estimator"].predict(r["X_test"])
        assert len(preds) == len(r["y_test"])
