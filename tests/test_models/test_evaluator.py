import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.dummy import DummyClassifier
from reconciliacao.models.evaluator import evaluate_all


@pytest.fixture
def dummy_results(tmp_path):
    rng = np.random.RandomState(42)
    X_test = pd.DataFrame({"a": rng.rand(30), "b": rng.rand(30)})
    y_test = pd.Series([0] * 15 + [1] * 15)
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_test, y_test)
    return {
        "dummy": {
            "estimator": clf,
            "X_test": X_test,
            "y_test": y_test,
            "cv_score_mean": 0.5,
            "cv_score_std": 0.05,
        }
    }


def test_evaluate_all_returns_dataframe(dummy_results, tmp_path):
    result = evaluate_all(dummy_results, n_classes=2, output_dir=tmp_path)
    assert isinstance(result, pd.DataFrame)


def test_evaluate_all_has_expected_columns(dummy_results, tmp_path):
    result = evaluate_all(dummy_results, n_classes=2, output_dir=tmp_path)
    for col in ("accuracy", "f1_macro", "precision_macro", "recall_macro", "cv_mean"):
        assert col in result.columns


def test_evaluate_all_saves_metrics_csv(dummy_results, tmp_path):
    evaluate_all(dummy_results, n_classes=2, output_dir=tmp_path)
    assert (tmp_path / "metrics_summary.csv").exists()


def test_evaluate_all_saves_confusion_matrix_png(dummy_results, tmp_path):
    evaluate_all(dummy_results, n_classes=2, output_dir=tmp_path)
    assert (tmp_path / "confusion_matrix_dummy.png").exists()
