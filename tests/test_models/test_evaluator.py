import numpy as np
import pandas as pd
import pytest
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


def test_given_results_when_evaluated_then_a_dataframe_is_returned(dummy_results, tmp_path):
    result = evaluate_all(dummy_results, n_classes=2, output_dir=tmp_path)
    assert isinstance(result, pd.DataFrame)


def test_given_results_when_evaluated_then_the_expected_metric_columns_exist(dummy_results, tmp_path):
    result = evaluate_all(dummy_results, n_classes=2, output_dir=tmp_path)
    for col in ("accuracy", "f1_macro", "precision_macro", "recall_macro", "cv_mean"):
        assert col in result.columns


def test_given_results_when_evaluated_then_the_metrics_csv_is_saved(dummy_results, tmp_path):
    evaluate_all(dummy_results, n_classes=2, output_dir=tmp_path)
    assert (tmp_path / "metrics_summary.csv").exists()


def test_given_results_when_evaluated_then_a_confusion_matrix_png_is_saved(dummy_results, tmp_path):
    evaluate_all(dummy_results, n_classes=2, output_dir=tmp_path)
    assert (tmp_path / "confusion_matrix_dummy.png").exists()


@pytest.fixture
def forest_results():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    rng = np.random.RandomState(0)
    X = pd.DataFrame({"a": rng.rand(40), "b": rng.rand(40)})
    y = pd.Series((X["a"] > 0.5).astype(int))
    model = Pipeline([("clf", RandomForestClassifier(n_estimators=5, random_state=0))]).fit(X, y)
    return {"random_forest": {"estimator": model, "X_test": X, "y_test": y,
                              "cv_score_mean": 0.9, "cv_score_std": 0.01}}


def test_given_a_binary_scenario_with_probabilities_when_evaluated_then_roc_auc_and_its_curve_are_produced(
        forest_results, tmp_path):
    result = evaluate_all(forest_results, n_classes=2, output_dir=tmp_path)
    assert 0.0 <= result.loc["random_forest", "roc_auc"] <= 1.0
    assert (tmp_path / "roc_curve.png").exists()


def test_given_a_random_forest_when_evaluated_then_its_feature_importances_are_plotted(forest_results, tmp_path):
    evaluate_all(forest_results, n_classes=2, output_dir=tmp_path)
    assert (tmp_path / "feature_importance_rf.png").exists()


def test_given_a_test_set_with_a_single_class_when_evaluated_then_roc_auc_is_nan_rather_than_an_error(
        forest_results, tmp_path):
    r = forest_results["random_forest"]
    single = r["y_test"] == 1
    r["X_test"], r["y_test"] = r["X_test"][single], r["y_test"][single]
    result = evaluate_all(forest_results, n_classes=2, output_dir=tmp_path)
    assert np.isnan(result.loc["random_forest", "roc_auc"])


def test_given_three_classes_when_evaluated_then_no_roc_curve_is_drawn(forest_results, tmp_path):
    result = evaluate_all(forest_results, n_classes=3, output_dir=tmp_path)
    assert "roc_auc" not in result.columns
    assert not (tmp_path / "roc_curve.png").exists()


def test_given_metrics_when_evaluated_then_the_csv_round_trips_the_returned_frame(dummy_results, tmp_path):
    result = evaluate_all(dummy_results, n_classes=2, output_dir=tmp_path)
    saved = pd.read_csv(tmp_path / "metrics_summary.csv", index_col="algorithm")
    pd.testing.assert_frame_equal(saved, result, check_dtype=False)
    assert result.loc["dummy", "accuracy"] == pytest.approx(0.5)
    assert result.loc["dummy", "cv_mean"] == 0.5
