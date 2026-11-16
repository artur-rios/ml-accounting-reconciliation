import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

SCRIPT = Path(__file__).parent.parent / "run_pipeline.py"


@pytest.mark.integration
@pytest.mark.parametrize("scenario, n_classes", [("exact", 2), ("fuzzy", 3)])
def test_given_a_tiny_config_when_the_pipeline_script_runs_then_every_artifact_is_written(
        tmp_path, monkeypatch, small_grids, scenario, n_classes):
    """Smoke test: run_pipeline.py end to end on a tiny dataset.

    The script runs in this process (runpy, as `python run_pipeline.py` would)
    so the one-point grids from conftest apply: the point is the wiring from
    simulation to metrics, not a real hyperparameter search.
    """
    cfg = {
        "simulation": {"n_records": 80, "conciliation_rate": 0.70, "random_seed": 42},
        "scenarios": {
            "exact": {"date_tolerance_days": 0, "value_tolerance_pct": 0.0, "n_classes": 2},
            "fuzzy": {"date_tolerance_days": 5, "value_tolerance_pct": 2.0, "n_classes": 3},
        },
        "models": {
            "random_forest": {"n_estimators": 5, "class_weight": "balanced_subsample"},
            "svm": {"kernel": "rbf", "class_weight": "balanced"},
            "logistic_regression": {"class_weight": "balanced"},
            "cv_folds": 2,
            "scoring": "f1_macro",
            "test_size": 0.20,
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(cfg), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--scenario", scenario])

    runpy.run_path(str(SCRIPT), run_name="__main__")

    results = tmp_path / "data" / "results" / scenario
    metrics = pd.read_csv(results / "metrics_summary.csv", index_col="algorithm")
    assert set(metrics.index) == {"random_forest", "svm", "logistic_regression"}
    for name in ("random_forest", "svm", "logistic_regression"):
        assert (results / "models" / f"{name}.joblib").exists()
        assert (results / f"confusion_matrix_{name}.png").exists()
    assert (results / "roc_curve.png").exists() is (n_classes == 2)
    assert (tmp_path / "data" / "raw" / "pagamentos.xlsx").exists()
    assert (tmp_path / "data" / "raw" / "nfse.xml").exists()
    assert (tmp_path / "data" / "processed" / f"{scenario}_reconciliado.csv").exists()
