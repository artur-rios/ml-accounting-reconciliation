import subprocess
import sys
from pathlib import Path
import pytest


@pytest.mark.integration
def test_pipeline_exact_scenario_runs_end_to_end(tmp_path, monkeypatch):
    """Smoke test: run the full pipeline on a tiny dataset."""
    import yaml
    cfg = {
        "simulation": {"n_records": 80, "conciliation_rate": 0.70, "random_seed": 42},
        "scenarios": {
            "exact": {"date_tolerance_days": 0, "value_tolerance_pct": 0.0, "n_classes": 2},
            "fuzzy": {"date_tolerance_days": 5, "value_tolerance_pct": 2.0, "n_classes": 3},
        },
        "models": {
            "random_forest": {"n_estimators": 5, "class_weight": "balanced_subsample"},
            "svm": {"kernel": "rbf", "class_weight": "balanced"},
            "logistic_regression": {"l1_ratio": 0, "class_weight": "balanced"},
            "cv_folds": 2,
            "scoring": "f1_macro",
            "test_size": 0.20,
        },
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(cfg))
    monkeypatch.chdir(tmp_path)

    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent / "run_pipeline.py"), "--scenario", "exact"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "data" / "results" / "exact" / "metrics_summary.csv").exists()
