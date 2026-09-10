from pathlib import Path

import pandas as pd
import pytest

from reconciliacao.comparison import run


@pytest.mark.integration
def test_end_to_end_produces_the_decision_artifacts(tmp_path, sample_config, comparison_config):
    cfg = dict(sample_config)
    comparison_config["n_records"] = 400
    comparison_config["cv_folds"] = 2
    cfg["comparison"] = comparison_config

    summary = run(cfg, n_seeds=2, output_dir=tmp_path)

    assert len(summary) == 6  # 2 seeds x 3 algorithms
    assert set(summary["algorithm"]) == {"random_forest", "svm", "logistic_regression"}
    for column in ["recall_excecao", "precisao_excecao", "taxa_encaminhamento"]:
        assert column in summary.columns

    for name in ["per_seed_metrics.csv", "wilcoxon.csv", "decision_summary.csv",
                 "leak_guard_report.csv", "pr_curves.png", "recall_boxplot.png"]:
        assert (tmp_path / name).exists(), name


@pytest.mark.integration
def test_threshold_is_chosen_on_validation_not_test(tmp_path, sample_config, comparison_config):
    """The reported test precision may fall below the floor; the validation one may not."""
    cfg = dict(sample_config)
    comparison_config["n_records"] = 400
    comparison_config["cv_folds"] = 2
    cfg["comparison"] = comparison_config

    summary = run(cfg, n_seeds=1, output_dir=tmp_path)
    applicable = summary[summary["threshold"].notna()]
    assert len(applicable) > 0
    assert (applicable["precisao_validacao"] >= cfg["comparison"]["min_precision"] - 1e-9).all()
