import math

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

import reconciliacao.comparison as comparison_module
from reconciliacao.comparison import run, run_seed
from reconciliacao.models.comparison_stats import paired_comparisons


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


_ALGORITHM_ORDER = ["random_forest", "svm", "logistic_regression"]


def _stub_fit_algorithms(X_train, y_train, cfg, seed):
    """Trivially cheap stand-in for fit_algorithms: a DummyClassifier per
    algorithm name, fit almost instantly and ignoring the actual feature
    values, so run_seed's own orchestration can be tested without paying for
    GridSearchCV over the real pipelines."""
    return {
        name: DummyClassifier(strategy="stratified", random_state=seed).fit(X_train, y_train)
        for name in _ALGORITHM_ORDER
    }


def test_run_seed_pins_partitions_supplier_terms_and_no_threshold_branch(
    monkeypatch, sample_config, comparison_config
):
    """Direct test of run_seed (not the slow end-to-end run), pinning three
    properties that the whole-pipeline integration tests do not touch:

    - the three partitions are disjoint, stratified, and sized per `split`
    - supplier terms are fitted on the training partition alone
    - the choose_threshold-is-None branch runs cleanly and feeds a row that
      paired_comparisons can consume without raising
    """
    cfg = dict(sample_config)
    comparison_config["n_records"] = 400
    cfg["comparison"] = comparison_config

    monkeypatch.setattr(comparison_module, "fit_algorithms", _stub_fit_algorithms)

    # Spy on the partition split to capture train/val/test without duplicating
    # its logic, and spy on fit_supplier_terms to see which frame it was
    # called with.
    real_split = comparison_module._split_three_ways
    captured_split = {}

    def spy_split(df, cmp_cfg, seed):
        train_df, val_df, test_df = real_split(df, cmp_cfg, seed)
        captured_split["train"] = train_df
        captured_split["val"] = val_df
        captured_split["test"] = test_df
        return train_df, val_df, test_df

    monkeypatch.setattr(comparison_module, "_split_three_ways", spy_split)

    real_fit_supplier_terms = comparison_module.fit_supplier_terms
    supplier_terms_calls = []

    def spy_fit_supplier_terms(df_train):
        supplier_terms_calls.append(df_train)
        return real_fit_supplier_terms(df_train)

    monkeypatch.setattr(comparison_module, "fit_supplier_terms", spy_fit_supplier_terms)

    # Force the first algorithm of each seed to miss the precision floor, so
    # the choose_threshold-is-None branch is exercised deterministically
    # regardless of what the (stubbed) model actually predicts.
    real_choose_threshold = comparison_module.choose_threshold
    call_count = {"n": 0}

    def fake_choose_threshold(y_val, proba, min_precision):
        call_count["n"] += 1
        if call_count["n"] % len(_ALGORITHM_ORDER) == 1:
            return None
        return real_choose_threshold(y_val, proba, min_precision)

    monkeypatch.setattr(comparison_module, "choose_threshold", fake_choose_threshold)

    all_rows = []
    for seed in (0, 1):
        rows, _leak_report, _curves = run_seed(cfg, seed)
        all_rows.extend(rows)

    # --- supplier terms fitted on the training partition alone ---
    # captured_split holds only the last seed's partitions (each seed
    # overwrites it), so compare against that same last call.
    assert len(supplier_terms_calls) == 2  # once per seed, never per partition
    assert supplier_terms_calls[-1].index.equals(captured_split["train"].index)
    assert not supplier_terms_calls[-1].index.equals(captured_split["val"].index)
    assert not supplier_terms_calls[-1].index.equals(captured_split["test"].index)

    # --- partitions disjoint, stratified, sized per `split` config ---
    train_idx = set(captured_split["train"].index)
    val_idx = set(captured_split["val"].index)
    test_idx = set(captured_split["test"].index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)

    total = len(train_idx) + len(val_idx) + len(test_idx)
    fractions = cfg["comparison"]["split"]
    assert len(train_idx) / total == pytest.approx(fractions["train"], abs=0.02)
    assert len(val_idx) / total == pytest.approx(fractions["val"], abs=0.02)
    assert len(test_idx) / total == pytest.approx(fractions["test"], abs=0.02)

    overall_labels = pd.concat(
        [captured_split["train"]["label"], captured_split["val"]["label"], captured_split["test"]["label"]]
    )
    overall_exception_rate = (overall_labels == 0).mean()
    for part in ("train", "val", "test"):
        part_rate = (captured_split[part]["label"] == 0).mean()
        assert part_rate == pytest.approx(overall_exception_rate, abs=0.08)

    # --- choose_threshold is None branch: row shape and downstream safety ---
    no_threshold_rows = [r for r in all_rows if r["threshold"] is None]
    assert len(no_threshold_rows) >= 1
    for row in no_threshold_rows:
        assert row["precisao_validacao"] is None
        assert row["recall_excecao"] == 0.0
        assert row["precisao_excecao"] == 0.0
        assert row["taxa_encaminhamento"] == 0.0
        assert not math.isnan(row["pr_auc_excecao"])  # threshold-free, still computable (F2)
        assert math.isnan(row["f1_macro"])

    # paired_comparisons must not raise even though some rows come from the
    # None-threshold branch -- this is the guard the branch exists to satisfy.
    per_seed = pd.DataFrame(all_rows)
    wilcoxon = paired_comparisons(per_seed, metric="recall_excecao")
    assert len(wilcoxon) == 3  # 3 pairwise comparisons among the 3 algorithms
