import math

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier

import reconciliacao.comparison as comparison_module
from reconciliacao.comparison import run, run_seed
from reconciliacao.etl.truth_labeler import label_from_truth
from reconciliacao.models.comparison_stats import paired_comparisons
from reconciliacao.models.features_v2 import build_features_v2, fit_supplier_terms
from reconciliacao.simulation.truth_generator import generate_comparison_dataset


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
        assert row["recall_excecao"] == 0.0  # deliberate conservative convention, stays non-NaN
        assert math.isnan(row["precisao_excecao"])  # no threshold to compute it at
        assert math.isnan(row["taxa_encaminhamento"])  # no threshold to compute it at
        assert not math.isnan(row["pr_auc_excecao"])  # threshold-free, still computable (F2)
        assert math.isnan(row["f1_macro"])

    # paired_comparisons must not raise even though some rows come from the
    # None-threshold branch -- this is the guard the branch exists to satisfy.
    per_seed = pd.DataFrame(all_rows)
    wilcoxon = paired_comparisons(per_seed, metric="recall_excecao")
    assert len(wilcoxon) == 3  # 3 pairwise comparisons among the 3 algorithms


class _RowScoreStub(ClassifierMixin, BaseEstimator):
    """A fitted classifier stand-in whose probability estimate is a
    deterministic function of the row's own feature values, and whose
    classes_ order is configurable.

    Two instances built with classes=(0, 1) and classes=(1, 0) represent the
    identical underlying probability model wearing a different column
    arrangement -- exactly what a real sklearn estimator's predict_proba
    would never do (classes_ is always sorted ascending for {0, 1} integer
    labels) but that run_seed must not silently assume. If run_seed reads
    off the exception column (label 0) via classes_ rather than by position,
    both instances must produce identical downstream metrics; the old
    ``[:, 0]`` code would instead invert one of them.
    """

    def __init__(self, classes=(0, 1)):
        self.classes = classes

    def fit(self, X, y):
        self.classes_ = np.array(self.classes)
        return self

    def predict_proba(self, X):
        scores = np.abs(np.sin(X.to_numpy(dtype=float).sum(axis=1) * 12.9898)) % 1.0
        proba_label0 = 1.0 - scores
        proba_label1 = scores
        if list(self.classes) == [0, 1]:
            return np.column_stack([proba_label0, proba_label1])
        return np.column_stack([proba_label1, proba_label0])  # classes_ == [1, 0]


def _make_stub_fit_algorithms(classes):
    def _stub(X_train, y_train, cfg, seed):
        return {name: _RowScoreStub(classes=classes).fit(X_train, y_train) for name in _ALGORITHM_ORDER}
    return _stub


def test_exception_proba_is_read_via_classes_not_column_position(monkeypatch, sample_config, comparison_config):
    """Regression test for the [:, 0] positional bug: predict_proba's column
    order must be resolved through classes_, since it is only [0, 1] because
    integer labels happen to sort that way."""
    cfg = dict(sample_config)
    comparison_config["n_records"] = 400
    cfg["comparison"] = comparison_config

    def run_with_classes(classes):
        monkeypatch.setattr(comparison_module, "fit_algorithms", _make_stub_fit_algorithms(classes))
        rows, _leak_report, _curves = run_seed(cfg, seed=0)
        return pd.DataFrame(rows).sort_values("algorithm").reset_index(drop=True)

    forward = run_with_classes((0, 1))
    reversed_ = run_with_classes((1, 0))

    for column in ["recall_excecao", "pr_auc_excecao"]:
        assert forward[column].to_numpy() == pytest.approx(reversed_[column].to_numpy(), abs=1e-9), column


def _built_features(comparison_config, seed=42):
    """Run the real pipeline from raw generation through build_features_v2,
    exactly as run_seed does, and return (X_train, X_test, pairs, test_df)."""
    df_payments, df_invoices, df_truth = generate_comparison_dataset(comparison_config, seed)
    pairs = label_from_truth(df_payments, df_invoices, df_truth)

    train_df, val_df, test_df = comparison_module._split_three_ways(pairs, comparison_config, seed)
    supplier_terms, global_term = fit_supplier_terms(train_df)

    X_train, _y_train = build_features_v2(train_df, comparison_config, supplier_terms, global_term)
    X_test, _y_test = build_features_v2(test_df, comparison_config, supplier_terms, global_term)
    return X_train, X_test, train_df, test_df


def test_desvio_prazo_fornecedor_has_variance_on_train(comparison_config):
    """Regression test for the defect: one payment per supplier collapsed
    the per-supplier median onto each row's own value, zeroing
    desvio_prazo_fornecedor on every training row."""
    X_train, _X_test, _train_df, _test_df = _built_features(comparison_config)
    assert X_train["desvio_prazo_fornecedor"].nunique() > 1


def test_no_feature_column_is_constant_on_train(comparison_config):
    """Generalises the above across every column -- the test whose absence
    let the defect through, and the one meant to catch the next one."""
    X_train, _X_test, _train_df, _test_df = _built_features(comparison_config)
    constant_columns = [c for c in X_train.columns if X_train[c].nunique() <= 1]
    assert constant_columns == []


def test_some_test_suppliers_are_also_present_in_train(comparison_config):
    """With several invoices per supplier, a supplier can land in both the
    train and test partitions, so the fitted median genuinely transfers
    instead of every test row falling back to the global median."""
    _X_train, _X_test, train_df, test_df = _built_features(comparison_config)
    train_suppliers = set(train_df["cnpj_fornecedor"])
    test_suppliers = set(test_df["cnpj_fornecedor"])
    assert len(train_suppliers & test_suppliers) > 0
