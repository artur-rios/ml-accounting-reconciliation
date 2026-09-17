import numpy as np
import pandas as pd
import pytest

from reconciliacao.models.decision import choose_threshold, exception_metrics


def test_chooses_threshold_that_meets_the_precision_floor():
    y = pd.Series([0, 0, 0, 1, 1, 1, 1, 1])
    proba = np.array([0.95, 0.90, 0.85, 0.40, 0.30, 0.20, 0.10, 0.05])
    limiar = choose_threshold(y, proba, min_precision=0.90)
    assert limiar is not None
    metricas = exception_metrics(y, proba, limiar)
    assert metricas["precisao_excecao"] >= 0.90


def test_returns_none_when_the_precision_floor_is_unreachable():
    y = pd.Series([0, 1, 1, 1, 1, 1, 1, 1])
    proba = np.array([0.10, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89])
    assert choose_threshold(y, proba, min_precision=0.90) is None


def test_maximises_recall_subject_to_the_constraint():
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    proba = np.array([0.99, 0.80, 0.70, 0.60, 0.20, 0.15, 0.10, 0.05])
    limiar = choose_threshold(y, proba, min_precision=0.90)
    metricas = exception_metrics(y, proba, limiar)
    # every exception is separable above 0.60 with no false alarm
    assert metricas["recall_excecao"] == pytest.approx(1.0)
    assert metricas["precisao_excecao"] == pytest.approx(1.0)


def test_referral_rate_is_the_flagged_fraction():
    y = pd.Series([0, 0, 1, 1, 1, 1, 1, 1])
    proba = np.array([0.99, 0.98, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    metricas = exception_metrics(y, proba, threshold=0.50)
    assert metricas["taxa_encaminhamento"] == pytest.approx(2 / 8)


def test_metrics_include_pr_auc_and_f1():
    y = pd.Series([0, 0, 1, 1])
    proba = np.array([0.9, 0.8, 0.2, 0.1])
    metricas = exception_metrics(y, proba, threshold=0.5)
    assert 0.0 <= metricas["pr_auc_excecao"] <= 1.0
    assert 0.0 <= metricas["f1_macro"] <= 1.0


def test_precision_margin_tightens_the_validation_requirement():
    """The margin exists to buy back the winner's curse: maximising recall
    over thousands of candidate cut points systematically selects a threshold
    whose validation precision clears the floor by luck. Requiring a margin
    must never loosen the choice, and must be a no-op at 0.0 so the published
    artifacts stay reproducible."""
    rng = np.random.default_rng(0)
    y_val = pd.Series(rng.integers(0, 2, size=400))
    proba = np.where(y_val == 0, rng.uniform(0.35, 1.0, 400), rng.uniform(0.0, 0.65, 400))

    sem_margem = choose_threshold(y_val, proba, min_precision=0.90)
    explicito_zero = choose_threshold(y_val, proba, min_precision=0.90, precision_margin=0.0)
    com_margem = choose_threshold(y_val, proba, min_precision=0.90, precision_margin=0.05)

    assert explicito_zero == sem_margem  # default is exactly the published behaviour
    assert sem_margem is not None
    # A stricter requirement can only move the cut point up, or find none.
    assert com_margem is None or com_margem >= sem_margem


def test_exception_metrics_reports_the_confusion_cells():
    """The full matrix, not only the aggregates -- the study's own checklist
    item 7, counted with the exception (label 0) as the positive class."""
    y_true = pd.Series([0, 0, 0, 1, 1, 1, 1, 1])
    proba = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1])

    metrics = exception_metrics(y_true, proba, threshold=0.5)

    assert metrics["vp_excecao"] == 2   # exceptions correctly referred
    assert metrics["fn_excecao"] == 1   # divergence silently approved
    assert metrics["fp_excecao"] == 1   # redundant manual review
    assert metrics["vn_excecao"] == 4
    total = sum(metrics[c] for c in ["vp_excecao", "fn_excecao", "fp_excecao", "vn_excecao"])
    assert total == len(y_true)
    assert metrics["recall_excecao"] == pytest.approx(2 / 3)
    assert metrics["precisao_excecao"] == pytest.approx(2 / 3)
