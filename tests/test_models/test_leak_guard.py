import numpy as np
import pandas as pd
import pytest

from reconciliacao.models.leak_guard import LeakDetected, assert_no_leak, stump_accuracies


def test_detects_a_deliberately_circular_feature():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, size=300))
    X = pd.DataFrame({
        "ruido": rng.normal(size=300),
        "copia_do_rotulo": y.astype(float),
    })
    with pytest.raises(LeakDetected) as exc:
        assert_no_leak(X, y, max_accuracy=0.95)
    assert "copia_do_rotulo" in str(exc.value)


def test_passes_when_no_single_feature_separates():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, size=300))
    X = pd.DataFrame({
        "a": rng.normal(size=300),
        "b": rng.normal(size=300),
    })
    report = assert_no_leak(X, y, max_accuracy=0.95)
    assert (report < 0.95).all()


def test_report_covers_every_feature():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, size=200))
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    report = stump_accuracies(X, y)
    assert set(report.index) == {"a", "b"}


def test_error_names_every_offending_feature():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, size=300))
    X = pd.DataFrame({
        "copia_1": y.astype(float),
        "copia_2": y.astype(float) * 3.0,
        "ruido": rng.normal(size=300),
    })
    with pytest.raises(LeakDetected) as exc:
        assert_no_leak(X, y, max_accuracy=0.95)
    mensagem = str(exc.value)
    assert "copia_1" in mensagem and "copia_2" in mensagem and "ruido" not in mensagem


def test_raises_on_degenerate_labels():
    """Verify clear error when a class has fewer samples than cv splits."""
    rng = np.random.default_rng(0)
    # Create a 4-sample dataset with only 2 samples of class 1 (cannot split into 3 folds)
    y = pd.Series([0, 0, 1, 1])
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError) as exc:
        stump_accuracies(X, y, cv=3)
    error_msg = str(exc.value)
    assert "Cannot create 3 stratified folds" in error_msg
    assert "minimum count" in error_msg.lower()
