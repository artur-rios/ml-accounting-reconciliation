import pandas as pd
import pytest

from reconciliacao.models.comparison_stats import paired_comparisons


def _per_seed(values: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for algorithm, series in values.items():
        for seed, value in enumerate(series):
            rows.append({"seed": seed, "algorithm": algorithm, "recall_excecao": value})
    return pd.DataFrame(rows)


def test_produces_one_row_per_pair():
    df = _per_seed({
        "random_forest": [0.80] * 10,
        "svm": [0.70] * 10,
        "logistic_regression": [0.60] * 10,
    })
    out = paired_comparisons(df, metric="recall_excecao")
    assert len(out) == 3
    assert set(out.columns) == {
        "algorithm_a", "algorithm_b", "median_diff",
        "statistic", "p_value", "p_holm", "significant",
    }


def test_detects_a_consistent_difference():
    df = _per_seed({
        "random_forest": [0.80, 0.81, 0.79, 0.82, 0.80, 0.83, 0.78, 0.81, 0.80, 0.82],
        "svm": [0.70, 0.71, 0.69, 0.72, 0.70, 0.73, 0.68, 0.71, 0.70, 0.72],
        "logistic_regression": [0.60] * 10,
    })
    out = paired_comparisons(df, metric="recall_excecao")
    row = out[(out["algorithm_a"] == "random_forest") & (out["algorithm_b"] == "svm")].iloc[0]
    assert row["median_diff"] == pytest.approx(0.10, abs=0.005)
    assert row["p_holm"] < 0.05
    assert bool(row["significant"]) is True


def test_reports_no_significance_for_identical_performance():
    df = _per_seed({
        "random_forest": [0.80, 0.70, 0.75, 0.82, 0.68, 0.79, 0.71, 0.77, 0.73, 0.76],
        "svm": [0.80, 0.70, 0.75, 0.82, 0.68, 0.79, 0.71, 0.77, 0.73, 0.76],
        "logistic_regression": [0.60] * 10,
    })
    out = paired_comparisons(df, metric="recall_excecao")
    row = out[(out["algorithm_a"] == "random_forest") & (out["algorithm_b"] == "svm")].iloc[0]
    assert row["median_diff"] == pytest.approx(0.0)
    assert bool(row["significant"]) is False


def test_holm_correction_is_no_smaller_than_the_raw_p():
    df = _per_seed({
        "random_forest": [0.80, 0.81, 0.79, 0.82, 0.80, 0.83, 0.78, 0.81, 0.80, 0.82],
        "svm": [0.70, 0.71, 0.69, 0.72, 0.70, 0.73, 0.68, 0.71, 0.70, 0.72],
        "logistic_regression": [0.60, 0.61, 0.59, 0.62, 0.60, 0.63, 0.58, 0.61, 0.60, 0.62],
    })
    out = paired_comparisons(df, metric="recall_excecao")
    assert (out["p_holm"] >= out["p_value"] - 1e-12).all()
    assert (out["p_holm"] <= 1.0).all()


def test_raises_on_missing_seed_algorithm_combination():
    """Missing (seed, algorithm) cells after pivot must raise ValueError naming the missing data."""
    # Create a DataFrame with missing data: logistic_regression has no seed 5
    df = _per_seed({
        "random_forest": [0.80, 0.81, 0.79, 0.82, 0.80, 0.83, 0.78, 0.81, 0.80, 0.82],
        "svm": [0.70, 0.71, 0.69, 0.72, 0.70, 0.73, 0.68, 0.71, 0.70, 0.72],
        "logistic_regression": [0.60, 0.61, 0.59, 0.62, 0.60],  # Only 5 seeds, not 10
    })
    with pytest.raises(ValueError) as exc_info:
        paired_comparisons(df, metric="recall_excecao")

    # Verify error message names the offending algorithm and seed
    error_msg = str(exc_info.value)
    assert "Missing (seed, algorithm) combinations" in error_msg
    assert "logistic_regression" in error_msg
    assert "seed=" in error_msg


def test_complete_frame_does_not_raise():
    """A complete (seed, algorithm) frame with no missing cells must not raise."""
    df = _per_seed({
        "random_forest": [0.80, 0.81, 0.79, 0.82, 0.80, 0.83, 0.78, 0.81, 0.80, 0.82],
        "svm": [0.70, 0.71, 0.69, 0.72, 0.70, 0.73, 0.68, 0.71, 0.70, 0.72],
        "logistic_regression": [0.60, 0.61, 0.59, 0.62, 0.60, 0.63, 0.58, 0.61, 0.60, 0.62],
    })
    # This must not raise — every seed has a value for every algorithm
    out = paired_comparisons(df, metric="recall_excecao")
    assert len(out) == 3
    assert not out.isnull().any().any()
