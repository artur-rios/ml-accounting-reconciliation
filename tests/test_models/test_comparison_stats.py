import pandas as pd
import pytest

from reconciliacao.models.comparison_stats import friedman_omnibus, paired_comparisons


def _per_seed(values: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for algorithm, series in values.items():
        for seed, value in enumerate(series):
            rows.append({"seed": seed, "algorithm": algorithm, "recall_excecao": value})
    return pd.DataFrame(rows)


def test_given_three_algorithms_when_compared_pairwise_then_there_is_one_row_per_pair():
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


def test_given_a_consistent_difference_when_compared_then_it_is_significant():
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


def test_given_identical_performance_when_compared_then_nothing_is_significant():
    df = _per_seed({
        "random_forest": [0.80, 0.70, 0.75, 0.82, 0.68, 0.79, 0.71, 0.77, 0.73, 0.76],
        "svm": [0.80, 0.70, 0.75, 0.82, 0.68, 0.79, 0.71, 0.77, 0.73, 0.76],
        "logistic_regression": [0.60] * 10,
    })
    out = paired_comparisons(df, metric="recall_excecao")
    row = out[(out["algorithm_a"] == "random_forest") & (out["algorithm_b"] == "svm")].iloc[0]
    assert row["median_diff"] == pytest.approx(0.0)
    assert bool(row["significant"]) is False


def test_given_several_comparisons_when_holm_corrected_then_no_p_value_shrinks():
    df = _per_seed({
        "random_forest": [0.80, 0.81, 0.79, 0.82, 0.80, 0.83, 0.78, 0.81, 0.80, 0.82],
        "svm": [0.70, 0.71, 0.69, 0.72, 0.70, 0.73, 0.68, 0.71, 0.70, 0.72],
        "logistic_regression": [0.60, 0.61, 0.59, 0.62, 0.60, 0.63, 0.58, 0.61, 0.60, 0.62],
    })
    out = paired_comparisons(df, metric="recall_excecao")
    assert (out["p_holm"] >= out["p_value"] - 1e-12).all()
    assert (out["p_holm"] <= 1.0).all()


def test_given_a_missing_seed_algorithm_cell_when_compared_then_it_raises_naming_it():
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


def test_given_a_complete_frame_when_compared_then_it_does_not_raise():
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


def test_given_one_algorithm_always_winning_when_friedman_runs_then_it_rejects():
    """The omnibus step: a consistent ordering across seeds must reject the
    global null before any pairwise table is read."""
    rows = []
    for seed in range(10):
        rows.append({"seed": seed, "algorithm": "a", "m": 0.90 + seed * 0.001})
        rows.append({"seed": seed, "algorithm": "b", "m": 0.85 + seed * 0.001})
        rows.append({"seed": seed, "algorithm": "c", "m": 0.80 + seed * 0.001})
    resultado = friedman_omnibus(pd.DataFrame(rows), metric="m")

    assert resultado["significant"] is True
    assert resultado["p_value"] < 0.05
    assert resultado["n_seeds"] == 10
    assert resultado["n_algorithms"] == 3
    # Rank 1 = best; a wins every seed, c loses every seed.
    assert resultado["rank_medio_a"] == pytest.approx(1.0)
    assert resultado["rank_medio_c"] == pytest.approx(3.0)


def test_given_no_consistent_ordering_when_friedman_runs_then_it_does_not_reject():
    """No stable ordering means no licence to read the pairwise table."""
    valores = {
        "a": [0.80, 0.90, 0.85, 0.83, 0.88, 0.81, 0.89, 0.84, 0.86, 0.82],
        "b": [0.90, 0.80, 0.83, 0.88, 0.81, 0.89, 0.84, 0.86, 0.82, 0.85],
        "c": [0.85, 0.83, 0.90, 0.81, 0.89, 0.84, 0.86, 0.82, 0.88, 0.80],
    }
    rows = [
        {"seed": seed, "algorithm": nome, "m": serie[seed]}
        for nome, serie in valores.items()
        for seed in range(10)
    ]
    resultado = friedman_omnibus(pd.DataFrame(rows), metric="m")
    assert resultado["significant"] is False


def test_given_incomplete_pairings_when_friedman_runs_then_it_raises():
    """Same completeness contract as the pairwise test: a missing cell would
    silently unbalance a paired test rather than fail it."""
    rows = [
        {"seed": 0, "algorithm": "a", "m": 0.9},
        {"seed": 0, "algorithm": "b", "m": 0.8},
        {"seed": 1, "algorithm": "a", "m": 0.9},
    ]
    with pytest.raises(ValueError, match="Missing"):
        friedman_omnibus(pd.DataFrame(rows), metric="m")
