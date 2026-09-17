"""Paired comparison of algorithms across seeds.

The protocol follows Demsar (2006), who is cited for it: a Friedman omnibus
test first, establishing that the algorithms are not all equivalent, and only
then the pairwise post-hoc tests. Reporting pairwise Wilcoxon alone -- which
this module did originally -- inverts the order Demsar recommends and invites
the objection that three dependent pairwise tests were run without first
rejecting the global null.

The post-hoc step is Wilcoxon signed-rank on the paired per-seed metric, with
Holm correction for the three pairwise comparisons. The median paired
difference is reported alongside the p-value: with ten seeds the test has
little power for small differences, so significance without magnitude would
not support a recommendation.
"""

from itertools import combinations

import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

ALPHA = 0.05


def _holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for position, index in enumerate(order):
        value = (m - position) * p_values[index]
        running_max = max(running_max, min(value, 1.0))
        adjusted[index] = running_max
    return adjusted


def _complete_wide(per_seed: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Seeds x algorithms matrix for one metric, rejecting any missing cell.

    Both the omnibus and the post-hoc tests are paired: a missing
    (seed, algorithm) cell would silently unbalance them, so it is an error
    rather than something to drop.
    """
    wide = per_seed.pivot(index="seed", columns="algorithm", values=metric)

    # Validate that no (seed, algorithm) combinations are missing
    missing_mask = wide.isna()
    if missing_mask.any().any():
        # Collect missing cells: (algorithm, seed) pairs
        missing_cells = []
        for algo in wide.columns:
            for seed in wide.index:
                if pd.isna(wide.loc[seed, algo]):
                    missing_cells.append((algo, seed))

        # Cap enumeration at a handful, provide total count if longer
        max_show = 5
        shown = missing_cells[:max_show]
        msg_lines = ["Missing (seed, algorithm) combinations after pivot:"]
        for algo, seed in shown:
            msg_lines.append(f"  algorithm={algo}, seed={seed}")

        if len(missing_cells) > max_show:
            msg_lines.append(f"  ... and {len(missing_cells) - max_show} more (total: {len(missing_cells)})")

        raise ValueError("\n".join(msg_lines))

    return wide


def friedman_omnibus(per_seed: pd.DataFrame, metric: str) -> dict[str, float]:
    """Friedman test over the seeds x algorithms matrix.

    The omnibus step Demsar (2006) puts before any post-hoc comparison: it
    asks whether the algorithms rank differently across seeds at all, without
    committing to which pair differs. Rejecting here is what licenses reading
    the pairwise table that follows.

    Returns the statistic, the p-value, and the mean rank of each algorithm
    (rank 1 = best on that seed), since the ranks are what the test actually
    operates on and they make the direction of the result readable.
    """
    wide = _complete_wide(per_seed, metric)
    algorithms = sorted(wide.columns)

    statistic, p_value = friedmanchisquare(*[wide[a] for a in algorithms])

    # Rank 1 = highest metric value on that seed; every metric fed here is
    # one where larger is better.
    ranks = wide[algorithms].rank(axis=1, ascending=False).mean()

    resultado = {
        "metric": metric,
        "n_seeds": int(len(wide)),
        "n_algorithms": len(algorithms),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < ALPHA),
    }
    for algorithm in algorithms:
        resultado[f"rank_medio_{algorithm}"] = float(ranks[algorithm])
    return resultado


def paired_comparisons(per_seed: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pairwise Wilcoxon over seeds, Holm-corrected, with effect size."""
    wide = _complete_wide(per_seed, metric)
    algorithms = sorted(wide.columns)

    rows, raw_p_values = [], []
    for a, b in combinations(algorithms, 2):
        diff = wide[a] - wide[b]
        if diff.abs().sum() == 0:
            statistic, p_value = 0.0, 1.0
        else:
            result = wilcoxon(wide[a], wide[b], zero_method="wilcox")
            statistic, p_value = float(result.statistic), float(result.pvalue)
        raw_p_values.append(p_value)
        rows.append({
            "algorithm_a": a,
            "algorithm_b": b,
            "median_diff": float(diff.median()),
            "statistic": statistic,
            "p_value": p_value,
        })

    for row, p_holm in zip(rows, _holm(raw_p_values)):
        row["p_holm"] = p_holm
        row["significant"] = p_holm < ALPHA

    return pd.DataFrame(rows)
