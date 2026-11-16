# tests/test_simulation/test_retentions.py
import pytest
from reconciliacao.simulation.retentions import (
    legal_combos,
    matches_any,
    compatibility_flags,
)

RATES = {"irrf": 1.50, "csrf": 4.65, "inss": 11.00}


def test_given_an_invoice_below_the_csrf_threshold_when_combos_are_listed_then_csrf_is_excluded():
    combos = legal_combos(4000.0, 3.00, RATES, csrf_threshold=5000.0)
    assert "csrf" not in combos
    assert "irrf_csrf" not in combos
    assert "irrf_csrf_iss" not in combos


def test_given_an_invoice_above_the_csrf_threshold_when_combos_are_listed_then_csrf_is_included():
    combos = legal_combos(6000.0, 3.00, RATES, csrf_threshold=5000.0)
    assert combos["csrf"] == pytest.approx(4.65)
    assert combos["irrf_csrf"] == pytest.approx(6.15)
    assert combos["irrf_csrf_iss"] == pytest.approx(9.15)


def test_given_any_invoice_when_combos_are_listed_then_no_withholding_and_iss_are_offered():
    combos = legal_combos(1000.0, 2.50, RATES, csrf_threshold=5000.0)
    assert combos["nenhuma"] == 0.0
    assert combos["iss"] == pytest.approx(2.50)
    assert combos["irrf_iss"] == pytest.approx(4.00)
    assert combos["inss"] == pytest.approx(11.00)


def test_given_a_tolerance_when_matched_then_only_values_within_it_match():
    combos = {"irrf": 1.50}
    assert matches_any(1.53, combos, tol_pp=0.05) is True
    assert matches_any(1.70, combos, tol_pp=0.05) is False


def test_given_a_small_invoice_when_flags_are_computed_then_csrf_is_not_gated_by_the_threshold():
    # 4.65% on a small invoice is not legal, but the flag still fires.
    # The model learns the interaction from acima_limite_csrf; we do not bake it in.
    flags = compatibility_flags(4.65, aliquota_iss=3.00, rates=RATES, tol_pp=0.05)
    assert flags["compat_csrf"] == 1


def test_given_a_combined_iss_withholding_when_flags_are_computed_then_it_is_detected():
    flags = compatibility_flags(4.50, aliquota_iss=3.00, rates=RATES, tol_pp=0.05)
    assert flags["compat_combo_iss"] == 1  # irrf 1.50 + iss 3.00
    assert flags["compat_irrf"] == 0


def test_given_an_illegitimate_value_when_flags_are_computed_then_all_are_zero():
    flags = compatibility_flags(7.30, aliquota_iss=3.00, rates=RATES, tol_pp=0.05)
    assert sum(flags.values()) == 0
