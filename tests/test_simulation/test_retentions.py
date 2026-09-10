# tests/test_simulation/test_retentions.py
import pytest
from reconciliacao.simulation.retentions import (
    legal_combos,
    matches_any,
    compatibility_flags,
)

RATES = {"irrf": 1.50, "csrf": 4.65, "inss": 11.00}


def test_legal_combos_excludes_csrf_below_threshold():
    combos = legal_combos(4000.0, 3.00, RATES, csrf_threshold=5000.0)
    assert "csrf" not in combos
    assert "irrf_csrf" not in combos
    assert "irrf_csrf_iss" not in combos


def test_legal_combos_includes_csrf_above_threshold():
    combos = legal_combos(6000.0, 3.00, RATES, csrf_threshold=5000.0)
    assert combos["csrf"] == pytest.approx(4.65)
    assert combos["irrf_csrf"] == pytest.approx(6.15)
    assert combos["irrf_csrf_iss"] == pytest.approx(9.15)


def test_legal_combos_always_offers_no_withholding_and_iss():
    combos = legal_combos(1000.0, 2.50, RATES, csrf_threshold=5000.0)
    assert combos["nenhuma"] == 0.0
    assert combos["iss"] == pytest.approx(2.50)
    assert combos["irrf_iss"] == pytest.approx(4.00)
    assert combos["inss"] == pytest.approx(11.00)


def test_matches_any_respects_tolerance():
    combos = {"irrf": 1.50}
    assert matches_any(1.53, combos, tol_pp=0.05) is True
    assert matches_any(1.70, combos, tol_pp=0.05) is False


def test_compatibility_flags_are_not_gated_by_csrf_threshold():
    # 4.65% on a small invoice is not legal, but the flag still fires.
    # The model learns the interaction from acima_limite_csrf; we do not bake it in.
    flags = compatibility_flags(4.65, aliquota_iss=3.00, rates=RATES, tol_pp=0.05)
    assert flags["compat_csrf"] == 1


def test_compatibility_flags_detect_combined_iss():
    flags = compatibility_flags(4.50, aliquota_iss=3.00, rates=RATES, tol_pp=0.05)
    assert flags["compat_combo_iss"] == 1  # irrf 1.50 + iss 3.00
    assert flags["compat_irrf"] == 0


def test_compatibility_flags_all_zero_for_illegitimate_value():
    flags = compatibility_flags(7.30, aliquota_iss=3.00, rates=RATES, tol_pp=0.05)
    assert sum(flags.values()) == 0
