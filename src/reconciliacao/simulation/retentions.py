"""Brazilian withholding-tax arithmetic for the comparison experiment.

Rates and the CSRF threshold come from config.yaml. This module only computes
which total withholding percentages are legally plausible for a given invoice,
and whether an observed percentage matches one of them.
"""


def legal_combos(
    valor_servicos: float,
    aliquota_iss: float,
    rates: dict,
    csrf_threshold: float,
) -> dict[str, float]:
    """Total withholding percentages that are legally plausible for this invoice.

    CSRF (PIS + COFINS + CSLL) only applies above csrf_threshold, so combos
    including it are omitted for smaller invoices.
    """
    combos = {
        "nenhuma": 0.0,
        "irrf": rates["irrf"],
        "inss": rates["inss"],
        "iss": aliquota_iss,
        "irrf_iss": rates["irrf"] + aliquota_iss,
    }
    if valor_servicos > csrf_threshold:
        combos["csrf"] = rates["csrf"]
        combos["irrf_csrf"] = rates["irrf"] + rates["csrf"]
        combos["irrf_csrf_iss"] = rates["irrf"] + rates["csrf"] + aliquota_iss
    return combos


def matches_any(retencao_pct: float, combos: dict[str, float], tol_pp: float) -> bool:
    """True when retencao_pct is within tol_pp percentage points of any combo."""
    return any(abs(retencao_pct - valor) <= tol_pp for valor in combos.values())


def compatibility_flags(
    retencao_pct: float,
    aliquota_iss: float,
    rates: dict,
    tol_pp: float,
) -> dict[str, int]:
    """Per-combination compatibility indicators, as 0/1 features.

    Deliberately NOT gated by the CSRF threshold: these are numeric comparisons
    only. The interaction between invoice size and CSRF legitimacy is left for
    the model to learn from the acima_limite_csrf feature -- that interaction is
    the intended source of discriminative power between algorithms.
    """
    def near(valor: float) -> int:
        return int(abs(retencao_pct - valor) <= tol_pp)

    return {
        "compat_irrf": near(rates["irrf"]),
        "compat_csrf": near(rates["csrf"]),
        "compat_irrf_csrf": near(rates["irrf"] + rates["csrf"]),
        "compat_iss": near(aliquota_iss),
        "compat_combo_iss": max(
            near(rates["irrf"] + aliquota_iss),
            near(rates["irrf"] + rates["csrf"] + aliquota_iss),
        ),
        "compat_inss": near(rates["inss"]),
    }
