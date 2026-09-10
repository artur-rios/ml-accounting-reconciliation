# tests/test_simulation/test_truth_generator.py
import pandas as pd
import pytest

from reconciliacao.simulation.retentions import legal_combos, matches_any
from reconciliacao.simulation.truth_generator import generate_comparison_dataset


def test_truth_table_has_one_row_per_payment(comparison_config):
    pag, nfse, verdade = generate_comparison_dataset(comparison_config, seed=42)
    assert len(pag) == comparison_config["n_records"]
    assert len(nfse) == comparison_config["n_records"]
    assert len(verdade) == comparison_config["n_records"]
    assert set(verdade["id_pagamento"]) == set(pag["id_pagamento"])


def test_match_rate_is_respected(comparison_config):
    _, _, verdade = generate_comparison_dataset(comparison_config, seed=42)
    matched = (verdade["nfse_numero"] != "").sum()
    expected = comparison_config["n_records"] * comparison_config["match_rate"]
    assert matched == pytest.approx(expected, abs=2)


def test_at_least_thirty_percent_of_negatives_are_hard(comparison_config):
    _, _, verdade = generate_comparison_dataset(comparison_config, seed=42)
    negatives = verdade[verdade["nfse_numero"] == ""]
    hard = (negatives["tipo_negativo"] == "hard").sum()
    assert hard / len(negatives) >= comparison_config["hard_negative_rate"] - 0.01


def test_hard_negatives_have_a_legitimate_value_ratio(comparison_config):
    """A hard negative is precisely one the value ratio cannot resolve."""
    pag, nfse, verdade = generate_comparison_dataset(comparison_config, seed=42)
    merged = pag.merge(
        nfse, left_on="cnpj_fornecedor", right_on="nfse_cnpj_prestador"
    ).merge(verdade, on="id_pagamento", suffixes=("", "_verdade"))
    hard = merged[merged["tipo_negativo"] == "hard"]
    assert len(hard) > 0
    for row in hard.itertuples(index=False):
        retencao = (1 - row.valor_pago / row.nfse_valor_servicos) * 100
        combos = legal_combos(
            row.nfse_valor_servicos,
            row.nfse_aliquota,
            comparison_config["retention_rates"],
            comparison_config["csrf_threshold_brl"],
        )
        assert matches_any(retencao, combos, tol_pp=0.5)


def test_soft_negatives_have_an_illegitimate_value_ratio(comparison_config):
    pag, nfse, verdade = generate_comparison_dataset(comparison_config, seed=42)
    merged = pag.merge(
        nfse, left_on="cnpj_fornecedor", right_on="nfse_cnpj_prestador"
    ).merge(verdade, on="id_pagamento", suffixes=("", "_verdade"))
    soft = merged[merged["tipo_negativo"] == "soft"]
    assert len(soft) > 0
    off_band = 0
    for row in soft.itertuples(index=False):
        retencao = (1 - row.valor_pago / row.nfse_valor_servicos) * 100
        combos = legal_combos(
            row.nfse_valor_servicos,
            row.nfse_aliquota,
            comparison_config["retention_rates"],
            comparison_config["csrf_threshold_brl"],
        )
        if not matches_any(retencao, combos, tol_pp=0.15):
            off_band += 1
    assert off_band == len(soft)


def test_true_pairs_sometimes_break_the_supplier_term(comparison_config):
    """No evidence dimension may separate the classes on its own."""
    pag, nfse, verdade = generate_comparison_dataset(comparison_config, seed=42)
    merged = pag.merge(nfse, left_on="cnpj_fornecedor", right_on="nfse_cnpj_prestador").merge(
        verdade, on="id_pagamento", suffixes=("", "_verdade")
    )
    matched = merged[merged["nfse_numero_verdade"] != ""]
    delta = (
        pd.to_datetime(matched["data_pagamento"])
        - pd.to_datetime(matched["nfse_data_emissao"])
    ).dt.days
    assert not delta.isin(comparison_config["payment_terms_days"]).all()


def test_generation_is_deterministic(comparison_config):
    a = generate_comparison_dataset(comparison_config, seed=7)[0]
    b = generate_comparison_dataset(comparison_config, seed=7)[0]
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_produce_different_data(comparison_config):
    a = generate_comparison_dataset(comparison_config, seed=1)[0]
    b = generate_comparison_dataset(comparison_config, seed=2)[0]
    assert not a["valor_pago"].equals(b["valor_pago"])
