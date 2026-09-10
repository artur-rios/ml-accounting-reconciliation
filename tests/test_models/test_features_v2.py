import pandas as pd
import pytest

from reconciliacao.models.features_v2 import build_features_v2, fit_supplier_terms

CFG = {
    "retention_rates": {"irrf": 1.50, "csrf": 4.65, "inss": 11.00},
    "retention_tolerance_pp": 0.05,
    "csrf_threshold_brl": 5000.0,
}


def _frame():
    return pd.DataFrame({
        "id_pagamento": ["PAG-000001", "PAG-000002", "PAG-000003"],
        "cnpj_fornecedor": ["11222333000181", "11222333000181", "44555666000195"],
        "data_pagamento": pd.to_datetime(["2026-02-14", "2026-03-02", "2026-01-20"]),
        "valor_pago": [9850.00, 5631.00, 970.00],  # 1.50%, 6.15% (irrf+csrf), 3.00% (iss)
        "descricao": ["consultoria de TI", "consultoria de TI", "algo diferente"],
        "centro_custo": ["CC-101", "CC-102", "CC-103"],
        "municipio_centro_custo": ["3550308", "3304557", "3550308"],
        "nfse_numero": ["000001", "000002", "000003"],
        "nfse_data_emissao": ["2026-01-15T00:00:00", "2026-01-31T00:00:00", "2026-01-20T00:00:00"],
        "nfse_valor_servicos": [10000.00, 6000.00, 1000.00],
        "nfse_valor_iss": [500.00, 300.00, 30.00],
        "nfse_aliquota": [5.00, 5.00, 3.00],
        "nfse_discriminacao": ["consultoria de TI", "consultoria de TI", "servico X"],
        "nfse_codigo_municipio": ["3550308", "3550308", "3550308"],
        "label": [1, 1, 0],
    })


def test_retention_and_ratio_are_computed():
    df = _frame()
    terms, global_term = fit_supplier_terms(df)
    X, y = build_features_v2(df, CFG, terms, global_term)
    assert X.loc[0, "retencao_implicita_pct"] == pytest.approx(1.50)
    assert X.loc[0, "razao_valor"] == pytest.approx(0.985)
    assert X.loc[0, "valor_retido_brl"] == pytest.approx(150.00)


def test_compatibility_flag_fires_for_legal_withholding():
    df = _frame()
    terms, global_term = fit_supplier_terms(df)
    X, _ = build_features_v2(df, CFG, terms, global_term)
    assert X.loc[0, "compat_irrf"] == 1          # 1.50%
    assert X.loc[1, "compat_irrf_csrf"] == 1     # 6.15% on a 6000 invoice


def test_csrf_threshold_flag_tracks_invoice_size():
    df = _frame()
    terms, global_term = fit_supplier_terms(df)
    X, _ = build_features_v2(df, CFG, terms, global_term)
    assert X.loc[0, "acima_limite_csrf"] == 1    # 10000 > 5000
    assert X.loc[2, "acima_limite_csrf"] == 0    # 1000 < 5000


def test_municipality_flag_compares_cost_centre_to_invoice():
    df = _frame()
    terms, global_term = fit_supplier_terms(df)
    X, _ = build_features_v2(df, CFG, terms, global_term)
    assert X.loc[0, "mesmo_municipio"] == 1
    assert X.loc[1, "mesmo_municipio"] == 0


def test_supplier_terms_are_fitted_on_training_data_only():
    """A supplier unseen in training falls back to the global training median."""
    df = _frame()
    train = df[df["cnpj_fornecedor"] == "11222333000181"]
    terms, global_term = fit_supplier_terms(train)
    assert "44555666000195" not in terms

    X, _ = build_features_v2(df, CFG, terms, global_term)
    # PAG-000003: delta 0 days, supplier unseen -> deviation from the global median
    assert X.loc[2, "desvio_prazo_fornecedor"] == pytest.approx(0.0 - global_term)


def test_no_feature_is_the_label():
    df = _frame()
    terms, global_term = fit_supplier_terms(df)
    X, y = build_features_v2(df, CFG, terms, global_term)
    assert "label" not in X.columns
    assert "tipo_negativo" not in X.columns
    assert list(y) == [1, 1, 0]


def test_text_similarity_is_higher_for_matching_descriptions():
    df = _frame()
    terms, global_term = fit_supplier_terms(df)
    X, _ = build_features_v2(df, CFG, terms, global_term)
    assert X.loc[0, "similaridade_descricao"] > X.loc[2, "similaridade_descricao"]
