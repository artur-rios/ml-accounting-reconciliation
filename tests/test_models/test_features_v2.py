import datetime

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


def test_given_a_withholding_when_features_are_built_then_retention_and_ratio_are_computed():
    df = _frame()
    terms, global_term = fit_supplier_terms(df)
    X, y = build_features_v2(df, CFG, terms, global_term)
    assert X.loc[0, "retencao_implicita_pct"] == pytest.approx(1.50)
    assert X.loc[0, "razao_valor"] == pytest.approx(0.985)
    assert X.loc[0, "valor_retido_brl"] == pytest.approx(150.00)


def test_given_a_legal_withholding_when_features_are_built_then_the_compatibility_flag_fires():
    df = _frame()
    terms, global_term = fit_supplier_terms(df)
    X, _ = build_features_v2(df, CFG, terms, global_term)
    assert X.loc[0, "compat_irrf"] == 1          # 1.50%
    assert X.loc[1, "compat_irrf_csrf"] == 1     # 6.15% on a 6000 invoice


def test_given_invoice_sizes_when_features_are_built_then_the_csrf_flag_tracks_the_threshold():
    df = _frame()
    terms, global_term = fit_supplier_terms(df)
    X, _ = build_features_v2(df, CFG, terms, global_term)
    assert X.loc[0, "acima_limite_csrf"] == 1    # 10000 > 5000
    assert X.loc[2, "acima_limite_csrf"] == 0    # 1000 < 5000


def test_given_cost_centre_and_invoice_municipalities_when_features_are_built_then_the_flag_compares_them():
    df = _frame()
    terms, global_term = fit_supplier_terms(df)
    X, _ = build_features_v2(df, CFG, terms, global_term)
    assert X.loc[0, "mesmo_municipio"] == 1
    assert X.loc[1, "mesmo_municipio"] == 0


def test_given_training_rows_when_supplier_terms_are_fitted_then_only_they_count():
    """A supplier unseen in training falls back to the global training median."""
    df = _frame()
    train = df[df["cnpj_fornecedor"] == "11222333000181"]
    terms, global_term = fit_supplier_terms(train)
    assert "44555666000195" not in terms

    X, _ = build_features_v2(df, CFG, terms, global_term)
    # PAG-000003: delta 0 days, supplier unseen -> deviation from the global median
    assert X.loc[2, "desvio_prazo_fornecedor"] == pytest.approx(0.0 - global_term)


def test_given_built_features_when_inspected_then_no_column_is_the_label():
    df = _frame()
    terms, global_term = fit_supplier_terms(df)
    X, y = build_features_v2(df, CFG, terms, global_term)
    assert "label" not in X.columns
    assert "tipo_negativo" not in X.columns
    assert list(y) == [1, 1, 0]


def test_given_matching_descriptions_when_features_are_built_then_similarity_is_higher():
    df = _frame()
    terms, global_term = fit_supplier_terms(df)
    X, _ = build_features_v2(df, CFG, terms, global_term)
    assert X.loc[0, "similaridade_descricao"] > X.loc[2, "similaridade_descricao"]


def test_given_a_nonpositive_service_value_when_features_are_built_then_it_raises_naming_the_rows():
    """A zero/missing nfse_valor_servicos means an upstream invariant broke;
    build_features_v2 must fail loudly rather than divide into inf/NaN."""
    df = _frame()
    df.loc[1, "nfse_valor_servicos"] = 0.0
    df.loc[2, "nfse_valor_servicos"] = None
    terms, global_term = fit_supplier_terms(_frame())

    with pytest.raises(ValueError) as exc_info:
        build_features_v2(df, CFG, terms, global_term)

    message = str(exc_info.value)
    assert "PAG-000002" in message
    assert "PAG-000003" in message


def test_given_dates_as_objects_or_iso_strings_when_features_are_built_then_delta_dias_agrees():
    """data_pagamento arrives as datetime.date while nfse_data_emissao
    arrives as an ISO-8601 string; both must resolve to the same delta."""
    df = _frame()
    df["data_pagamento"] = [
        datetime.date(2026, 2, 14),
        datetime.date(2026, 3, 2),
        datetime.date(2026, 1, 20),
    ]
    df["nfse_data_emissao"] = [
        "2026-01-15T00:00:00",
        "2026-01-31T00:00:00",
        "2026-01-20T00:00:00",
    ]
    terms, global_term = fit_supplier_terms(df)
    X, _ = build_features_v2(df, CFG, terms, global_term)

    assert X.loc[0, "delta_dias"] == pytest.approx(30.0)
    assert X.loc[1, "delta_dias"] == pytest.approx(30.0)
    assert X.loc[2, "delta_dias"] == pytest.approx(0.0)


def test_given_an_empty_training_frame_when_supplier_terms_are_fitted_then_it_raises():
    """An empty training partition is a programming error in the split,
    not a legitimate scenario: delta.median() on it would be NaN and
    would silently poison desvio_prazo_fornecedor for every row."""
    empty = _frame().iloc[0:0]
    with pytest.raises(ValueError):
        fit_supplier_terms(empty)


def test_given_a_single_training_row_when_supplier_terms_are_fitted_then_it_works():
    """A single-row training frame is legitimate and must work."""
    single = _frame().iloc[[0]]
    terms, global_term = fit_supplier_terms(single)
    assert terms == {"11222333000181": 30.0}
    assert global_term == pytest.approx(30.0)


def test_given_descriptions_with_no_words_when_features_are_built_then_similarity_is_zero():
    """TfidfVectorizer refuses an empty vocabulary; the feature falls back to 0."""
    df = _frame().assign(descricao=["", "", ""], nfse_discriminacao=["", None, ""])
    terms, global_term = fit_supplier_terms(df)
    X, _y = build_features_v2(df, CFG, terms, global_term)
    assert (X["similaridade_descricao"] == 0.0).all()
