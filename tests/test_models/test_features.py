import datetime
import pandas as pd
import pytest
from reconciliacao.models.features import build_features


@pytest.fixture
def sample_merged():
    return pd.DataFrame({
        "id_pagamento": ["PAG-1", "PAG-2", "PAG-3"],
        "cnpj_fornecedor": ["11222333000181", "44555666000195", None],
        "data_pagamento": [datetime.date(2024, 1, 15)] * 3,
        "valor_pago": [1000.0, 2500.0, 800.0],
        "descricao": ["consultoria TI", "manutencao predial", "limpeza"],
        "centro_custo": ["CC-1", "CC-2", "CC-3"],
        "nfse_cnpj_prestador": ["11222333000181", "44555666000195", None],
        "nfse_data_emissao": [datetime.date(2024, 1, 15), datetime.date(2024, 2, 15), None],
        "nfse_valor_servicos": [1000.0, 2600.0, None],
        "nfse_valor_iss": [50.0, 130.0, None],
        "nfse_aliquota": [5.0, 5.0, None],
        "nfse_discriminacao": ["Servicos de TI", "Manutencao Predial", None],
        "nfse_codigo_municipio": ["3550308", "3550308", None],
        "delta_days": [0.0, 31.0, 9999.0],
        "delta_valor_pct": [0.0, 4.0, 100.0],
        "label": [2, 0, 0],
    })


def test_given_merged_records_when_features_are_built_then_a_frame_and_a_series_come_back(sample_merged):
    X, y = build_features(sample_merged)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_given_merged_records_when_features_are_built_then_there_is_one_row_per_record(sample_merged):
    X, y = build_features(sample_merged)
    assert len(X) == len(sample_merged)
    assert len(y) == len(sample_merged)


def test_given_missing_values_when_features_are_built_then_none_is_nan(sample_merged):
    X, y = build_features(sample_merged)
    assert not X.isna().any().any()
    assert not y.isna().any()


def test_given_merged_records_when_features_are_built_then_the_expected_columns_exist(sample_merged):
    X, _ = build_features(sample_merged)
    expected = {"delta_days", "delta_valor_pct", "valor_pago", "nfse_valor_iss",
                "nfse_aliquota", "cnpj_match", "descricao_similarity"}
    assert expected.issubset(set(X.columns))


def test_given_a_missing_cnpj_when_features_are_built_then_the_match_flag_is_zero(sample_merged):
    X, _ = build_features(sample_merged)
    assert X["cnpj_match"].tolist() == [1, 1, 0]


def test_given_merged_records_when_features_are_built_then_the_labels_match_the_input(sample_merged):
    _, y = build_features(sample_merged)
    assert y.tolist() == [2, 0, 0]


def test_given_descriptions_with_no_words_when_features_are_built_then_similarity_is_zero(sample_merged):
    """TfidfVectorizer refuses an empty vocabulary; the feature falls back to 0."""
    df = sample_merged.assign(descricao=["", "", ""], nfse_discriminacao=[None, "", None])
    X, _y = build_features(df)
    assert (X["descricao_similarity"] == 0.0).all()
