import datetime
import numpy as np
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


def test_build_features_returns_dataframe_and_series(sample_merged):
    X, y = build_features(sample_merged)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_build_features_row_count(sample_merged):
    X, y = build_features(sample_merged)
    assert len(X) == len(sample_merged)
    assert len(y) == len(sample_merged)


def test_build_features_no_nan(sample_merged):
    X, y = build_features(sample_merged)
    assert not X.isna().any().any()
    assert not y.isna().any()


def test_build_features_expected_columns(sample_merged):
    X, _ = build_features(sample_merged)
    expected = {"delta_days", "delta_valor_pct", "valor_pago", "nfse_valor_iss",
                "nfse_aliquota", "cnpj_match", "descricao_similarity"}
    assert expected.issubset(set(X.columns))


def test_build_features_cnpj_match_flag(sample_merged):
    X, _ = build_features(sample_merged)
    assert X["cnpj_match"].tolist() == [1, 1, 0]


def test_build_features_labels_match_input(sample_merged):
    _, y = build_features(sample_merged)
    assert y.tolist() == [2, 0, 0]
