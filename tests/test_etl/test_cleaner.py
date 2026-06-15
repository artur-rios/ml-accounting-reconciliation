import datetime
import pandas as pd
import pytest
from reconciliacao.etl.cleaner import clean_pagamentos, clean_nfse


def test_clean_pagamentos_normalizes_cnpj():
    df = pd.DataFrame({
        "id_pagamento": ["PAG-1"],
        "cnpj_fornecedor": ["11.222.333/0001-81"],
        "data_pagamento": ["2024-01-15"],
        "valor_pago": ["1000.00"],
        "descricao": ["servico"],
        "centro_custo": ["CC-001"],
    })
    result = clean_pagamentos(df)
    assert result["cnpj_fornecedor"].iloc[0] == "11222333000181"


def test_clean_pagamentos_validates_cnpj():
    df = pd.DataFrame({
        "id_pagamento": ["PAG-1", "PAG-2"],
        "cnpj_fornecedor": ["11222333000181", "11222333000182"],
        "data_pagamento": ["2024-01-15", "2024-01-15"],
        "valor_pago": ["1000.00", "2000.00"],
        "descricao": ["a", "b"],
        "centro_custo": ["CC-1", "CC-2"],
    })
    result = clean_pagamentos(df)
    assert result["cnpj_valid"].tolist() == [True, False]


def test_clean_pagamentos_parses_dates():
    df = pd.DataFrame({
        "id_pagamento": ["PAG-1"],
        "cnpj_fornecedor": ["11222333000181"],
        "data_pagamento": ["2024-06-15"],
        "valor_pago": ["999.50"],
        "descricao": ["x"],
        "centro_custo": ["CC-1"],
    })
    result = clean_pagamentos(df)
    assert result["data_pagamento"].iloc[0] == datetime.date(2024, 6, 15)


def test_clean_pagamentos_parses_float_values():
    df = pd.DataFrame({
        "id_pagamento": ["PAG-1"],
        "cnpj_fornecedor": ["11222333000181"],
        "data_pagamento": ["2024-01-15"],
        "valor_pago": ["1.234,56"],
        "descricao": ["x"],
        "centro_custo": ["CC-1"],
    })
    result = clean_pagamentos(df)
    assert result["valor_pago"].iloc[0] == pytest.approx(1234.56)


def test_clean_pagamentos_flags_duplicates():
    df = pd.DataFrame({
        "id_pagamento": ["PAG-1", "PAG-1"],
        "cnpj_fornecedor": ["11222333000181", "11222333000181"],
        "data_pagamento": ["2024-01-15", "2024-01-15"],
        "valor_pago": ["1000", "1000"],
        "descricao": ["x", "x"],
        "centro_custo": ["CC-1", "CC-1"],
    })
    result = clean_pagamentos(df)
    assert result["is_duplicate"].all()


def test_clean_nfse_parses_numeric_fields():
    df = pd.DataFrame({
        "nfse_numero": ["1"],
        "nfse_codigo_verificacao": ["ABC"],
        "nfse_data_emissao": ["2024-01-15T00:00:00"],
        "nfse_competencia": ["2024-01-01T00:00:00"],
        "nfse_cnpj_prestador": ["11222333000181"],
        "nfse_razao_social_prestador": ["Empresa A"],
        "nfse_cnpj_tomador": ["44555666000195"],
        "nfse_razao_social_tomador": ["Empresa B"],
        "nfse_valor_servicos": ["2500.00"],
        "nfse_valor_iss": ["125.00"],
        "nfse_aliquota": ["5.00"],
        "nfse_valor_liquido": ["2375.00"],
        "nfse_item_lista_servico": ["1.01"],
        "nfse_discriminacao": ["Servicos"],
        "nfse_codigo_municipio": ["3550308"],
    })
    result = clean_nfse(df)
    assert result["nfse_valor_servicos"].iloc[0] == pytest.approx(2500.00)
    assert result["nfse_data_emissao"].iloc[0] == datetime.date(2024, 1, 15)
