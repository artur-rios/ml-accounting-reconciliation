import datetime
import pytest
import pandas as pd
from reconciliacao.simulation.excel_generator import generate_payment_records, write_excel
from reconciliacao.simulation.xml_generator import generate_nfse, write_xml
from reconciliacao.etl.loader import load_pagamentos, load_nfse


@pytest.fixture
def excel_file(tmp_path):
    df = generate_payment_records(n=20, seed=42)
    path = tmp_path / "pagamentos.xlsx"
    write_excel(df, path)
    return path


@pytest.fixture
def xml_file(tmp_path):
    df_pag = generate_payment_records(n=20, seed=42)
    df_nfse = generate_nfse(df_pag, conciliation_rate=0.70, seed=42)
    path = tmp_path / "nfse.xml"
    write_xml(df_nfse, path)
    return path


def test_load_pagamentos_count(excel_file):
    df = load_pagamentos(excel_file)
    assert len(df) == 20


def test_load_pagamentos_schema(excel_file):
    df = load_pagamentos(excel_file)
    assert set(df.columns) == {
        "id_pagamento", "cnpj_fornecedor", "data_pagamento",
        "valor_pago", "descricao", "centro_custo",
    }


def test_load_nfse_count(xml_file):
    df = load_nfse(xml_file)
    assert len(df) == 20


def test_load_nfse_schema(xml_file):
    df = load_nfse(xml_file)
    expected = {
        "nfse_numero", "nfse_codigo_verificacao", "nfse_data_emissao",
        "nfse_competencia", "nfse_cnpj_prestador", "nfse_razao_social_prestador",
        "nfse_cnpj_tomador", "nfse_razao_social_tomador", "nfse_valor_servicos",
        "nfse_valor_iss", "nfse_aliquota", "nfse_valor_liquido",
        "nfse_item_lista_servico", "nfse_discriminacao", "nfse_codigo_municipio",
    }
    assert expected.issubset(set(df.columns))


def test_load_nfse_cnpj_field_populated(xml_file):
    df = load_nfse(xml_file)
    assert df["nfse_cnpj_prestador"].notna().all()
    assert (df["nfse_cnpj_prestador"].str.len() > 0).all()
