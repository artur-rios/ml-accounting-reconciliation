import datetime
from pathlib import Path
import pandas as pd
import pytest
from lxml import etree
from reconciliacao.simulation.xml_generator import generate_nfse, write_xml

NS = "http://www.abrasf.org.br/nfse.xsd"


@pytest.fixture
def sample_pagamentos():
    return pd.DataFrame({
        "id_pagamento": [f"PAG-{i:06d}" for i in range(1, 11)],
        "cnpj_fornecedor": ["11222333000181"] * 5 + ["44555666000195"] * 5,
        "data_pagamento": [datetime.date(2024, 1, 15)] * 10,
        "valor_pago": [1000.00] * 10,
        "descricao": ["consultoria"] * 10,
        "centro_custo": ["CC-001"] * 10,
    })


def test_generate_nfse_count(sample_pagamentos):
    df = generate_nfse(sample_pagamentos, conciliation_rate=0.70, seed=42)
    assert len(df) == len(sample_pagamentos)


def test_generate_nfse_schema(sample_pagamentos):
    df = generate_nfse(sample_pagamentos, conciliation_rate=0.70, seed=42)
    expected_cols = {
        "nfse_numero", "nfse_codigo_verificacao", "nfse_data_emissao",
        "nfse_competencia", "nfse_cnpj_prestador", "nfse_razao_social_prestador",
        "nfse_cnpj_tomador", "nfse_razao_social_tomador", "nfse_valor_servicos",
        "nfse_valor_iss", "nfse_aliquota", "nfse_valor_liquido",
        "nfse_item_lista_servico", "nfse_discriminacao", "nfse_codigo_municipio",
    }
    assert expected_cols.issubset(set(df.columns))


def test_generate_nfse_reproducible(sample_pagamentos):
    df1 = generate_nfse(sample_pagamentos, conciliation_rate=0.70, seed=42)
    df2 = generate_nfse(sample_pagamentos, conciliation_rate=0.70, seed=42)
    assert df1.equals(df2)


def test_write_xml_creates_valid_abrasf_file(sample_pagamentos, tmp_path):
    df = generate_nfse(sample_pagamentos, conciliation_rate=0.70, seed=42)
    out = tmp_path / "nfse.xml"
    write_xml(df, out)
    assert out.exists()
    tree = etree.parse(str(out))
    ns = {"ns": NS}
    comps = tree.xpath("//ns:CompNfse", namespaces=ns)
    assert len(comps) == len(sample_pagamentos)


def test_write_xml_contains_cnpj(sample_pagamentos, tmp_path):
    df = generate_nfse(sample_pagamentos, conciliation_rate=1.0, seed=42)
    out = tmp_path / "nfse.xml"
    write_xml(df, out)
    tree = etree.parse(str(out))
    ns = {"ns": NS}
    cnpjs = tree.xpath("//ns:PrestadorServico//ns:Cnpj/text()", namespaces=ns)
    assert len(cnpjs) == len(sample_pagamentos)


def test_generate_nfse_injects_mismatches(sample_pagamentos):
    # With conciliation_rate=0.0, ALL records should be mismatched
    # so no NFS-e CNPJ should match its source payment CNPJ
    df = generate_nfse(sample_pagamentos, conciliation_rate=0.0, seed=42)
    # date_shift and value_discrepancy mismatches keep the same CNPJ but change
    # date or value, so we only check that NOT ALL CNPJs match their source
    payment_cnpjs = sample_pagamentos["cnpj_fornecedor"].tolist()
    nfse_cnpjs = df["nfse_cnpj_prestador"].tolist()
    # At least some records should have different CNPJ (cnpj_error or ghost mismatches)
    # With seed=42 and 10 records, statistical probability of zero CNPJ mismatches is negligible
    matches = sum(p == n for p, n in zip(payment_cnpjs, nfse_cnpjs))
    assert matches < len(sample_pagamentos), "Expected some CNPJ mismatches with conciliation_rate=0.0"
