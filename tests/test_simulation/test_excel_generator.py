import pandas as pd
from reconciliacao.simulation.excel_generator import generate_payment_records, write_excel


def test_generate_payment_records_count():
    df = generate_payment_records(n=100, seed=42)
    assert len(df) == 100


def test_generate_payment_records_schema():
    df = generate_payment_records(n=10, seed=42)
    assert set(df.columns) == {
        "id_pagamento", "cnpj_fornecedor", "data_pagamento",
        "valor_pago", "descricao", "centro_custo",
    }


def test_generate_payment_records_unique_cnpj():
    df = generate_payment_records(n=100, seed=42)
    assert df["cnpj_fornecedor"].nunique() == 100


def test_generate_payment_records_cnpj_length():
    df = generate_payment_records(n=50, seed=42)
    assert df["cnpj_fornecedor"].str.len().eq(14).all()


def test_generate_payment_records_reproducible():
    df1 = generate_payment_records(n=20, seed=42)
    df2 = generate_payment_records(n=20, seed=42)
    assert df1.equals(df2)


def test_generate_payment_records_positive_values():
    df = generate_payment_records(n=50, seed=42)
    assert (df["valor_pago"] > 0).all()


def test_write_excel_creates_readable_file(tmp_path):
    df = generate_payment_records(n=10, seed=42)
    out = tmp_path / "pagamentos.xlsx"
    write_excel(df, out)
    assert out.exists()
    loaded = pd.read_excel(out, engine="openpyxl")
    assert len(loaded) == 10
    assert "id_pagamento" in loaded.columns
