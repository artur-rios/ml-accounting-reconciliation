import datetime
import os
import subprocess
import sys

import pandas as pd

from reconciliacao.simulation.excel_generator import (
    DEFAULT_REFERENCE_DATE,
    generate_payment_records,
    write_excel,
)


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


def _generate_in_subprocess(hash_seed: str) -> list[str]:
    """Generate the CNPJ column in a fresh interpreter with a chosen
    PYTHONHASHSEED, which is the only way to observe per-process string
    hashing from inside a test."""
    code = (
        "from reconciliacao.simulation.excel_generator import generate_payment_records;"
        "print(chr(10).join(generate_payment_records(n=50, seed=42)['cnpj_fornecedor']))"
    )
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    return out.stdout.split()


def test_generate_payment_records_reproducible_across_processes():
    """Regression test for the defect the in-process test could not see.

    `test_generate_payment_records_reproducible` calls the generator twice in
    one interpreter, where PYTHONHASHSEED is fixed for the process lifetime,
    so a set's iteration order is stable *within* the comparison and the test
    passes however the order was obtained. The registration numbers used to
    come out of `list(set(...))`, and therefore differed between runs of the
    pipeline while every aggregate stayed identical. Two subprocesses with
    deliberately different hash seeds are what makes that visible.
    """
    assert _generate_in_subprocess("0") == _generate_in_subprocess("12345")


def test_generate_payment_records_dates_do_not_depend_on_the_clock():
    """The window was anchored on `today`, so the simulated ledger slid
    forward every day it was regenerated. It is pinned to a reference date."""
    reference = datetime.date(2020, 3, 1)
    df = generate_payment_records(n=50, seed=42, reference_date=reference)

    assert df["data_pagamento"].max() <= reference
    assert df["data_pagamento"].min() >= reference - datetime.timedelta(days=365)
    # And the default is a constant, not a clock read.
    assert generate_payment_records(n=10, seed=42)["data_pagamento"].max() <= DEFAULT_REFERENCE_DATE
