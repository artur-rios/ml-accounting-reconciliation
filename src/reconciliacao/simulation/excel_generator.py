"""Simulate the supplier payment ledger, deterministically.

Two independent sources of non-determinism were removed from this module, and
both are worth naming because each produced a pipeline that looked
reproducible and was not.

The first was ``list(set(cnpjs))``. Registration numbers were accumulated in a
set and then listed, but the iteration order of a set of strings depends on
per-process string hashing, which CPython randomises at interpreter start
unless ``PYTHONHASHSEED`` is pinned. Every run therefore assigned a *different*
registration number to each payment while drawing the identical amounts,
dates and descriptions -- a permutation invisible in every aggregate, which is
why the existing reproducibility test never saw it: it compared two calls
inside one process, where the hash seed is fixed. The list is now built in
draw order, which is both deterministic and the order the random stream
actually produced.

The second was ``end_date="today"``. Faker resolves the relative window
against the wall clock, so the twelve-month span slid forward every day and
the generated ledger silently depended on when it was run. The window is now
anchored to an explicit reference date.
"""

import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

from reconciliacao.utils.cnpj import generate_cnpj

# The date the published dataset was generated. Pinned rather than read from
# the clock so the simulated ledger is a function of the configuration alone.
DEFAULT_REFERENCE_DATE = date(2026, 6, 14)
_WINDOW_DAYS = 365


def generate_payment_records(
    n: int, seed: int, reference_date: date = DEFAULT_REFERENCE_DATE
) -> pd.DataFrame:
    rng = random.Random(seed)
    Faker.seed(seed)
    fake = Faker("pt_BR")

    # Generate N unique valid CNPJs upfront to guarantee 1:1 join later.
    # Accumulated in a list, in draw order: see the module docstring for why
    # the set that used to hold them made the output non-reproducible.
    cnpj_list: list[str] = []
    seen: set[str] = set()
    while len(seen) < n:
        cnpj = generate_cnpj(rng)
        if cnpj not in seen:
            seen.add(cnpj)
            cnpj_list.append(cnpj)

    window_start = reference_date - timedelta(days=_WINDOW_DAYS)

    records = []
    for i, cnpj in enumerate(cnpj_list):
        records.append({
            "id_pagamento": f"PAG-{i + 1:06d}",
            "cnpj_fornecedor": cnpj,
            "data_pagamento": fake.date_between(
                start_date=window_start, end_date=reference_date
            ),
            "valor_pago": round(rng.uniform(500.0, 50_000.0), 2),
            "descricao": fake.bs(),
            "centro_custo": f"CC-{rng.randint(100, 999)}",
        })
    return pd.DataFrame(records)


def write_excel(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False, engine="openpyxl")
