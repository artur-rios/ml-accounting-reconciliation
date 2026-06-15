import random
from pathlib import Path

import pandas as pd
from faker import Faker

from reconciliacao.utils.cnpj import generate_cnpj


def generate_payment_records(n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    Faker.seed(seed)
    fake = Faker("pt_BR")

    # Generate N unique valid CNPJs upfront to guarantee 1:1 join later
    cnpjs: set[str] = set()
    while len(cnpjs) < n:
        cnpjs.add(generate_cnpj(rng))
    cnpj_list = list(cnpjs)

    records = []
    for i, cnpj in enumerate(cnpj_list):
        records.append({
            "id_pagamento": f"PAG-{i + 1:06d}",
            "cnpj_fornecedor": cnpj,
            "data_pagamento": fake.date_between(start_date="-1y", end_date="today"),
            "valor_pago": round(rng.uniform(500.0, 50_000.0), 2),
            "descricao": fake.bs(),
            "centro_custo": f"CC-{rng.randint(100, 999)}",
        })
    return pd.DataFrame(records)


def write_excel(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False, engine="openpyxl")
