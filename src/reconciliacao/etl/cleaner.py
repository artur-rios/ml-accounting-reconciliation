"""Normalise the two raw sources onto comparable types.

Note on the diagnostic columns. `cnpj_valid`, `nfse_cnpj_valid`,
`pagamento_duplicado` and `nfse_duplicada` are computed here but read by
nothing downstream: no row is filtered on them and no feature is derived from
them. They are descriptive output of the cleaning step, not controls over it.
This is deliberate and has a consequence worth stating -- because the labeler
joins on the supplier registration number, a payment whose invoice carries a
corrupted number simply fails to join and becomes "no correspondence", so the
`cnpj_valid` flag never reaches the classifier and CNPJ errors are resolved
upstream of the model rather than by it.
"""

import pandas as pd

from reconciliacao.utils.cnpj import normalize_cnpj, validate_cnpj


def _parse_float(series: pd.Series) -> pd.Series:
    # Try English format first ("1234.56"), fall back to Brazilian ("1.234,56") for NaN
    s = series.astype(str).str.strip()
    result = pd.to_numeric(s, errors="coerce")
    if result.isna().any():
        br_cleaned = s.str.replace(r"\.", "", regex=True).str.replace(",", ".", regex=False)
        result = result.fillna(pd.to_numeric(br_cleaned, errors="coerce"))
    return result


def _parse_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.date


def clean_pagamentos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cnpj_fornecedor"] = df["cnpj_fornecedor"].astype(str).apply(normalize_cnpj)
    df["cnpj_valid"] = df["cnpj_fornecedor"].apply(validate_cnpj)
    df["data_pagamento"] = _parse_date(df["data_pagamento"])
    df["valor_pago"] = _parse_float(df["valor_pago"].astype(str))
    df["pagamento_duplicado"] = df.duplicated("id_pagamento", keep=False)
    return df


def clean_nfse(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["nfse_cnpj_prestador"] = df["nfse_cnpj_prestador"].astype(str).apply(normalize_cnpj)
    df["nfse_cnpj_valid"] = df["nfse_cnpj_prestador"].apply(validate_cnpj)
    df["nfse_data_emissao"] = _parse_date(df["nfse_data_emissao"])
    df["nfse_valor_servicos"] = pd.to_numeric(df["nfse_valor_servicos"], errors="coerce")
    df["nfse_valor_iss"] = pd.to_numeric(df["nfse_valor_iss"], errors="coerce")
    df["nfse_aliquota"] = pd.to_numeric(df["nfse_aliquota"], errors="coerce")
    df["nfse_valor_liquido"] = pd.to_numeric(df["nfse_valor_liquido"], errors="coerce")
    df["nfse_duplicada"] = df.duplicated("nfse_numero", keep=False)
    return df
