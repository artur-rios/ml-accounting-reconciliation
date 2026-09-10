"""Pair each payment with its candidate invoice and label from ground truth.

The label comes from the truth table and from nothing else. No observable
quantity -- date delta, value ratio, text similarity -- takes part in it.
"""

import pandas as pd


def label_from_truth(
    df_pag: pd.DataFrame,
    df_nfse: pd.DataFrame,
    df_verdade: pd.DataFrame,
) -> pd.DataFrame:
    """Join payment to its candidate invoice and label the pairing.

    Returns one row per payment with label 1 when the candidate invoice is the
    one the truth table records, 0 otherwise. tipo_negativo is dropped: it is
    diagnostic metadata, never a feature.
    """
    merged = df_pag.merge(
        df_nfse,
        left_on="cnpj_fornecedor",
        right_on="nfse_cnpj_prestador",
        how="inner",
    )

    truth = df_verdade[["id_pagamento", "nfse_numero"]].rename(
        columns={"nfse_numero": "nfse_verdadeira"}
    )
    merged = merged.merge(truth, on="id_pagamento", how="left")

    candidata = merged["nfse_numero"].fillna("").astype(str)
    verdadeira = merged["nfse_verdadeira"].fillna("").astype(str)
    merged["label"] = ((verdadeira != "") & (candidata == verdadeira)).astype(int)

    return merged.drop(columns=["nfse_verdadeira"])
