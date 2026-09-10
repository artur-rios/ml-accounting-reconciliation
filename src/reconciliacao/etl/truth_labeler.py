"""Pair each payment with its candidate invoice and label from ground truth.

The label comes from the truth table and from nothing else. No observable
quantity -- date delta, value ratio, text similarity -- takes part in it.
"""

import math

import pandas as pd


def _normalize_numero(valores: pd.Series) -> pd.Series:
    """Canonicalise invoice numbers so a CSV round-trip cannot break equality.

    ``pd.read_csv`` infers numeric dtypes for a column of digit strings, so a
    clean ``nfse_numero`` column becomes ``int64`` (``"000001"`` -> ``1``) and
    one containing blanks becomes ``float64`` (``"000001"`` -> ``1.0``, blank
    -> ``NaN``). Without normalisation the two sides of the comparison
    stringify differently and never match. This maps ``"000001"``, ``"1"``,
    ``"1.0"``, ``1`` and ``1.0`` to the same key, and ``None``, ``NaN``,
    ``""``, whitespace and the literal string ``"nan"`` to the empty string.
    A value that does not parse as numeric survives as its stripped self,
    since invoice numbers are not guaranteed to be digits forever.
    """

    def canonical(valor) -> str:
        if valor is None or (isinstance(valor, float) and math.isnan(valor)):
            return ""
        texto = str(valor).strip()
        if texto == "" or texto.lower() == "nan":
            return ""
        try:
            return str(int(float(texto)))
        except (ValueError, OverflowError):
            return texto

    return valores.map(canonical)


def label_from_truth(
    df_pag: pd.DataFrame,
    df_nfse: pd.DataFrame,
    df_verdade: pd.DataFrame,
) -> pd.DataFrame:
    """Join payment to its candidate invoice and label the pairing.

    Every payment keeps exactly one candidate invoice, provided the caller's
    data contract holds: each ``cnpj_fornecedor`` maps to exactly one
    ``nfse_cnpj_prestador`` row in ``df_nfse``. The inner join below does not
    enforce that -- a CNPJ with no invoice is silently dropped, and one with
    multiple invoices would multiply rows -- it relies on the generator's
    guarantee of one invoice per supplier.

    Label 1 when the candidate invoice is the one the truth table records,
    0 otherwise. tipo_negativo is dropped: it is diagnostic metadata, never a
    feature.
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

    candidata = _normalize_numero(merged["nfse_numero"])
    verdadeira = _normalize_numero(merged["nfse_verdadeira"])
    merged["label"] = ((verdadeira != "") & (candidata == verdadeira)).astype(int)

    return merged.drop(columns=["nfse_verdadeira"])
