"""Pair each payment with its candidate invoice and label from ground truth.

The label comes from the truth table and from nothing else. No observable
quantity -- date delta, value ratio, text similarity -- takes part in it.
"""

import math
import re

import pandas as pd

_INTEGER_RE = re.compile(r"[+-]?\d+")


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

    A pure integer string (optionally signed) is compared exactly via
    ``int()``, at arbitrary precision -- never routed through ``float()``.
    ``float`` only has 53 bits of mantissa, so two distinct 16+ digit
    invoice numbers can round to the same ``float`` and collapse onto the
    same key, producing a false match. The ``float`` path below is reserved
    for genuinely non-integer numeric forms such as ``"1.0"`` or ``"1e3"``,
    which still need to canonicalise to ``"1"`` and ``"1000"``. Do not
    reintroduce ``int(float(texto))`` for the integer case.
    """

    def canonical(valor) -> str:
        if valor is None or (isinstance(valor, float) and math.isnan(valor)):
            return ""
        texto = str(valor).strip()
        if texto == "" or texto.lower() == "nan":
            return ""
        if _INTEGER_RE.fullmatch(texto):
            return str(int(texto))
        try:
            numero = float(texto)
        except (ValueError, OverflowError):
            return texto
        if math.isnan(numero) or math.isinf(numero):
            return texto
        if numero.is_integer():
            return str(int(numero))
        return texto

    return valores.map(canonical)


def label_from_truth(
    df_pag: pd.DataFrame,
    df_nfse: pd.DataFrame,
    df_verdade: pd.DataFrame,
) -> pd.DataFrame:
    """Join payment to its candidate invoice and label the pairing.

    A supplier may have several invoices, so the candidate can no longer be
    found by joining on ``cnpj_fornecedor``: that would multiply rows, one
    per invoice the supplier has. Instead each payment names its candidate
    invoice explicitly in ``nfse_numero_candidata``, and the join keys off
    that against ``nfse_numero``. Both sides are canonicalised the same way
    the label comparison already is -- a CSV round-trip can turn one side
    into ``int64`` and the other into ``float64``, and an un-normalised join
    key would silently drop every row instead of merely mislabeling it.

    Every payment keeps exactly one candidate invoice, provided the caller's
    data contract holds: each ``nfse_numero_candidata`` maps to exactly one
    ``nfse_numero`` row in ``df_nfse``. The inner join below does not enforce
    that -- a candidate with no matching invoice is silently dropped, and one
    matching several invoices would multiply rows -- it relies on the
    generator's guarantee of one invoice per candidate reference.

    Label 1 when the candidate invoice is the one the truth table records,
    0 otherwise. tipo_negativo is dropped: it is diagnostic metadata, never a
    feature.
    """
    chave_pag = _normalize_numero(df_pag["nfse_numero_candidata"]).rename("_chave_candidata")
    chave_nfse = _normalize_numero(df_nfse["nfse_numero"]).rename("_chave_candidata")

    merged = pd.concat([df_pag, chave_pag], axis=1).merge(
        pd.concat([df_nfse, chave_nfse], axis=1),
        on="_chave_candidata",
        how="inner",
    ).drop(columns=["_chave_candidata"])

    truth = df_verdade[["id_pagamento", "nfse_numero"]].rename(
        columns={"nfse_numero": "nfse_verdadeira"}
    )
    merged = merged.merge(truth, on="id_pagamento", how="left")

    candidata = _normalize_numero(merged["nfse_numero"])
    verdadeira = _normalize_numero(merged["nfse_verdadeira"])
    merged["label"] = ((verdadeira != "") & (candidata == verdadeira)).astype(int)

    return merged.drop(columns=["nfse_verdadeira"])
