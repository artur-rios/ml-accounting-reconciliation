"""Observable evidence about a payment-invoice pairing.

Every feature is computable by someone who does not know whether the pairing is
genuine. Nothing here participates in defining the label.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reconciliacao.simulation.retentions import compatibility_flags

FEATURE_ORDER = [
    "delta_dias",
    "razao_valor",
    "retencao_implicita_pct",
    "valor_retido_brl",
    "compat_irrf",
    "compat_csrf",
    "compat_irrf_csrf",
    "compat_iss",
    "compat_combo_iss",
    "compat_inss",
    "acima_limite_csrf",
    "desvio_prazo_fornecedor",
    "similaridade_descricao",
    "mesmo_municipio",
]


def _delta_dias(df: pd.DataFrame) -> pd.Series:
    pagamento = pd.to_datetime(df["data_pagamento"])
    emissao = pd.to_datetime(df["nfse_data_emissao"])
    return (pagamento - emissao).dt.days.astype(float)


def fit_supplier_terms(df_train: pd.DataFrame) -> tuple[dict[str, float], float]:
    """Median payment term per supplier, fitted on the training partition only.

    An empty training frame cannot produce a global median and signals a
    programming error in the caller's split logic, not a legitimate data
    scenario, so it is rejected rather than silently producing a NaN
    fallback that would poison every row's `desvio_prazo_fornecedor`.
    """
    if len(df_train) == 0:
        raise ValueError(
            "fit_supplier_terms recebeu uma partição de treino vazia; "
            "isso indica um erro no split treino/teste (não um cenário de "
            "dados legítimo), pois a mediana global não pode ser calculada."
        )
    delta = _delta_dias(df_train)
    por_fornecedor = delta.groupby(df_train["cnpj_fornecedor"]).median()
    return por_fornecedor.to_dict(), float(delta.median())


def build_features_v2(
    df: pd.DataFrame,
    cmp_cfg: dict,
    supplier_terms: dict[str, float],
    global_term: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the evidence matrix and the target from a labeled pairing frame."""
    rates = cmp_cfg["retention_rates"]
    tol = cmp_cfg["retention_tolerance_pp"]
    limite = cmp_cfg["csrf_threshold_brl"]

    df = df.reset_index(drop=True)

    valor_servicos = df["nfse_valor_servicos"].astype(float)
    valor_pago = df["valor_pago"].astype(float)

    # `nfse_valor_servicos` is guaranteed strictly positive by the upstream
    # generator, and the labeler inner-joins on it, so a non-positive or
    # missing value here means an upstream invariant has already broken.
    # A sentinel substitution (e.g. filling with 0 or a magic constant)
    # would repeat the `delta_days = 9999` mistake this project exists to
    # correct: it would hide the break and corrupt the feature scale
    # instead of surfacing it. Fail loudly so the break is diagnosable.
    invalido = valor_servicos.isna() | (valor_servicos <= 0)
    if invalido.any():
        ids_invalidos = df.loc[invalido, "id_pagamento"].tolist()
        total = len(ids_invalidos)
        amostra = ids_invalidos[:5]
        raise ValueError(
            "nfse_valor_servicos deve ser estritamente positivo e não nulo "
            f"para todo pagamento pareado; {total} pagamento(s) violam essa "
            f"invariante, incluindo: {amostra}"
        )

    feat = pd.DataFrame(index=df.index)

    feat["delta_dias"] = _delta_dias(df)
    feat["razao_valor"] = valor_pago / valor_servicos
    feat["retencao_implicita_pct"] = (1.0 - feat["razao_valor"]) * 100.0
    feat["valor_retido_brl"] = valor_servicos - valor_pago

    aliquotas = df["nfse_aliquota"].astype(float)
    flags = [
        compatibility_flags(ret, aliq, rates, tol)
        for ret, aliq in zip(feat["retencao_implicita_pct"], aliquotas)
    ]
    for nome in ["compat_irrf", "compat_csrf", "compat_irrf_csrf",
                 "compat_iss", "compat_combo_iss", "compat_inss"]:
        feat[nome] = [f[nome] for f in flags]

    feat["acima_limite_csrf"] = (valor_servicos > limite).astype(int)

    prazo_esperado = df["cnpj_fornecedor"].map(supplier_terms).fillna(global_term)
    feat["desvio_prazo_fornecedor"] = feat["delta_dias"] - prazo_esperado.astype(float)

    desc_pag = df["descricao"].fillna("").astype(str).tolist()
    desc_nfse = df["nfse_discriminacao"].fillna("").astype(str).tolist()
    try:
        vectorizer = TfidfVectorizer(min_df=1)
        tfidf = vectorizer.fit_transform(desc_pag + desc_nfse)
        n = len(df)
        feat["similaridade_descricao"] = cosine_similarity(tfidf[:n], tfidf[n:]).diagonal()
    except ValueError:
        feat["similaridade_descricao"] = 0.0

    feat["mesmo_municipio"] = (
        df["municipio_centro_custo"].astype(str)
        == df["nfse_codigo_municipio"].astype(str)
    ).astype(int)

    y = df["label"].astype(int)
    return feat[FEATURE_ORDER], y
