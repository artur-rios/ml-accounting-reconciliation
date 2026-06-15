import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build ML-ready features from merged payment and NFS-e records.

    Args:
        df: Merged DataFrame from label_records with columns like delta_days,
            delta_valor_pct, valor_pago, nfse_valor_iss, nfse_aliquota,
            cnpj_fornecedor, descricao, nfse_discriminacao, and label.

    Returns:
        Tuple of (feature_matrix, labels) where feature_matrix is a DataFrame
        with computed features and labels is a Series with target values.
    """
    feat = pd.DataFrame(index=df.index)

    # Fill numeric features with appropriate defaults
    feat["delta_days"] = df["delta_days"].fillna(9999.0).astype(float)
    feat["delta_valor_pct"] = df["delta_valor_pct"].fillna(100.0).astype(float)
    feat["valor_pago"] = df["valor_pago"].fillna(0.0).astype(float)
    feat["nfse_valor_iss"] = df["nfse_valor_iss"].fillna(0.0).astype(float)
    feat["nfse_aliquota"] = df["nfse_aliquota"].fillna(0.0).astype(float)

    # CNPJ match flag: 1 if cnpj_fornecedor is not null, 0 otherwise
    feat["cnpj_match"] = df["cnpj_fornecedor"].notna().astype(int)

    # TF-IDF cosine similarity between payment description and NFS-e description
    desc_pag = df["descricao"].fillna("").astype(str).tolist()
    desc_nfse = df["nfse_discriminacao"].fillna("").astype(str).tolist()
    all_texts = desc_pag + desc_nfse

    try:
        vectorizer = TfidfVectorizer(min_df=1)
        tfidf = vectorizer.fit_transform(all_texts)
        n = len(df)
        similarities = cosine_similarity(tfidf[:n], tfidf[n:]).diagonal()
        feat["descricao_similarity"] = similarities
    except ValueError:
        feat["descricao_similarity"] = 0.0

    # Extract labels
    y = df["label"].fillna(0).astype(int).reset_index(drop=True)
    feat = feat.reset_index(drop=True)

    return feat, y
