import pandas as pd


def label_records(
    df_pag: pd.DataFrame,
    df_nfse: pd.DataFrame,
    date_tolerance_days: int,
    value_tolerance_pct: float,
    n_classes: int,
) -> pd.DataFrame:
    """
    Label payment records by matching with NFS-e records based on CNPJ, date, and value.

    Args:
        df_pag: Payment records DataFrame with columns: id_pagamento, cnpj_fornecedor,
                data_pagamento, valor_pago, etc.
        df_nfse: NFS-e records DataFrame with columns: nfse_cnpj_prestador, nfse_data_emissao,
                 nfse_valor_servicos, etc.
        date_tolerance_days: Maximum days difference allowed between payment and NFS-e date
        value_tolerance_pct: Maximum percentage difference allowed between amounts
        n_classes: Number of label classes (2 for binary, 3 for ternary with partial)

    Returns:
        Merged DataFrame with added columns: label, delta_days, delta_valor_pct
    """
    # Left join on CNPJ to keep all payment records
    merged = df_pag.merge(
        df_nfse,
        left_on="cnpj_fornecedor",
        right_on="nfse_cnpj_prestador",
        how="left",
    )

    # Calculate delta_days (days difference between payment and NFS-e date)
    pag_dates = pd.to_datetime(merged["data_pagamento"])
    nfse_dates = pd.to_datetime(merged["nfse_data_emissao"])
    merged["delta_days"] = (pag_dates - nfse_dates).abs().dt.days.fillna(9999)

    # Calculate delta_valor_pct (percentage difference)
    # When nfse_valor_servicos is 0, fill with NaN to get 100% delta
    safe_valor = merged["nfse_valor_servicos"].replace(0, float("nan"))
    merged["delta_valor_pct"] = (
        (merged["valor_pago"] - merged["nfse_valor_servicos"]).abs() / safe_valor * 100
    ).fillna(100.0)

    # Initialize all labels as 0 (not conciliated)
    merged["label"] = 0

    # Create masks for matching criteria
    mask_date_ok = merged["delta_days"] <= date_tolerance_days
    mask_valor_ok = merged["delta_valor_pct"] <= value_tolerance_pct

    if n_classes == 2:
        # Binary: 0 (not conciliated) or 1 (conciliated)
        merged.loc[mask_date_ok & mask_valor_ok, "label"] = 1
    else:
        # Ternary: 0 (not conciliated), 1 (partially conciliated), 2 (fully conciliated)
        # Partially conciliated: within 4× date tolerance OR within 5× value tolerance
        mask_date_close = merged["delta_days"] <= date_tolerance_days * 4
        mask_valor_close = merged["delta_valor_pct"] <= value_tolerance_pct * 5
        mask_parcial = (~(mask_date_ok & mask_valor_ok)) & (mask_date_close | mask_valor_close)

        merged.loc[mask_parcial, "label"] = 1
        merged.loc[mask_date_ok & mask_valor_ok, "label"] = 2

    return merged
