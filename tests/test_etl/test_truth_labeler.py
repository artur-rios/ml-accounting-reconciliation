import pandas as pd

from reconciliacao.etl.truth_labeler import label_from_truth


def _fixtures_with_third_row():
    """Three payments/invoices: one true pair, one non-pair, one non-numeric."""
    pag = pd.DataFrame({
        "id_pagamento": ["PAG-000001", "PAG-000002", "PAG-000003"],
        "cnpj_fornecedor": [
            "11222333000181",
            "44555666000195",
            "77888999000102",
        ],
        "data_pagamento": pd.to_datetime(
            ["2026-01-15", "2026-02-10", "2026-03-05"]
        ),
        "valor_pago": [1000.00, 2500.00, 750.00],
        "descricao": ["consultoria", "manutencao", "auditoria"],
        "centro_custo": ["CC-101", "CC-102", "CC-103"],
        "municipio_centro_custo": ["3550308", "3550308", "3550308"],
    })
    nfse = pd.DataFrame({
        "nfse_numero": ["000001", "000002", "NF-2026-A"],
        "nfse_cnpj_prestador": [
            "11222333000181",
            "44555666000195",
            "77888999000102",
        ],
        "nfse_data_emissao": [
            "2026-01-15T00:00:00",
            "2026-02-10T00:00:00",
            "2026-03-05T00:00:00",
        ],
        "nfse_valor_servicos": [1000.00, 2500.00, 750.00],
        "nfse_valor_iss": [50.00, 125.00, 37.50],
        "nfse_aliquota": [5.00, 5.00, 5.00],
        "nfse_discriminacao": ["consultoria", "manutencao", "auditoria"],
        "nfse_codigo_municipio": ["3550308", "3550308", "3550308"],
    })
    return pag, nfse


def _fixtures():
    pag = pd.DataFrame({
        "id_pagamento": ["PAG-000001", "PAG-000002"],
        "cnpj_fornecedor": ["11222333000181", "44555666000195"],
        "data_pagamento": pd.to_datetime(["2026-01-15", "2026-02-10"]),
        "valor_pago": [1000.00, 2500.00],
        "descricao": ["consultoria", "manutencao"],
        "centro_custo": ["CC-101", "CC-102"],
        "municipio_centro_custo": ["3550308", "3550308"],
    })
    nfse = pd.DataFrame({
        "nfse_numero": ["000001", "000002"],
        "nfse_cnpj_prestador": ["11222333000181", "44555666000195"],
        "nfse_data_emissao": ["2026-01-15T00:00:00", "2026-02-10T00:00:00"],
        "nfse_valor_servicos": [1000.00, 2500.00],
        "nfse_valor_iss": [50.00, 125.00],
        "nfse_aliquota": [5.00, 5.00],
        "nfse_discriminacao": ["consultoria", "manutencao"],
        "nfse_codigo_municipio": ["3550308", "3550308"],
    })
    return pag, nfse


def test_true_pair_is_labeled_one():
    pag, nfse = _fixtures()
    verdade = pd.DataFrame({
        "id_pagamento": ["PAG-000001", "PAG-000002"],
        "nfse_numero": ["000001", ""],
        "tipo_negativo": ["", "hard"],
    })
    out = label_from_truth(pag, nfse, verdade)
    assert out.loc[out["id_pagamento"] == "PAG-000001", "label"].iloc[0] == 1


def test_label_ignores_perfect_deltas_when_truth_says_no_pair():
    """PAG-000002 matches on date and value exactly, yet is not a true pair.

    This is the property the original experiment lacked: no combination of
    observable deltas can override the ground truth.
    """
    pag, nfse = _fixtures()
    verdade = pd.DataFrame({
        "id_pagamento": ["PAG-000001", "PAG-000002"],
        "nfse_numero": ["000001", ""],
        "tipo_negativo": ["", "hard"],
    })
    out = label_from_truth(pag, nfse, verdade)
    row = out[out["id_pagamento"] == "PAG-000002"].iloc[0]
    assert row["valor_pago"] == row["nfse_valor_servicos"]
    assert row["label"] == 0


def test_every_payment_keeps_exactly_one_candidate_invoice():
    pag, nfse = _fixtures()
    verdade = pd.DataFrame({
        "id_pagamento": ["PAG-000001", "PAG-000002"],
        "nfse_numero": ["000001", "000002"],
        "tipo_negativo": ["", ""],
    })
    out = label_from_truth(pag, nfse, verdade)
    assert len(out) == 2
    assert out["id_pagamento"].is_unique


def test_truth_columns_are_not_leaked_into_the_output():
    pag, nfse = _fixtures()
    verdade = pd.DataFrame({
        "id_pagamento": ["PAG-000001", "PAG-000002"],
        "nfse_numero": ["000001", ""],
        "tipo_negativo": ["", "soft"],
    })
    out = label_from_truth(pag, nfse, verdade)
    assert "tipo_negativo" not in out.columns
    assert "nfse_verdadeira" not in out.columns


def test_true_pair_survives_real_csv_round_trip(tmp_path):
    """A CSV round-trip lets pandas infer numeric dtypes for nfse_numero.

    A clean column becomes int64 ("000001" -> 1); a column with a blank
    becomes float64 ("000001" -> 1.0, "" -> NaN). The true pair must still
    label 1 after both frames have actually gone through to_csv/read_csv --
    not a hand-built dtype substitute -- to pin real pandas behaviour.
    """
    pag, nfse = _fixtures()
    verdade = pd.DataFrame({
        "id_pagamento": ["PAG-000001", "PAG-000002"],
        "nfse_numero": ["000001", ""],
        "tipo_negativo": ["", "hard"],
    })

    pag_path = tmp_path / "pag.csv"
    nfse_path = tmp_path / "nfse.csv"
    verdade_path = tmp_path / "verdade.csv"
    pag.to_csv(pag_path, index=False)
    nfse.to_csv(nfse_path, index=False)
    verdade.to_csv(verdade_path, index=False)

    # cnpj_fornecedor / nfse_cnpj_prestador are read back as strings, matching
    # how the real ETL loads them (see F3 note: the join depends on both
    # sides sharing dtype), so only nfse_numero's inferred dtype is at play.
    pag_roundtrip = pd.read_csv(
        pag_path, dtype={"cnpj_fornecedor": str}, parse_dates=["data_pagamento"]
    )
    nfse_roundtrip = pd.read_csv(nfse_path, dtype={"nfse_cnpj_prestador": str})
    verdade_roundtrip = pd.read_csv(verdade_path)

    # Confirm the round-trip actually changed the dtypes -- otherwise this
    # test would not exercise the defect at all.
    assert nfse_roundtrip["nfse_numero"].dtype != object
    assert verdade_roundtrip["nfse_numero"].dtype != object

    out = label_from_truth(pag_roundtrip, nfse_roundtrip, verdade_roundtrip)
    assert out.loc[out["id_pagamento"] == "PAG-000001", "label"].iloc[0] == 1


def test_true_pair_labeled_one_with_mixed_numeric_dtypes():
    """Truth column float64 with NaN, candidate column int64: still labels 1."""
    pag, nfse = _fixtures()
    nfse = nfse.copy()
    nfse["nfse_numero"] = nfse["nfse_numero"].astype("int64")
    verdade = pd.DataFrame({
        "id_pagamento": ["PAG-000001", "PAG-000002"],
        "nfse_numero": [1.0, float("nan")],
        "tipo_negativo": ["", "hard"],
    })
    assert verdade["nfse_numero"].dtype == "float64"

    out = label_from_truth(pag, nfse, verdade)
    assert out.loc[out["id_pagamento"] == "PAG-000001", "label"].iloc[0] == 1


def test_non_pair_still_labeled_zero_with_mixed_numeric_dtypes():
    """The fix must not make everything match: non-pairs stay labeled 0."""
    pag, nfse = _fixtures()
    nfse = nfse.copy()
    nfse["nfse_numero"] = nfse["nfse_numero"].astype("int64")
    verdade = pd.DataFrame({
        "id_pagamento": ["PAG-000001", "PAG-000002"],
        "nfse_numero": [1.0, float("nan")],
        "tipo_negativo": ["", "hard"],
    })

    out = label_from_truth(pag, nfse, verdade)
    assert out.loc[out["id_pagamento"] == "PAG-000002", "label"].iloc[0] == 0


def test_non_numeric_invoice_number_matches_correctly():
    """A non-digit invoice number must not be swallowed by the numeric branch."""
    pag, nfse = _fixtures_with_third_row()
    verdade = pd.DataFrame({
        "id_pagamento": ["PAG-000001", "PAG-000002", "PAG-000003"],
        "nfse_numero": ["000001", "", "NF-2026-A"],
        "tipo_negativo": ["", "hard", ""],
    })

    out = label_from_truth(pag, nfse, verdade)
    assert out.loc[out["id_pagamento"] == "PAG-000003", "label"].iloc[0] == 1

    # A near-miss non-numeric value must not match.
    verdade_miss = verdade.copy()
    verdade_miss.loc[
        verdade_miss["id_pagamento"] == "PAG-000003", "nfse_numero"
    ] = "NF-2026-B"
    out_miss = label_from_truth(pag, nfse, verdade_miss)
    assert out_miss.loc[out_miss["id_pagamento"] == "PAG-000003", "label"].iloc[0] == 0


def test_large_invoice_numbers_are_not_collapsed_by_float_precision():
    """Two distinct 18-digit invoice numbers must not canonicalise to the same key.

    float() only carries 53 bits of mantissa, so
    int(float("100000000000000001")) == int(float("100000000000000002")).
    A non-pair whose candidate happens to carry one of these large numbers
    must not be swept into a false match against the truth value.
    """
    pag, nfse = _fixtures()
    nfse = nfse.copy()
    nfse.loc[nfse["nfse_cnpj_prestador"] == "44555666000195", "nfse_numero"] = (
        "100000000000000002"
    )
    verdade = pd.DataFrame({
        "id_pagamento": ["PAG-000001", "PAG-000002"],
        "nfse_numero": ["000001", "100000000000000001"],
        "tipo_negativo": ["", "hard"],
    })

    out = label_from_truth(pag, nfse, verdade)
    assert out.loc[out["id_pagamento"] == "PAG-000002", "label"].iloc[0] == 0
