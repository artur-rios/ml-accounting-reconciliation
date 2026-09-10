import pandas as pd

from reconciliacao.etl.truth_labeler import label_from_truth


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
