import pytest
from reconciliacao.etl.labeler import label_records


def test_exact_match_labels_conciliated(sample_pagamentos_clean, sample_nfse_clean):
    # PAG-000001: CNPJ match, date match, value match → label 1
    result = label_records(sample_pagamentos_clean, sample_nfse_clean,
                           date_tolerance_days=0, value_tolerance_pct=0.0, n_classes=2)
    row = result[result["id_pagamento"] == "PAG-000001"].iloc[0]
    assert row["label"] == 1


def test_exact_match_labels_date_mismatch_as_not_conciliated(sample_pagamentos_clean, sample_nfse_clean):
    # PAG-000002: CNPJ match, date off by 5 days, value off → label 0
    result = label_records(sample_pagamentos_clean, sample_nfse_clean,
                           date_tolerance_days=0, value_tolerance_pct=0.0, n_classes=2)
    row = result[result["id_pagamento"] == "PAG-000002"].iloc[0]
    assert row["label"] == 0


def test_fuzzy_match_labels_within_tolerance(sample_pagamentos_clean, sample_nfse_clean):
    # PAG-000002: CNPJ match, 5-day tolerance, value is 2600 vs 2500 (4% off) → label 0 or 1
    result = label_records(sample_pagamentos_clean, sample_nfse_clean,
                           date_tolerance_days=5, value_tolerance_pct=2.0, n_classes=3)
    row = result[result["id_pagamento"] == "PAG-000002"].iloc[0]
    # delta_days=5 ≤ 5 ✓ BUT delta_valor_pct=4% > 2% → not fully conciliated
    assert row["label"] in (0, 1)


def test_fuzzy_match_fully_conciliated(sample_pagamentos_clean, sample_nfse_clean):
    # PAG-000001: CNPJ match, date exact, value exact → label 2
    result = label_records(sample_pagamentos_clean, sample_nfse_clean,
                           date_tolerance_days=5, value_tolerance_pct=2.0, n_classes=3)
    row = result[result["id_pagamento"] == "PAG-000001"].iloc[0]
    assert row["label"] == 2


def test_unmatched_cnpj_labeled_zero(sample_pagamentos_clean, sample_nfse_clean):
    # PAG-000003 has CNPJ 77888999000177 but NFS-e row 3 has 00000000000000 → no match → label 0
    result = label_records(sample_pagamentos_clean, sample_nfse_clean,
                           date_tolerance_days=0, value_tolerance_pct=0.0, n_classes=2)
    row = result[result["id_pagamento"] == "PAG-000003"].iloc[0]
    assert row["label"] == 0


def test_result_has_label_and_delta_columns(sample_pagamentos_clean, sample_nfse_clean):
    result = label_records(sample_pagamentos_clean, sample_nfse_clean,
                           date_tolerance_days=0, value_tolerance_pct=0.0, n_classes=2)
    assert "label" in result.columns
    assert "delta_days" in result.columns
    assert "delta_valor_pct" in result.columns
