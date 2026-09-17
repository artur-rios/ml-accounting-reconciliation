import pandas as pd
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


def test_merge_does_not_collide_cleaner_diagnostic_columns(sample_pagamentos_clean, sample_nfse_clean):
    """Both cleaners used to emit a column called `is_duplicate`, so the join
    silently renamed them to `is_duplicate_x` and `is_duplicate_y` and shipped
    both into the persisted dataset, where neither name says which source it
    describes. The columns are now distinct at the source."""
    merged = label_records(
        sample_pagamentos_clean, sample_nfse_clean,
        date_tolerance_days=0, value_tolerance_pct=0.0, n_classes=2,
    )

    suffixed = [c for c in merged.columns if c.endswith("_x") or c.endswith("_y")]
    assert suffixed == [], f"merge produced suffixed columns: {suffixed}"
    assert "pagamento_duplicado" in merged.columns
    assert "nfse_duplicada" in merged.columns


def test_cnpj_join_fanout_is_visible_in_the_output(sample_pagamentos_clean, sample_nfse_clean):
    """The join keys on the supplier registration number, which the generator
    does not guarantee to be unique among invoices. When two invoices carry the
    same number, one payment becomes two rows -- and, because the label is a
    threshold over each row's own deltas, those two rows can disagree about
    whether the same payment was reconciled.

    This is the defect documented in the write-up as the fourth silent failure
    mode. The test does not forbid it (the published artifacts were produced
    with it present); it pins that a fan-out is detectable by counting
    id_pagamento, so the condition can never again be mistaken for a harmless
    row-count discrepancy.
    """
    nfse_duplicada = pd.concat([sample_nfse_clean, sample_nfse_clean.iloc[[0]]], ignore_index=True)
    merged = label_records(
        sample_pagamentos_clean, nfse_duplicada,
        date_tolerance_days=0, value_tolerance_pct=0.0, n_classes=2,
    )

    contagem = merged["id_pagamento"].value_counts()
    assert (contagem > 1).any(), "duplicate supplier numbers must fan the join out"
    assert len(merged) > len(sample_pagamentos_clean)
