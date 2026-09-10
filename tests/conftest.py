import matplotlib

# Force the non-interactive Agg backend before any test imports pyplot.
# evaluator.py only ever saves figures to disk (fig.savefig) and never
# displays them, so the default GUI backend (TkAgg on this machine) is
# unnecessary and its Tk/Tcl window creation is a flaky dependency in a
# headless test run.
matplotlib.use("Agg")

import pytest
import pandas as pd
import datetime


@pytest.fixture
def sample_config():
    return {
        "simulation": {"n_records": 50, "conciliation_rate": 0.70, "random_seed": 42},
        "scenarios": {
            "exact": {"date_tolerance_days": 0, "value_tolerance_pct": 0.0, "n_classes": 2},
            "fuzzy": {"date_tolerance_days": 5, "value_tolerance_pct": 2.0, "n_classes": 3},
        },
        "models": {
            "random_forest": {"n_estimators": 10, "class_weight": "balanced_subsample"},
            "svm": {"kernel": "rbf", "class_weight": "balanced"},
            "logistic_regression": {"l1_ratio": 0, "class_weight": "balanced"},
            "cv_folds": 2,
            "scoring": "f1_macro",
            "test_size": 0.20,
        },
    }


@pytest.fixture
def sample_pagamentos_clean():
    return pd.DataFrame({
        "id_pagamento": ["PAG-000001", "PAG-000002", "PAG-000003"],
        "cnpj_fornecedor": ["11222333000181", "44555666000195", "77888999000177"],
        "data_pagamento": [
            datetime.date(2024, 1, 15),
            datetime.date(2024, 2, 10),
            datetime.date(2024, 3, 20),
        ],
        "valor_pago": [1000.00, 2500.00, 800.00],
        "descricao": ["consultoria TI", "manutencao predial", "limpeza escritorio"],
        "centro_custo": ["CC-001", "CC-002", "CC-003"],
        "cnpj_valid": [True, True, True],
        "is_duplicate": [False, False, False],
    })


@pytest.fixture
def sample_nfse_clean():
    return pd.DataFrame({
        "nfse_numero": ["000001", "000002", "000003"],
        "nfse_codigo_verificacao": ["ABC12345", "DEF67890", "GHI11111"],
        "nfse_data_emissao": [
            datetime.date(2024, 1, 15),
            datetime.date(2024, 2, 15),
            datetime.date(2024, 3, 20),
        ],
        "nfse_competencia": ["2024-01-01T00:00:00", "2024-02-01T00:00:00", "2024-03-01T00:00:00"],
        "nfse_cnpj_prestador": ["11222333000181", "44555666000195", "00000000000000"],
        "nfse_razao_social_prestador": ["Empresa A LTDA", "Empresa B SA", "Empresa C ME"],
        "nfse_cnpj_tomador": ["99888777000166"] * 3,
        "nfse_razao_social_tomador": ["Tomador XYZ SA"] * 3,
        "nfse_valor_servicos": [1000.00, 2600.00, 800.00],
        "nfse_valor_iss": [50.00, 130.00, 40.00],
        "nfse_aliquota": [5.00, 5.00, 5.00],
        "nfse_valor_liquido": [950.00, 2470.00, 760.00],
        "nfse_item_lista_servico": ["1.01", "1.02", "1.03"],
        "nfse_discriminacao": ["Servicos de TI", "Manutencao Predial", "Limpeza"],
        "nfse_codigo_municipio": ["3550308", "3550308", "3550308"],
        "nfse_cnpj_valid": [True, True, False],
        "is_duplicate": [False, False, False],
    })


@pytest.fixture
def comparison_config():
    return {
        "n_seeds": 2,
        "n_records": 400,
        "match_rate": 0.70,
        "hard_negative_rate": 0.30,
        "atypical_term_rate": 0.10,
        "term_shift_range": [-30, 60],
        "atypical_description_rate": 0.12,
        "same_municipality_rate": 0.85,
        "payment_terms_days": [0, 15, 30, 45, 60],
        "csrf_threshold_brl": 5000.0,
        "invoice_value_range_brl": [500.0, 50000.0],
        "iss_rate_range_pct": [2.0, 5.0],
        "retention_rates": {"irrf": 1.50, "csrf": 4.65, "inss": 11.00},
        "retention_tolerance_pp": 0.05,
        "split": {"train": 0.60, "val": 0.20, "test": 0.20},
        "min_precision": 0.90,
        "cv_folds": 2,
        "scoring": "average_precision_excecao",
        "leak_guard_max_stump_accuracy": 0.95,
    }
