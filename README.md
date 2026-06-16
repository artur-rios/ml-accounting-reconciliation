# Financial Reconciliation Automation with Machine Learning

End of course project for the Software Engineering MBA at USP - ESALQ. Licensed under the [MIT License](LICENSE). Automates financial reconciliation using supervised machine learning by simulating two real-world data sources (supplier payment records in Excel and electronic service invoices / NFS-e in XML), running them through an ETL pipeline to produce labeled datasets, then training and evaluating three classifiers across two experimental scenarios.

## Research Question

Which algorithm (Random Forest, SVM, or Logistic Regression) best automates financial reconciliation, and how does label strictness affect that ranking?

## Scenarios

| Scenario | Date tolerance | Value tolerance | Labels |
| --- | --- | --- | --- |
| `exact` | 0 days | 0% | 2 (conciliated / not conciliated) |
| `fuzzy` | ±5 business days | ±2% | 3 (conciliated / partially / not conciliated) |

## Project Structure

```text
├── config.yaml              # All experiment parameters
├── run_pipeline.py          # End-to-end CLI runner
├── pyproject.toml
├── data/                    # Generated at runtime (not versioned)
│   ├── raw/                 # Simulated .xlsx and .xml files
│   ├── processed/           # Post-ETL labeled CSVs
│   └── results/
│       ├── exact/           # Metrics, plots, and models for exact scenario
│       └── fuzzy/           # Metrics, plots, and models for fuzzy scenario
├── src/reconciliacao/
│   ├── simulation/          # Synthetic data generators (Excel + XML/ABRASF)
│   ├── etl/                 # Loader, cleaner, labeler
│   ├── models/              # Feature engineering, trainer, evaluator
│   └── utils/               # Config loader, CNPJ validation
├── notebooks/
│   ├── 01_simulacao.ipynb   # Data generation methodology
│   ├── 02_etl.ipynb         # EDA, normalization, cleaning
│   ├── 03_modelagem.ipynb   # Training and cross-validation
│   └── 04_avaliacao.ipynb   # Results, cross-scenario comparison, error analysis
└── tests/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -e ".[dev]"
```

## Running the Pipeline

```bash
python run_pipeline.py --scenario exact
python run_pipeline.py --scenario fuzzy
```

Each run:

1. Simulates 7 500 payment records and matching NFS-e XML files
2. Runs ETL (load → clean → label)
3. Engineers features and trains RF, SVM, and LR via GridSearchCV (5-fold CV, F1-macro)
4. Evaluates and saves metrics, confusion matrices, ROC curve, and feature importance plots

All outputs are deterministic given the same `config.yaml` and `random_seed`.

## Configuration

All parameters are in `config.yaml` — no hardcoded values in source modules.

```yaml
simulation:
  n_records: 7500
  conciliation_rate: 0.70   # 70% conciliated, 30% not
  random_seed: 42
```

## Evaluation Artifacts

Saved to `data/results/<scenario>/`:

```text
metrics_summary.csv
confusion_matrix_rf.png
confusion_matrix_svm.png
confusion_matrix_lr.png
roc_curve.png               # exact scenario only (binary)
feature_importance_rf.png
models/
  random_forest.joblib
  svm.joblib
  logistic_regression.joblib
```

## Features

| Feature | Derivation |
| --- | --- |
| `delta_days` | `abs(data_pagamento − DataEmissao)` in days |
| `delta_valor_pct` | `abs(valor_pago − ValorServicos) / ValorServicos` |
| `valor_pago` | Raw payment amount |
| `ValorIss` | ISS tax from NFS-e |
| `Aliquota` | ISS rate |
| `descricao_similarity` | TF-IDF cosine similarity between payment description and NFS-e discriminação |
| `is_mesmo_municipio` | Boolean: IBGE municipality code matches company |

## Tests

```bash
pytest
pytest -m "not integration"   # skip slow tests
pytest --cov=src/reconciliacao --cov-report=term-missing
```

## Dependencies

- **Core:** pandas, numpy, scikit-learn, scipy
- **Data I/O:** openpyxl, lxml, pyyaml, joblib
- **Simulation:** Faker (pt_BR locale)
- **Visualization:** matplotlib, seaborn
- **Dev:** pytest, pytest-cov, jupyter, ipykernel
