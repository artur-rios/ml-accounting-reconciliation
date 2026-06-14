---
name: reconciliacao-financeira-ml-design
description: Design spec for MBA TCC — ML-based financial reconciliation system with simulated datasets, ETL pipeline, and supervised learning experiments
metadata:
  type: project
---

# Design: Automação de Conciliação Financeira com Machine Learning

**Date:** 2026-06-14
**Project:** MBA TCC — Applied Research

---

## 1. Overview

An applied research project that automates financial reconciliation using supervised machine learning. The system simulates two real-world data sources — supplier payment records (Excel) and electronic service invoices / NFS-e (XML, ABRASF standard) — runs them through an ETL pipeline to produce labeled datasets, then trains and evaluates three classifiers: Random Forest, SVM, and Logistic Regression.

Two experimental scenarios are run against the same raw data, differing only in how "conciliated" is defined:

- **Exact match:** strict equality on CNPJ, date, and value
- **Fuzzy match:** tolerant windows on date (±5 business days) and value (±2%), producing a 3-class label

The primary research question is: which algorithm best automates financial reconciliation, and how does label strictness affect that ranking?

---

## 2. Architecture

### 2.1 Project Structure

```
mba-tcc/
├── config.yaml                  # All experiment parameters
├── run_pipeline.py              # End-to-end runner (--scenario exact|fuzzy)
├── pyproject.toml               # Package definition
├── requirements.txt
├── data/
│   ├── raw/                     # Simulated files (.xlsx + .xml)
│   ├── processed/               # Post-ETL unified CSVs
│   └── results/
│       ├── exact/               # Artifacts for exact scenario
│       └── fuzzy/               # Artifacts for fuzzy scenario
├── src/
│   └── reconciliacao/
│       ├── simulation/
│       │   ├── excel_generator.py
│       │   └── xml_generator.py
│       ├── etl/
│       │   ├── loader.py
│       │   ├── cleaner.py
│       │   └── labeler.py
│       ├── models/
│       │   ├── trainer.py
│       │   └── evaluator.py
│       └── utils/
│           └── config.py
└── notebooks/
    ├── 01_simulacao.ipynb
    ├── 02_etl.ipynb
    ├── 03_modelagem.ipynb
    └── 04_avaliacao.ipynb
```

### 2.2 Pipeline Flow

```
config.yaml
    │
    ▼
simulation/ ──► data/raw/pagamentos.xlsx
             ──► data/raw/nfse/*.xml
    │
    ▼
etl/loader   ──► df_pagamentos, df_nfse
etl/cleaner  ──► normalized DataFrames
etl/labeler  ──► data/processed/<scenario>_reconciliado.csv
    │
    ▼
models/trainer   ──► GridSearchCV (RF, SVM, LR)
models/evaluator ──► data/results/<scenario>/metrics_summary.csv
                 ──► confusion matrices, ROC curve, feature importances
    │
    ▼
notebooks/04_avaliacao.ipynb ──► cross-scenario comparison table
```

---

## 3. Configuration

All parameters live in `config.yaml`. No hardcoded values in source modules.

```yaml
simulation:
  n_records: 7500
  conciliation_rate: 0.70   # 70% conciliated, 30% not
  random_seed: 42

scenarios:
  exact:
    date_tolerance_days: 0
    value_tolerance_pct: 0.0
    n_classes: 2             # conciliado / não conciliado
  fuzzy:
    date_tolerance_days: 5
    value_tolerance_pct: 2.0
    n_classes: 3             # conciliado / parcialmente conciliado / não conciliado

models:
  random_forest:
    n_estimators: 200
    class_weight: balanced_subsample
  svm:
    kernel: rbf
    class_weight: balanced
  logistic_regression:
    penalty: l2
    class_weight: balanced
  cv_folds: 5
  scoring: f1_macro
  test_size: 0.20
```

---

## 4. Data Simulation

### 4.1 Excel — Payment Records (`pagamentos.xlsx`)

Generated using `Faker(locale='pt_BR')`. Fields:

| Field | Type | Notes |
|---|---|---|
| `id_pagamento` | string | UUID |
| `cnpj_fornecedor` | string | 14-digit, validated check digit |
| `data_pagamento` | date | ISO 8601 |
| `valor_pago` | Decimal | BRL, 2 decimal places |
| `descricao` | string | Free-text service description |
| `centro_custo` | string | Cost center code |

### 4.2 XML — NFS-e (ABRASF Standard)

One `.xml` file per invoice. Key fields mapped to ABRASF schema:

| ABRASF Path | Description |
|---|---|
| `InfNfse/Numero` | NFS-e number |
| `InfNfse/CodigoVerificacao` | Verification code |
| `InfNfse/DataEmissao` | Issue datetime |
| `InfNfse/Competencia` | Competency period |
| `PrestadorServico/IdentificacaoPrestador/CpfCnpj/Cnpj` | Supplier CNPJ |
| `PrestadorServico/RazaoSocial` | Supplier name |
| `TomadorServico/IdentificacaoTomador/CpfCnpj/Cnpj` | Buyer CNPJ |
| `TomadorServico/RazaoSocial` | Buyer name |
| `Servico/Valores/ValorServicos` | Gross service value |
| `Servico/Valores/ValorIss` | ISS tax amount |
| `Servico/Valores/Aliquota` | ISS rate |
| `Servico/Valores/ValorLiquidoNfse` | Net value |
| `Servico/ItemListaServico` | LC 116 service code |
| `Servico/Discriminacao` | Service description |
| `Servico/CodigoMunicipio` | IBGE municipality code |

### 4.3 Mismatch Injection (30% non-conciliated records)

| Mismatch Type | Description |
|---|---|
| CNPJ error | Digit transposition in supplier CNPJ |
| Date shift | Payment 10–30 days before/after invoice date |
| Value discrepancy | Partial payment or rounding difference >2% |
| Ghost record | Invoice with no corresponding payment (or vice versa) |

---

## 5. ETL Pipeline

### 5.1 Load (`etl/loader.py`)
- Reads `.xlsx` with `pandas` + `openpyxl` → `df_pagamentos`
- Parses each NFS-e `.xml` with `lxml`, flattens nested ABRASF structure → `df_nfse`

### 5.2 Clean & Normalize (`etl/cleaner.py`)
- **CNPJ:** strip formatting, zero-pad to 14 digits, validate check digits; invalid records flagged (not dropped)
- **Dates:** parse to `datetime`, enforce ISO 8601; resolve `Competencia` vs `DataEmissao` ambiguity
- **Values:** cast to `Decimal` to avoid float rounding errors; strip currency symbols
- **Deduplication:** flag duplicate `id_pagamento` or `Numero` NFS-e within each dataset

### 5.3 Label Assignment (`etl/labeler.py`)
- Fuzzy left-join on CNPJ (always exact), then apply date and value tolerance windows from config
- **Exact scenario labels:** `0 = não conciliado`, `1 = conciliado`
- **Fuzzy scenario labels:** `0 = não conciliado`, `1 = parcialmente conciliado`, `2 = conciliado`
- Output: `data/processed/<scenario>_reconciliado.csv` with `pag_*` and `nfse_*` prefixed columns, plus `label` and `scenario`

---

## 6. Feature Engineering

Features computed from the unified DataFrame before model training:

| Feature | Derivation |
|---|---|
| `delta_days` | `abs(data_pagamento - DataEmissao)` in days |
| `delta_valor_pct` | `abs(valor_pago - ValorServicos) / ValorServicos` |
| `valor_pago` | Raw payment amount |
| `ValorIss` | ISS tax from NFS-e |
| `Aliquota` | ISS rate |
| `descricao_similarity` | Cosine similarity between `descricao` and `Discriminacao` (TF-IDF) |
| `is_mesmo_municipio` | Boolean: CodigoMunicipio matches company municipality |
| `cnpj_match` | Exact boolean (sanity check; always 1 post-join) |

---

## 7. Modeling

All three algorithms are trained via a uniform interface in `models/trainer.py`.

**Training protocol:**
- 80/20 stratified train/test split (stratified by `label`)
- 5-fold stratified cross-validation on training set
- `GridSearchCV` with `f1_macro` scoring
- No data leakage: split before scaling

**Algorithm specifics:**

| Algorithm | Preprocessing | Key config |
|---|---|---|
| Logistic Regression | `StandardScaler` | L2 regularization |
| SVM | `StandardScaler` | RBF kernel, `class_weight='balanced'` |
| Random Forest | None | 200 trees, `class_weight='balanced_subsample'` |

Trained models serialized with `joblib` to `data/results/<scenario>/models/`.

---

## 8. Evaluation

All metrics computed in `models/evaluator.py`, visualizations in `notebooks/04_avaliacao.ipynb`.

**Metrics per algorithm per scenario:**

| Metric | Notes |
|---|---|
| Accuracy | Baseline; expected by academic audience |
| Precision / Recall / F1 (macro) | Primary metric; handles class imbalance |
| ROC-AUC | Exact scenario only (binary) |
| CV mean ± std | Demonstrates stability across folds |
| Confusion matrix | Per-algorithm, per-scenario |

**Artifacts saved to `data/results/<scenario>/`:**

```
metrics_summary.csv
confusion_matrix_rf.png
confusion_matrix_svm.png
confusion_matrix_lr.png
roc_curve.png              # exact scenario only
feature_importance_rf.png
```

**Cross-scenario comparison table:** `notebooks/04_avaliacao.ipynb` produces a single summary table (all algorithms × both scenarios) as the core exhibit for the results chapter.

**Error analysis:** final notebook section flags highest-confidence wrong predictions for qualitative discussion in the conclusion.

---

## 9. Notebooks (Paper Narrative)

| Notebook | Chapter alignment |
|---|---|
| `01_simulacao.ipynb` | Data generation methodology |
| `02_etl.ipynb` | Exploratory analysis, normalization, cleaning |
| `03_modelagem.ipynb` | Algorithm training and cross-validation |
| `04_avaliacao.ipynb` | Results, comparison, error analysis |

Each notebook imports from `src/reconciliacao/` and adds narrative text, inline charts, and interpretation for the academic reader.

---

## 10. Reproducibility

Running the full pipeline for both scenarios:

```bash
python run_pipeline.py --scenario exact
python run_pipeline.py --scenario fuzzy
```

All outputs are deterministic given the same `config.yaml` and `random_seed`.
