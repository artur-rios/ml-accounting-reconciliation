# Financial Reconciliation Automation with Machine Learning

> [!NOTE]
> This is the real, open-source code of ml-accounting-reconciliation. Its home is Forgejo, at
> https://git.techcraftsmen.net/artur-rios/ml-accounting-reconciliation; the copy on GitHub is a read-only mirror, for viewing only, and
> its issues and pull requests are not followed. If you want to contribute, email me at
> arturdev@duck.com.

Final project (TCC) for the MBA in Software Engineering at USP/ESALQ. Licensed under the [Artur Rios Attribution License](LICENSE). Automates financial reconciliation using supervised machine learning by simulating two real-world data sources (supplier payment records in Excel and electronic service invoices / NFS-e in XML), running them through an ETL pipeline to produce labeled datasets, then training and evaluating three classifiers.

## Research Question

Which algorithm (Random Forest, SVM, or Logistic Regression) best automates financial reconciliation, and how much of the observed performance comes from the algorithm rather than from the representation given to the data?

## The three experiments, and which one answers the question

The repository contains three experiments, run in this order. Reading them out of order is misleading, because the first experiment's headline result is a diagnostic rather than a finding.

| # | Runner | What it establishes |
| --- | --- | --- |
| 1 | `run_pipeline.py` | The main pipeline. Random Forest reaches accuracy 1.0000 — because the label is a deterministic threshold over two of its own features. It cannot answer the research question. |
| 2 | `run_ablation.py` | Quantifies that circularity, and shows the 11–13 p.p. F1-macro gap between algorithms was entirely an artifact of sentinel encoding. |
| 3 | `run_comparison.py` | The corrective experiment that does answer the question: the label comes from generator ground truth, a leak guard fails the run if any single feature separates the classes, and selection is by exception recall under a precision floor. |

## Findings

Under experiment 3, averaged over 10 independent datasets of 7,500 payments each (`data/results/comparison/decision_summary.csv`):

| Algorithm | Exception recall | Exception precision | Referral rate | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 0.8216 ± 0.0169 | 0.8931 | 27.6% | 0.9356 |
| SVM (RBF) | 0.8002 ± 0.0191 | 0.8996 | 26.7% | 0.9258 |
| Logistic Regression | 0.7698 ± 0.0251 | 0.9030 | 25.6% | 0.9153 |

Random Forest detects 1.44 p.p. more divergences than the SVM and 4.56 p.p. more than Logistic Regression. Both gaps are significant after Holm correction (p = 0.0059), the ranking holds under the threshold-free PR-AUC, and the three precisions are statistically equivalent (Friedman p = 0.1225). The margins are narrow, though: interpretability and cost are legitimate reasons to pick Logistic Regression instead. The broader answer is the one in the monograph's title — data representation and label provenance weighed more than the choice of algorithm.

## Documentation

| Document | Contents |
| --- | --- |
| Monograph — [Português](docs/monography/monografia.md) · [English](docs/monography/monography.md) | The full paper in Markdown. The original [`.docx`](docs/monography/TCC%20-%20Artur%20Rios%20da%20Silva%20Oliveira.docx) and [`.pdf`](docs/monography/TCC%20-%20Artur%20Rios%20da%20Silva%20Oliveira.pdf) remain the reference. |
| [`docs/discussao-limitacoes-contribuicoes.md`](docs/discussao-limitacoes-contribuicoes.md) | Results, discussion, limitations and reproduction notes, in Portuguese. The most detailed account of every finding. |
| [`docs/glossary.md`](docs/glossary.md) | Terms used across the code, configuration and write-up — ablation, leakage, sentinel encoding, exception recall, the Brazilian tax withholdings, and others — in English and Portuguese, with references. |
| [`docs/presentation/`](docs/presentation/) | Defense slide deck (`.pptx`, `.pdf`) and its script, [`roteiro.md`](docs/presentation/roteiro.md). |

## Scenarios (experiments 1 and 2)

| Scenario | Date tolerance | Value tolerance | Labels |
| --- | --- | --- | --- |
| `exact` | 0 days | 0% | 2 (conciliated / not conciliated) |
| `fuzzy` | ±5 days | ±2% | 3 (conciliated / partially / not conciliated) |

In `fuzzy`, a record is *partially* conciliated when it falls within 4× the date tolerance or 5× the value tolerance. Because the generator's divergences (6–30 days, 3–20%) all fall outside both tolerances, the `fuzzy` scenario does not actually exercise tolerance — see the discussion document, §2.4.

## Project Structure

```text
├── config.yaml                  # All experiment parameters
├── run_pipeline.py              # Experiment 1 — end-to-end CLI runner
├── run_ablation.py              # Experiment 2 — feature-representation ablation
├── run_comparison.py            # Experiment 3 — leak-free algorithm comparison
├── run_figures.py               # Regenerates the monograph's Figures 6 and 7
├── pyproject.toml
├── data/                        # Raw and processed data are generated at runtime
│   ├── raw/                     # Simulated .xlsx and .xml           (not versioned)
│   ├── processed/               # Post-ETL labeled CSVs              (not versioned)
│   └── results/                 # Metrics, plots and reports          (versioned)
│       ├── exact/               # Experiment 1, exact scenario
│       ├── fuzzy/               # Experiment 1, fuzzy scenario
│       ├── ablation/            # Experiment 2
│       ├── comparison/          # Experiment 3
│       └── figuras/             # Figures embedded in the monograph
├── src/reconciliacao/
│   ├── simulation/
│   │   ├── excel_generator.py   # Payment ledger (experiments 1-2)
│   │   ├── xml_generator.py     # NFS-e / ABRASF XML (experiments 1-2)
│   │   ├── retentions.py        # Brazilian withholding arithmetic (experiment 3)
│   │   └── truth_generator.py   # Ground-truth dataset (experiment 3)
│   ├── etl/
│   │   ├── loader.py, cleaner.py
│   │   ├── labeler.py           # Threshold rule (experiments 1-2)
│   │   └── truth_labeler.py     # Label from ground truth (experiment 3)
│   ├── models/
│   │   ├── features.py          # 7 features (experiments 1-2)
│   │   ├── features_v2.py       # 14 features (experiment 3)
│   │   ├── trainer.py, evaluator.py
│   │   ├── comparison_trainer.py, comparison_stats.py
│   │   ├── decision.py          # Threshold under a precision floor
│   │   └── leak_guard.py        # Fails the run on single-feature separation
│   ├── comparison.py            # Experiment 3 orchestration
│   └── utils/                   # Config loader, CNPJ validation
├── notebooks/                   # Narrative walkthrough of experiment 1 only, in Portuguese
├── docs/
│   ├── monography/              # Monograph (.docx, .pdf, Markdown) and earlier deliverables
│   ├── presentation/            # Defense slides and script
│   ├── academic-standards/      # USP/ESALQ manual and template
│   ├── superpowers/             # Design specs and implementation plans
│   ├── discussao-limitacoes-contribuicoes.md
│   └── glossary.md
└── tests/                       # 140 tests, mirroring src/
```

## Setup

Requires Python 3.10 or later. The versioned results were produced with Python 3.14, scikit-learn 1.9.0 and pandas 3.0.5.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -e ".[dev]"
```

## Running the experiments

Experiments 1 and 2 must run in order: the ablation reads the labeled CSVs the pipeline writes to `data/processed/`, and fails with `FileNotFoundError` if they are absent. Experiment 3 is independent of both.

```bash
python run_pipeline.py --scenario exact
```

```bash
python run_pipeline.py --scenario fuzzy
```

```bash
python run_ablation.py
```

```bash
python run_comparison.py
```

Each `run_pipeline.py` run:

1. Simulates 7,500 payment records and matching NFS-e XML files
2. Runs ETL (load → clean → label)
3. Engineers features and trains RF, SVM, and LR via GridSearchCV (5-fold CV, F1-macro)
4. Evaluates and saves metrics, confusion matrices, ROC curve, and feature importance plots

Use `python run_comparison.py --seeds 2` for a quick run of experiment 3 instead of the full 10 seeds; `--output` sends its artifacts somewhere other than `data/results/comparison/`. Both `run_pipeline.py` and `run_comparison.py` accept `--config` to point at another configuration file; `run_ablation.py` always reads `config.yaml`.

After experiments 2 and 3, regenerate the monograph's two charts from their artifacts:

```bash
python run_figures.py
```

It writes to `data/results/figuras/`. Adding `--embed` also replaces the two images inside the monograph `.docx` in place.

### Reproducibility

All three experiments are deterministic given the same `config.yaml`: every source of randomness is seeded, and the simulated date window is anchored to `simulation.reference_date` rather than to the wall clock.

This was not always true, and the write-up documents why. Until it was corrected, the payment generator assigned supplier registration numbers from `list(set(...))`, whose iteration order depends on the per-process string hash seed, so each run produced a different payment-to-CNPJ assignment while every aggregate stayed identical. `tests/test_simulation/test_excel_generator.py` now pins the property by generating in two subprocesses under different `PYTHONHASHSEED` values — a property no single-process test can observe.

## Configuration

All parameters are in `config.yaml` — no hardcoded values in source modules. Experiments 1 and 2 read the `simulation`, `scenarios` and `models` sections:

```yaml
simulation:
  n_records: 7500
  conciliation_rate: 0.70   # 70% conciliated, 30% not
  random_seed: 42
  reference_date: 2026-06-14
```

Experiment 3 reads the `comparison` section. Its main settings:

| Key | Value | Meaning |
| --- | --- | --- |
| `n_seeds` | 10 | Independent datasets generated and compared |
| `min_precision` | 0.90 | Precision floor on the exception class |
| `precision_margin` | 0.0 | Extra margin above the floor on validation; `0.0` reproduces the published results |
| `leak_guard_max_stump_accuracy` | 0.95 | A single feature above this accuracy fails the run |
| `hard_negative_rate` | 0.30 | Share of non-pairs whose value gap matches a legal withholding |
| `csrf_threshold_brl` | 5000.0 | CSRF waiver threshold; no longer current law, kept for the interaction it creates (limitation 25) |

## Evaluation Artifacts

Experiment 1 writes to `data/results/<scenario>/`:

```text
metrics_summary.csv
confusion_matrix_random_forest.png
confusion_matrix_svm.png
confusion_matrix_logistic_regression.png
roc_curve.png                    # exact scenario only (binary)
feature_importance_rf.png
models/
  random_forest.joblib
  svm.joblib
  logistic_regression.joblib
```

Experiment 2 writes to `data/results/ablation/`:

```text
ablation_summary.csv                     # 24 runs: 4 configurations x 2 scenarios x 3 algorithms
<config>/<scenario>/cm_<algorithm>.txt   # confusion matrices
<config>/<scenario>/models/*.joblib
```

Experiment 3 writes to `data/results/comparison/`:

```text
per_seed_metrics.csv             # 10 seeds x 3 algorithms
decision_summary.csv             # final table, with n_seeds_applicable
friedman.csv                     # omnibus test, before any pairwise comparison
wilcoxon.csv                     # pairwise post-hoc on exception recall
wilcoxon_secundarias.csv         # the same, on PR-AUC and on exception precision
confusion_matrices.csv           # exception-class cells, summed over seeds
leak_guard_report.csv            # stump accuracy per feature, per seed
pr_curves.png, recall_boxplot.png
```

Trained models (`*.joblib`) are not versioned; everything else under `data/results/` is.

## Features

Experiments 1 and 2 use seven features:

| Feature | Derivation |
| --- | --- |
| `delta_days` | `abs(data_pagamento − nfse_data_emissao)` in days; `9999` when no invoice joined |
| `delta_valor_pct` | `abs(valor_pago − nfse_valor_servicos) / nfse_valor_servicos × 100`; `100` when no invoice joined |
| `valor_pago` | Raw payment amount |
| `nfse_valor_iss` | ISS tax from the NFS-e; `0` when no invoice joined |
| `nfse_aliquota` | ISS rate; `0` when no invoice joined |
| `descricao_similarity` | TF-IDF cosine similarity between payment description and NFS-e discriminação |
| `cnpj_match` | Boolean: supplier CNPJ present (constant in practice — see the discussion document) |

The first two are the quantities the labeler thresholds to produce the target, which is why experiment 1 cannot compare algorithms. Experiment 3 uses a different, 14-feature matrix built by `features_v2.py`, none of which participates in defining the label.

## Algorithm Comparison (leak-free)

In experiment 3 the label comes from generator ground truth rather than from a threshold rule, so it measures the algorithms and not the labeling. True pairs are derived from their invoices by a legal transformation — Brazilian withholding taxes and a per-supplier payment term — so a value gap no longer implies a mismatch. A leak guard fails the run if any single feature separates the classes. Selection is by exception recall under precision >= 0.90, with the threshold chosen on validation, and algorithms are compared with a Friedman omnibus test followed by paired Wilcoxon post-hoc tests with Holm correction. The results are under [Findings](#findings).

## Tests

The suite has 140 tests. Three of them are marked `integration`: they run the pipeline and experiment 3 end to end on small configurations, writing only to temporary directories.

```bash
pytest
```

```bash
pytest -m "not integration"
```

```bash
pytest --cov=src/reconciliacao --cov-report=term-missing
```

## Dependencies

- **Core:** pandas, numpy, scikit-learn, scipy
- **Data I/O:** openpyxl, lxml, pyyaml, joblib
- **Simulation:** Faker (pt_BR locale)
- **Visualization:** matplotlib, seaborn
- **Dev:** pytest, pytest-cov, jupyter, ipykernel
