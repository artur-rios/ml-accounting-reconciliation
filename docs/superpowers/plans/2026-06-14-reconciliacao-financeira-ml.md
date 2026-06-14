# Reconciliação Financeira com ML — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a config-driven ML pipeline that simulates supplier payment records and NFS-e invoices, applies ETL, trains three classifiers (Random Forest, SVM, Logistic Regression), and evaluates them across exact and fuzzy reconciliation scenarios.

**Architecture:** A `src/reconciliacao` Python package provides simulation, ETL, modeling, and evaluation modules. `run_pipeline.py` ties them together via `config.yaml`. Four Jupyter notebooks import from the package to serve as the paper narrative.

**Tech Stack:** Python 3.10+, pandas 2.0, scikit-learn 1.3, Faker (pt_BR), lxml, openpyxl, PyYAML, joblib, matplotlib, seaborn, pytest

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package definition and dev dependencies |
| `config.yaml` | All experiment parameters |
| `run_pipeline.py` | CLI: `--scenario exact\|fuzzy` |
| `src/reconciliacao/__init__.py` | Package root |
| `src/reconciliacao/utils/__init__.py` | Utils subpackage |
| `src/reconciliacao/utils/config.py` | YAML config loader |
| `src/reconciliacao/utils/cnpj.py` | CNPJ generation and validation |
| `src/reconciliacao/simulation/__init__.py` | Simulation subpackage |
| `src/reconciliacao/simulation/excel_generator.py` | Generate payment records → `.xlsx` |
| `src/reconciliacao/simulation/xml_generator.py` | Generate NFS-e records → `.xml` (ABRASF) |
| `src/reconciliacao/etl/__init__.py` | ETL subpackage |
| `src/reconciliacao/etl/loader.py` | Read Excel + XML → DataFrames |
| `src/reconciliacao/etl/cleaner.py` | Normalize CNPJ, dates, values |
| `src/reconciliacao/etl/labeler.py` | Join records, compute deltas, assign labels |
| `src/reconciliacao/models/__init__.py` | Models subpackage |
| `src/reconciliacao/models/features.py` | Compute ML feature matrix from merged DF |
| `src/reconciliacao/models/trainer.py` | Train RF/SVM/LR with GridSearchCV |
| `src/reconciliacao/models/evaluator.py` | Metrics, confusion matrices, artifact export |
| `tests/conftest.py` | Shared pytest fixtures |
| `tests/test_utils/test_config.py` | Config loader tests |
| `tests/test_utils/test_cnpj.py` | CNPJ utility tests |
| `tests/test_simulation/test_excel_generator.py` | Excel generator tests |
| `tests/test_simulation/test_xml_generator.py` | XML generator tests |
| `tests/test_etl/test_loader.py` | Loader tests |
| `tests/test_etl/test_cleaner.py` | Cleaner tests |
| `tests/test_etl/test_labeler.py` | Labeler tests |
| `tests/test_models/test_features.py` | Feature engineering tests |
| `tests/test_models/test_trainer.py` | Trainer tests |
| `tests/test_models/test_evaluator.py` | Evaluator tests |
| `notebooks/01_simulacao.ipynb` | Simulation narrative |
| `notebooks/02_etl.ipynb` | ETL and EDA narrative |
| `notebooks/03_modelagem.ipynb` | Modeling and cross-validation narrative |
| `notebooks/04_avaliacao.ipynb` | Results and comparison narrative |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `config.yaml`
- Create: all `__init__.py` files and empty module stubs
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "reconciliacao"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "openpyxl>=3.1",
    "lxml>=4.9",
    "faker>=20.0",
    "pyyaml>=6.0",
    "joblib>=1.3",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "scipy>=1.11",
]

[project.optional-dependencies]
dev = ["pytest>=7.4", "pytest-cov>=4.1", "jupyter>=1.0", "ipykernel>=6.0"]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create `config.yaml`**

```yaml
simulation:
  n_records: 7500
  conciliation_rate: 0.70
  random_seed: 42

scenarios:
  exact:
    date_tolerance_days: 0
    value_tolerance_pct: 0.0
    n_classes: 2
  fuzzy:
    date_tolerance_days: 5
    value_tolerance_pct: 2.0
    n_classes: 3

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

- [ ] **Step 3: Create directory structure and empty `__init__.py` files**

```bash
mkdir -p src/reconciliacao/utils
mkdir -p src/reconciliacao/simulation
mkdir -p src/reconciliacao/etl
mkdir -p src/reconciliacao/models
mkdir -p tests/test_utils
mkdir -p tests/test_simulation
mkdir -p tests/test_etl
mkdir -p tests/test_models
mkdir -p notebooks
mkdir -p data/raw data/processed data/results

touch src/reconciliacao/__init__.py
touch src/reconciliacao/utils/__init__.py
touch src/reconciliacao/simulation/__init__.py
touch src/reconciliacao/etl/__init__.py
touch src/reconciliacao/models/__init__.py
touch tests/__init__.py
touch tests/test_utils/__init__.py
touch tests/test_simulation/__init__.py
touch tests/test_etl/__init__.py
touch tests/test_models/__init__.py
```

- [ ] **Step 4: Create `tests/conftest.py`**

```python
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
            "logistic_regression": {"penalty": "l2", "class_weight": "balanced"},
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
```

- [ ] **Step 5: Install the package in dev mode**

```bash
pip install -e ".[dev]"
```

Expected output: `Successfully installed reconciliacao-0.1.0`

- [ ] **Step 6: Verify pytest discovers the test structure**

```bash
pytest --collect-only
```

Expected: `no tests ran` (no test files yet, but no collection errors)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml config.yaml src/ tests/ notebooks/ data/.gitkeep
git commit -m "chore: project scaffolding — package structure and config"
```

---

## Task 2: Config Loader

**Files:**
- Create: `src/reconciliacao/utils/config.py`
- Create: `tests/test_utils/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_utils/test_config.py
import pytest
from pathlib import Path
from reconciliacao.utils.config import load_config


def test_load_config_returns_dict(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("simulation:\n  n_records: 100\n  random_seed: 42\n")
    cfg = load_config(cfg_file)
    assert isinstance(cfg, dict)
    assert cfg["simulation"]["n_records"] == 100


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")


def test_load_config_default_path_reads_project_config():
    cfg = load_config("config.yaml")
    assert cfg["simulation"]["n_records"] == 7500
    assert "exact" in cfg["scenarios"]
    assert "fuzzy" in cfg["scenarios"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_utils/test_config.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` for `reconciliacao.utils.config`

- [ ] **Step 3: Write `src/reconciliacao/utils/config.py`**

```python
from pathlib import Path
import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_utils/test_config.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/reconciliacao/utils/config.py tests/test_utils/test_config.py
git commit -m "feat: config loader — YAML-based experiment parameter management"
```

---

## Task 3: CNPJ Utilities

**Files:**
- Create: `src/reconciliacao/utils/cnpj.py`
- Create: `tests/test_utils/test_cnpj.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_utils/test_cnpj.py
import random
from reconciliacao.utils.cnpj import generate_cnpj, validate_cnpj, normalize_cnpj


def test_generate_cnpj_has_14_digits():
    assert len(generate_cnpj()) == 14


def test_generate_cnpj_produces_valid_cnpj():
    rng = random.Random(42)
    for _ in range(100):
        assert validate_cnpj(generate_cnpj(rng))


def test_generate_cnpj_is_reproducible():
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    assert generate_cnpj(rng1) == generate_cnpj(rng2)


def test_validate_cnpj_known_valid():
    # 11.222.333/0001-81 — check digits: d1=8, d2=1
    assert validate_cnpj("11222333000181")


def test_validate_cnpj_known_invalid():
    assert not validate_cnpj("11222333000182")


def test_validate_cnpj_with_formatting():
    assert validate_cnpj("11.222.333/0001-81")


def test_validate_cnpj_wrong_length():
    assert not validate_cnpj("1122233300018")


def test_normalize_cnpj_strips_formatting():
    assert normalize_cnpj("11.222.333/0001-81") == "11222333000181"


def test_normalize_cnpj_pads_short_string():
    # Already 14 digits but with leading zero stripped
    assert normalize_cnpj("1222333000181") == "01222333000181"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_utils/test_cnpj.py -v
```

Expected: `ImportError` for `reconciliacao.utils.cnpj`

- [ ] **Step 3: Write `src/reconciliacao/utils/cnpj.py`**

```python
import random as _random


def _calc_check_digit(digits: list[int], weights: list[int]) -> int:
    s = sum(d * w for d, w in zip(digits, weights))
    r = s % 11
    return 0 if r < 2 else 11 - r


def generate_cnpj(rng: _random.Random | None = None) -> str:
    r = rng or _random.Random()
    digits = [r.randint(0, 9) for _ in range(12)]
    w1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    w2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    d1 = _calc_check_digit(digits, w1)
    d2 = _calc_check_digit(digits + [d1], w2)
    return "".join(map(str, digits + [d1, d2]))


def validate_cnpj(cnpj: str) -> bool:
    digits_str = "".join(c for c in cnpj if c.isdigit())
    if len(digits_str) != 14:
        return False
    digits = [int(c) for c in digits_str]
    w1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    w2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    d1 = _calc_check_digit(digits[:12], w1)
    d2 = _calc_check_digit(digits[:12] + [d1], w2)
    return digits[12] == d1 and digits[13] == d2


def normalize_cnpj(cnpj: str) -> str:
    return "".join(c for c in cnpj if c.isdigit()).zfill(14)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_utils/test_cnpj.py -v
```

Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add src/reconciliacao/utils/cnpj.py tests/test_utils/test_cnpj.py
git commit -m "feat: CNPJ utilities — generate, validate, and normalize Brazilian company IDs"
```

---

## Task 4: Excel Generator

**Files:**
- Create: `src/reconciliacao/simulation/excel_generator.py`
- Create: `tests/test_simulation/test_excel_generator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_simulation/test_excel_generator.py
import pandas as pd
from reconciliacao.simulation.excel_generator import generate_payment_records, write_excel


def test_generate_payment_records_count():
    df = generate_payment_records(n=100, seed=42)
    assert len(df) == 100


def test_generate_payment_records_schema():
    df = generate_payment_records(n=10, seed=42)
    assert set(df.columns) == {
        "id_pagamento", "cnpj_fornecedor", "data_pagamento",
        "valor_pago", "descricao", "centro_custo",
    }


def test_generate_payment_records_unique_cnpj():
    df = generate_payment_records(n=100, seed=42)
    assert df["cnpj_fornecedor"].nunique() == 100


def test_generate_payment_records_cnpj_length():
    df = generate_payment_records(n=50, seed=42)
    assert df["cnpj_fornecedor"].str.len().eq(14).all()


def test_generate_payment_records_reproducible():
    df1 = generate_payment_records(n=20, seed=42)
    df2 = generate_payment_records(n=20, seed=42)
    assert df1.equals(df2)


def test_generate_payment_records_positive_values():
    df = generate_payment_records(n=50, seed=42)
    assert (df["valor_pago"] > 0).all()


def test_write_excel_creates_readable_file(tmp_path):
    df = generate_payment_records(n=10, seed=42)
    out = tmp_path / "pagamentos.xlsx"
    write_excel(df, out)
    assert out.exists()
    loaded = pd.read_excel(out, engine="openpyxl")
    assert len(loaded) == 10
    assert "id_pagamento" in loaded.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_simulation/test_excel_generator.py -v
```

Expected: `ImportError` for `reconciliacao.simulation.excel_generator`

- [ ] **Step 3: Write `src/reconciliacao/simulation/excel_generator.py`**

```python
import random
from datetime import date
from pathlib import Path

import pandas as pd
from faker import Faker

from reconciliacao.utils.cnpj import generate_cnpj


def generate_payment_records(n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    Faker.seed(seed)
    fake = Faker("pt_BR")

    # Generate N unique valid CNPJs upfront to guarantee 1:1 join later
    cnpjs: set[str] = set()
    while len(cnpjs) < n:
        cnpjs.add(generate_cnpj(rng))
    cnpj_list = list(cnpjs)

    records = []
    for i, cnpj in enumerate(cnpj_list):
        records.append({
            "id_pagamento": f"PAG-{i + 1:06d}",
            "cnpj_fornecedor": cnpj,
            "data_pagamento": fake.date_between(start_date="-1y", end_date="today"),
            "valor_pago": round(rng.uniform(500.0, 50_000.0), 2),
            "descricao": fake.bs(),
            "centro_custo": f"CC-{rng.randint(100, 999)}",
        })
    return pd.DataFrame(records)


def write_excel(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False, engine="openpyxl")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_simulation/test_excel_generator.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/reconciliacao/simulation/excel_generator.py tests/test_simulation/test_excel_generator.py
git commit -m "feat: Excel generator — simulate supplier payment records with unique CNPJs"
```

---

## Task 5: XML Generator (NFS-e ABRASF)

**Files:**
- Create: `src/reconciliacao/simulation/xml_generator.py`
- Create: `tests/test_simulation/test_xml_generator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_simulation/test_xml_generator.py
import datetime
from pathlib import Path
import pandas as pd
import pytest
from lxml import etree
from reconciliacao.simulation.xml_generator import generate_nfse, write_xml

NS = "http://www.abrasf.org.br/nfse.xsd"


@pytest.fixture
def sample_pagamentos():
    return pd.DataFrame({
        "id_pagamento": [f"PAG-{i:06d}" for i in range(1, 11)],
        "cnpj_fornecedor": ["11222333000181"] * 5 + ["44555666000195"] * 5,
        "data_pagamento": [datetime.date(2024, 1, 15)] * 10,
        "valor_pago": [1000.00] * 10,
        "descricao": ["consultoria"] * 10,
        "centro_custo": ["CC-001"] * 10,
    })


def test_generate_nfse_count(sample_pagamentos):
    df = generate_nfse(sample_pagamentos, conciliation_rate=0.70, seed=42)
    assert len(df) == len(sample_pagamentos)


def test_generate_nfse_schema(sample_pagamentos):
    df = generate_nfse(sample_pagamentos, conciliation_rate=0.70, seed=42)
    expected_cols = {
        "nfse_numero", "nfse_codigo_verificacao", "nfse_data_emissao",
        "nfse_competencia", "nfse_cnpj_prestador", "nfse_razao_social_prestador",
        "nfse_cnpj_tomador", "nfse_razao_social_tomador", "nfse_valor_servicos",
        "nfse_valor_iss", "nfse_aliquota", "nfse_valor_liquido",
        "nfse_item_lista_servico", "nfse_discriminacao", "nfse_codigo_municipio",
    }
    assert expected_cols.issubset(set(df.columns))


def test_generate_nfse_reproducible(sample_pagamentos):
    df1 = generate_nfse(sample_pagamentos, conciliation_rate=0.70, seed=42)
    df2 = generate_nfse(sample_pagamentos, conciliation_rate=0.70, seed=42)
    assert df1.equals(df2)


def test_write_xml_creates_valid_abrasf_file(sample_pagamentos, tmp_path):
    df = generate_nfse(sample_pagamentos, conciliation_rate=0.70, seed=42)
    out = tmp_path / "nfse.xml"
    write_xml(df, out)
    assert out.exists()
    tree = etree.parse(str(out))
    ns = {"ns": NS}
    comps = tree.xpath("//ns:CompNfse", namespaces=ns)
    assert len(comps) == len(sample_pagamentos)


def test_write_xml_contains_cnpj(sample_pagamentos, tmp_path):
    df = generate_nfse(sample_pagamentos, conciliation_rate=1.0, seed=42)
    out = tmp_path / "nfse.xml"
    write_xml(df, out)
    tree = etree.parse(str(out))
    ns = {"ns": NS}
    cnpjs = tree.xpath("//ns:PrestadorServico//ns:Cnpj/text()", namespaces=ns)
    assert len(cnpjs) == len(sample_pagamentos)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_simulation/test_xml_generator.py -v
```

Expected: `ImportError` for `reconciliacao.simulation.xml_generator`

- [ ] **Step 3: Write `src/reconciliacao/simulation/xml_generator.py`**

```python
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker
from lxml import etree

from reconciliacao.utils.cnpj import generate_cnpj

_ABRASF_NS = "http://www.abrasf.org.br/nfse.xsd"
_LC116_CODES = ["1.01", "1.02", "1.03", "1.04", "1.05", "7.01", "7.02", "14.01"]
_MUNICIPIOS = ["3550308", "3304557", "4106902", "2304400", "5300108"]


def _sub(parent: etree._Element, tag: str, text: str | None = None) -> etree._Element:
    el = etree.SubElement(parent, tag)
    if text is not None:
        el.text = str(text)
    return el


def generate_nfse(df_pag: pd.DataFrame, conciliation_rate: float, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    Faker.seed(seed)
    fake = Faker("pt_BR")

    n = len(df_pag)
    conciliated_indices = set(rng.sample(range(n), k=int(n * conciliation_rate)))
    mismatch_types = ["cnpj_error", "date_shift", "value_discrepancy", "ghost"]

    records = []
    for i, row in enumerate(df_pag.itertuples(index=False)):
        pag_date: date = row.data_pagamento
        pag_valor: float = row.valor_pago
        pag_cnpj: str = row.cnpj_fornecedor

        if i in conciliated_indices:
            cnpj = pag_cnpj
            emit_date = pag_date
            valor = pag_valor
        else:
            mismatch = rng.choice(mismatch_types)
            if mismatch == "cnpj_error":
                # Flip one of the first 12 digits to produce an invalid CNPJ
                digits = list(pag_cnpj)
                pos = rng.randint(0, 11)
                digits[pos] = str((int(digits[pos]) + rng.randint(1, 9)) % 10)
                cnpj = "".join(digits)
                emit_date = pag_date
                valor = pag_valor
            elif mismatch == "date_shift":
                cnpj = pag_cnpj
                shift = rng.randint(6, 30) * rng.choice([-1, 1])
                emit_date = pag_date + timedelta(days=shift)
                valor = pag_valor
            elif mismatch == "value_discrepancy":
                cnpj = pag_cnpj
                emit_date = pag_date
                factor = 1 + rng.uniform(0.03, 0.20) * rng.choice([-1, 1])
                valor = round(pag_valor * factor, 2)
            else:  # ghost — completely unrelated invoice
                cnpj = generate_cnpj(rng)
                emit_date = fake.date_between(start_date="-1y", end_date="today")
                valor = round(rng.uniform(500.0, 50_000.0), 2)

        aliquota = round(rng.uniform(2.0, 5.0), 2)
        valor_iss = round(valor * aliquota / 100, 2)

        records.append({
            "nfse_numero": f"{i + 1:06d}",
            "nfse_codigo_verificacao": "".join(
                rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=8)
            ),
            "nfse_data_emissao": emit_date.isoformat() + "T00:00:00",
            "nfse_competencia": emit_date.replace(day=1).isoformat() + "T00:00:00",
            "nfse_cnpj_prestador": cnpj,
            "nfse_razao_social_prestador": fake.company(),
            "nfse_cnpj_tomador": generate_cnpj(rng),
            "nfse_razao_social_tomador": fake.company(),
            "nfse_valor_servicos": valor,
            "nfse_valor_iss": valor_iss,
            "nfse_aliquota": aliquota,
            "nfse_valor_liquido": round(valor - valor_iss, 2),
            "nfse_item_lista_servico": rng.choice(_LC116_CODES),
            "nfse_discriminacao": fake.bs(),
            "nfse_codigo_municipio": rng.choice(_MUNICIPIOS),
        })

    return pd.DataFrame(records)


def write_xml(df: pd.DataFrame, path: str | Path) -> None:
    nsmap = {None: _ABRASF_NS}
    root = etree.Element("ListaNfse", nsmap=nsmap)

    for _, row in df.iterrows():
        comp = _sub(root, "CompNfse")
        nfse = _sub(comp, "Nfse")
        inf = _sub(nfse, "InfNfse")

        _sub(inf, "Numero", row["nfse_numero"])
        _sub(inf, "CodigoVerificacao", row["nfse_codigo_verificacao"])
        _sub(inf, "DataEmissao", row["nfse_data_emissao"])
        _sub(inf, "Competencia", row["nfse_competencia"])

        prest = _sub(inf, "PrestadorServico")
        id_prest = _sub(prest, "IdentificacaoPrestador")
        _sub(_sub(id_prest, "CpfCnpj"), "Cnpj", row["nfse_cnpj_prestador"])
        _sub(id_prest, "InscricaoMunicipal", "000001")
        _sub(prest, "RazaoSocial", row["nfse_razao_social_prestador"])

        tom = _sub(inf, "TomadorServico")
        id_tom = _sub(tom, "IdentificacaoTomador")
        _sub(_sub(id_tom, "CpfCnpj"), "Cnpj", row["nfse_cnpj_tomador"])
        _sub(tom, "RazaoSocial", row["nfse_razao_social_tomador"])

        servico = _sub(inf, "Servico")
        valores = _sub(servico, "Valores")
        _sub(valores, "ValorServicos", f"{row['nfse_valor_servicos']:.2f}")
        _sub(valores, "ValorIss", f"{row['nfse_valor_iss']:.2f}")
        _sub(valores, "Aliquota", f"{row['nfse_aliquota']:.2f}")
        _sub(valores, "ValorLiquidoNfse", f"{row['nfse_valor_liquido']:.2f}")
        _sub(servico, "ItemListaServico", row["nfse_item_lista_servico"])
        _sub(servico, "Discriminacao", row["nfse_discriminacao"])
        _sub(servico, "CodigoMunicipio", row["nfse_codigo_municipio"])

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    etree.ElementTree(root).write(
        str(path), xml_declaration=True, encoding="UTF-8", pretty_print=True
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_simulation/test_xml_generator.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/reconciliacao/simulation/xml_generator.py tests/test_simulation/test_xml_generator.py
git commit -m "feat: XML generator — simulate NFS-e invoices in ABRASF format with mismatch injection"
```

---

## Task 6: ETL Loader

**Files:**
- Create: `src/reconciliacao/etl/loader.py`
- Create: `tests/test_etl/test_loader.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_etl/test_loader.py
import datetime
import pytest
import pandas as pd
from reconciliacao.simulation.excel_generator import generate_payment_records, write_excel
from reconciliacao.simulation.xml_generator import generate_nfse, write_xml
from reconciliacao.etl.loader import load_pagamentos, load_nfse


@pytest.fixture
def excel_file(tmp_path):
    df = generate_payment_records(n=20, seed=42)
    path = tmp_path / "pagamentos.xlsx"
    write_excel(df, path)
    return path


@pytest.fixture
def xml_file(tmp_path):
    df_pag = generate_payment_records(n=20, seed=42)
    df_nfse = generate_nfse(df_pag, conciliation_rate=0.70, seed=42)
    path = tmp_path / "nfse.xml"
    write_xml(df_nfse, path)
    return path


def test_load_pagamentos_count(excel_file):
    df = load_pagamentos(excel_file)
    assert len(df) == 20


def test_load_pagamentos_schema(excel_file):
    df = load_pagamentos(excel_file)
    assert set(df.columns) == {
        "id_pagamento", "cnpj_fornecedor", "data_pagamento",
        "valor_pago", "descricao", "centro_custo",
    }


def test_load_nfse_count(xml_file):
    df = load_nfse(xml_file)
    assert len(df) == 20


def test_load_nfse_schema(xml_file):
    df = load_nfse(xml_file)
    expected = {
        "nfse_numero", "nfse_codigo_verificacao", "nfse_data_emissao",
        "nfse_competencia", "nfse_cnpj_prestador", "nfse_razao_social_prestador",
        "nfse_cnpj_tomador", "nfse_razao_social_tomador", "nfse_valor_servicos",
        "nfse_valor_iss", "nfse_aliquota", "nfse_valor_liquido",
        "nfse_item_lista_servico", "nfse_discriminacao", "nfse_codigo_municipio",
    }
    assert expected.issubset(set(df.columns))


def test_load_nfse_cnpj_field_populated(xml_file):
    df = load_nfse(xml_file)
    assert df["nfse_cnpj_prestador"].notna().all()
    assert (df["nfse_cnpj_prestador"].str.len() > 0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_etl/test_loader.py -v
```

Expected: `ImportError` for `reconciliacao.etl.loader`

- [ ] **Step 3: Write `src/reconciliacao/etl/loader.py`**

```python
from pathlib import Path

import pandas as pd
from lxml import etree

_NS = {"ns": "http://www.abrasf.org.br/nfse.xsd"}


def _text(el: etree._Element, xpath: str) -> str:
    nodes = el.xpath(xpath, namespaces=_NS)
    return nodes[0].text.strip() if nodes and nodes[0].text else ""


def load_pagamentos(path: str | Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl", dtype=str)


def load_nfse(path: str | Path) -> pd.DataFrame:
    tree = etree.parse(str(path))
    records = []
    for comp in tree.xpath("//ns:CompNfse", namespaces=_NS):
        records.append({
            "nfse_numero": _text(comp, ".//ns:InfNfse/ns:Numero"),
            "nfse_codigo_verificacao": _text(comp, ".//ns:CodigoVerificacao"),
            "nfse_data_emissao": _text(comp, ".//ns:DataEmissao"),
            "nfse_competencia": _text(comp, ".//ns:Competencia"),
            "nfse_cnpj_prestador": _text(comp, ".//ns:PrestadorServico//ns:Cnpj"),
            "nfse_razao_social_prestador": _text(comp, ".//ns:PrestadorServico/ns:RazaoSocial"),
            "nfse_cnpj_tomador": _text(comp, ".//ns:TomadorServico//ns:Cnpj"),
            "nfse_razao_social_tomador": _text(comp, ".//ns:TomadorServico/ns:RazaoSocial"),
            "nfse_valor_servicos": _text(comp, ".//ns:ValorServicos"),
            "nfse_valor_iss": _text(comp, ".//ns:ValorIss"),
            "nfse_aliquota": _text(comp, ".//ns:Aliquota"),
            "nfse_valor_liquido": _text(comp, ".//ns:ValorLiquidoNfse"),
            "nfse_item_lista_servico": _text(comp, ".//ns:ItemListaServico"),
            "nfse_discriminacao": _text(comp, ".//ns:Discriminacao"),
            "nfse_codigo_municipio": _text(comp, ".//ns:CodigoMunicipio"),
        })
    return pd.DataFrame(records)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_etl/test_loader.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/reconciliacao/etl/loader.py tests/test_etl/test_loader.py
git commit -m "feat: ETL loader — read Excel payments and ABRASF XML NFS-e into DataFrames"
```

---

## Task 7: ETL Cleaner

**Files:**
- Create: `src/reconciliacao/etl/cleaner.py`
- Create: `tests/test_etl/test_cleaner.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_etl/test_cleaner.py
import datetime
import pandas as pd
import pytest
from reconciliacao.etl.cleaner import clean_pagamentos, clean_nfse


def test_clean_pagamentos_normalizes_cnpj():
    df = pd.DataFrame({
        "id_pagamento": ["PAG-1"],
        "cnpj_fornecedor": ["11.222.333/0001-81"],
        "data_pagamento": ["2024-01-15"],
        "valor_pago": ["1000.00"],
        "descricao": ["servico"],
        "centro_custo": ["CC-001"],
    })
    result = clean_pagamentos(df)
    assert result["cnpj_fornecedor"].iloc[0] == "11222333000181"


def test_clean_pagamentos_validates_cnpj():
    df = pd.DataFrame({
        "id_pagamento": ["PAG-1", "PAG-2"],
        "cnpj_fornecedor": ["11222333000181", "11222333000182"],
        "data_pagamento": ["2024-01-15", "2024-01-15"],
        "valor_pago": ["1000.00", "2000.00"],
        "descricao": ["a", "b"],
        "centro_custo": ["CC-1", "CC-2"],
    })
    result = clean_pagamentos(df)
    assert result["cnpj_valid"].tolist() == [True, False]


def test_clean_pagamentos_parses_dates():
    df = pd.DataFrame({
        "id_pagamento": ["PAG-1"],
        "cnpj_fornecedor": ["11222333000181"],
        "data_pagamento": ["2024-06-15"],
        "valor_pago": ["999.50"],
        "descricao": ["x"],
        "centro_custo": ["CC-1"],
    })
    result = clean_pagamentos(df)
    assert result["data_pagamento"].iloc[0] == datetime.date(2024, 6, 15)


def test_clean_pagamentos_parses_float_values():
    df = pd.DataFrame({
        "id_pagamento": ["PAG-1"],
        "cnpj_fornecedor": ["11222333000181"],
        "data_pagamento": ["2024-01-15"],
        "valor_pago": ["1.234,56"],
        "descricao": ["x"],
        "centro_custo": ["CC-1"],
    })
    result = clean_pagamentos(df)
    assert result["valor_pago"].iloc[0] == pytest.approx(1234.56)


def test_clean_pagamentos_flags_duplicates():
    df = pd.DataFrame({
        "id_pagamento": ["PAG-1", "PAG-1"],
        "cnpj_fornecedor": ["11222333000181", "11222333000181"],
        "data_pagamento": ["2024-01-15", "2024-01-15"],
        "valor_pago": ["1000", "1000"],
        "descricao": ["x", "x"],
        "centro_custo": ["CC-1", "CC-1"],
    })
    result = clean_pagamentos(df)
    assert result["is_duplicate"].all()


def test_clean_nfse_parses_numeric_fields():
    df = pd.DataFrame({
        "nfse_numero": ["1"],
        "nfse_codigo_verificacao": ["ABC"],
        "nfse_data_emissao": ["2024-01-15T00:00:00"],
        "nfse_competencia": ["2024-01-01T00:00:00"],
        "nfse_cnpj_prestador": ["11222333000181"],
        "nfse_razao_social_prestador": ["Empresa A"],
        "nfse_cnpj_tomador": ["44555666000195"],
        "nfse_razao_social_tomador": ["Empresa B"],
        "nfse_valor_servicos": ["2500.00"],
        "nfse_valor_iss": ["125.00"],
        "nfse_aliquota": ["5.00"],
        "nfse_valor_liquido": ["2375.00"],
        "nfse_item_lista_servico": ["1.01"],
        "nfse_discriminacao": ["Servicos"],
        "nfse_codigo_municipio": ["3550308"],
    })
    result = clean_nfse(df)
    assert result["nfse_valor_servicos"].iloc[0] == pytest.approx(2500.00)
    assert result["nfse_data_emissao"].iloc[0] == datetime.date(2024, 1, 15)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_etl/test_cleaner.py -v
```

Expected: `ImportError` for `reconciliacao.etl.cleaner`

- [ ] **Step 3: Write `src/reconciliacao/etl/cleaner.py`**

```python
import datetime

import pandas as pd

from reconciliacao.utils.cnpj import normalize_cnpj, validate_cnpj


def _parse_float(series: pd.Series) -> pd.Series:
    # Try English format first ("1234.56"), fall back to Brazilian ("1.234,56") for NaN
    s = series.astype(str).str.strip()
    result = pd.to_numeric(s, errors="coerce")
    if result.isna().any():
        br_cleaned = s.str.replace(r"\.", "", regex=True).str.replace(",", ".", regex=False)
        result = result.fillna(pd.to_numeric(br_cleaned, errors="coerce"))
    return result


def _parse_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.date


def clean_pagamentos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cnpj_fornecedor"] = df["cnpj_fornecedor"].astype(str).apply(normalize_cnpj)
    df["cnpj_valid"] = df["cnpj_fornecedor"].apply(validate_cnpj)
    df["data_pagamento"] = _parse_date(df["data_pagamento"])
    df["valor_pago"] = _parse_float(df["valor_pago"].astype(str))
    df["is_duplicate"] = df.duplicated("id_pagamento", keep=False)
    return df


def clean_nfse(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["nfse_cnpj_prestador"] = df["nfse_cnpj_prestador"].astype(str).apply(normalize_cnpj)
    df["nfse_cnpj_valid"] = df["nfse_cnpj_prestador"].apply(validate_cnpj)
    df["nfse_data_emissao"] = _parse_date(df["nfse_data_emissao"])
    df["nfse_valor_servicos"] = pd.to_numeric(df["nfse_valor_servicos"], errors="coerce")
    df["nfse_valor_iss"] = pd.to_numeric(df["nfse_valor_iss"], errors="coerce")
    df["nfse_aliquota"] = pd.to_numeric(df["nfse_aliquota"], errors="coerce")
    df["nfse_valor_liquido"] = pd.to_numeric(df["nfse_valor_liquido"], errors="coerce")
    df["is_duplicate"] = df.duplicated("nfse_numero", keep=False)
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_etl/test_cleaner.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/reconciliacao/etl/cleaner.py tests/test_etl/test_cleaner.py
git commit -m "feat: ETL cleaner — normalize CNPJ, parse dates and values, flag duplicates"
```

---

## Task 8: ETL Labeler

**Files:**
- Create: `src/reconciliacao/etl/labeler.py`
- Create: `tests/test_etl/test_labeler.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_etl/test_labeler.py
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
    # PAG-000002: CNPJ match, 5-day tolerance, value is 2600 vs 2500 (4% off) → label 0 (outside 2%)
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_etl/test_labeler.py -v
```

Expected: `ImportError` for `reconciliacao.etl.labeler`

- [ ] **Step 3: Write `src/reconciliacao/etl/labeler.py`**

```python
import pandas as pd


def label_records(
    df_pag: pd.DataFrame,
    df_nfse: pd.DataFrame,
    date_tolerance_days: int,
    value_tolerance_pct: float,
    n_classes: int,
) -> pd.DataFrame:
    merged = df_pag.merge(
        df_nfse,
        left_on="cnpj_fornecedor",
        right_on="nfse_cnpj_prestador",
        how="left",
    )

    pag_dates = pd.to_datetime(merged["data_pagamento"])
    nfse_dates = pd.to_datetime(merged["nfse_data_emissao"])
    merged["delta_days"] = (pag_dates - nfse_dates).abs().dt.days.fillna(9999)

    safe_valor = merged["nfse_valor_servicos"].replace(0, float("nan"))
    merged["delta_valor_pct"] = (
        (merged["valor_pago"] - merged["nfse_valor_servicos"]).abs() / safe_valor * 100
    ).fillna(100.0)

    merged["label"] = 0

    mask_date_ok = merged["delta_days"] <= date_tolerance_days
    mask_valor_ok = merged["delta_valor_pct"] <= value_tolerance_pct

    if n_classes == 2:
        merged.loc[mask_date_ok & mask_valor_ok, "label"] = 1
    else:
        # Parcialmente conciliated: within 4× date tolerance OR within 5× value tolerance
        mask_date_close = merged["delta_days"] <= date_tolerance_days * 4
        mask_valor_close = merged["delta_valor_pct"] <= value_tolerance_pct * 5
        mask_parcial = (~(mask_date_ok & mask_valor_ok)) & (mask_date_close | mask_valor_close)
        merged.loc[mask_parcial, "label"] = 1
        merged.loc[mask_date_ok & mask_valor_ok, "label"] = 2

    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_etl/test_labeler.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/reconciliacao/etl/labeler.py tests/test_etl/test_labeler.py
git commit -m "feat: ETL labeler — join records and assign exact/fuzzy reconciliation labels"
```

---

## Task 9: Feature Engineering

**Files:**
- Create: `src/reconciliacao/models/features.py`
- Create: `tests/test_models/test_features.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models/test_features.py
import datetime
import numpy as np
import pandas as pd
import pytest
from reconciliacao.models.features import build_features


@pytest.fixture
def sample_merged():
    return pd.DataFrame({
        "id_pagamento": ["PAG-1", "PAG-2", "PAG-3"],
        "cnpj_fornecedor": ["11222333000181", "44555666000195", None],
        "data_pagamento": [datetime.date(2024, 1, 15)] * 3,
        "valor_pago": [1000.0, 2500.0, 800.0],
        "descricao": ["consultoria TI", "manutencao predial", "limpeza"],
        "centro_custo": ["CC-1", "CC-2", "CC-3"],
        "nfse_cnpj_prestador": ["11222333000181", "44555666000195", None],
        "nfse_data_emissao": [datetime.date(2024, 1, 15), datetime.date(2024, 2, 15), None],
        "nfse_valor_servicos": [1000.0, 2600.0, None],
        "nfse_valor_iss": [50.0, 130.0, None],
        "nfse_aliquota": [5.0, 5.0, None],
        "nfse_discriminacao": ["Servicos de TI", "Manutencao Predial", None],
        "nfse_codigo_municipio": ["3550308", "3550308", None],
        "delta_days": [0.0, 31.0, 9999.0],
        "delta_valor_pct": [0.0, 4.0, 100.0],
        "label": [2, 0, 0],
    })


def test_build_features_returns_dataframe_and_series(sample_merged):
    X, y = build_features(sample_merged)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_build_features_row_count(sample_merged):
    X, y = build_features(sample_merged)
    assert len(X) == len(sample_merged)
    assert len(y) == len(sample_merged)


def test_build_features_no_nan(sample_merged):
    X, y = build_features(sample_merged)
    assert not X.isna().any().any()
    assert not y.isna().any()


def test_build_features_expected_columns(sample_merged):
    X, _ = build_features(sample_merged)
    expected = {"delta_days", "delta_valor_pct", "valor_pago", "nfse_valor_iss",
                "nfse_aliquota", "cnpj_match", "descricao_similarity"}
    assert expected.issubset(set(X.columns))


def test_build_features_cnpj_match_flag(sample_merged):
    X, _ = build_features(sample_merged)
    assert X["cnpj_match"].tolist() == [1, 1, 0]


def test_build_features_labels_match_input(sample_merged):
    _, y = build_features(sample_merged)
    assert y.tolist() == [2, 0, 0]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_models/test_features.py -v
```

Expected: `ImportError` for `reconciliacao.models.features`

- [ ] **Step 3: Write `src/reconciliacao/models/features.py`**

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feat = pd.DataFrame(index=df.index)

    feat["delta_days"] = df["delta_days"].fillna(9999.0).astype(float)
    feat["delta_valor_pct"] = df["delta_valor_pct"].fillna(100.0).astype(float)
    feat["valor_pago"] = df["valor_pago"].fillna(0.0).astype(float)
    feat["nfse_valor_iss"] = df["nfse_valor_iss"].fillna(0.0).astype(float)
    feat["nfse_aliquota"] = df["nfse_aliquota"].fillna(0.0).astype(float)
    feat["cnpj_match"] = df["cnpj_fornecedor"].notna().astype(int)

    desc_pag = df["descricao"].fillna("").astype(str).tolist()
    desc_nfse = df["nfse_discriminacao"].fillna("").astype(str).tolist()
    all_texts = desc_pag + desc_nfse

    vectorizer = TfidfVectorizer(min_df=1)
    tfidf = vectorizer.fit_transform(all_texts)
    n = len(df)
    similarities = cosine_similarity(tfidf[:n], tfidf[n:]).diagonal()
    feat["descricao_similarity"] = similarities

    y = df["label"].fillna(0).astype(int).reset_index(drop=True)
    feat = feat.reset_index(drop=True)

    return feat, y
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_models/test_features.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/reconciliacao/models/features.py tests/test_models/test_features.py
git commit -m "feat: feature engineering — delta metrics, CNPJ match flag, TF-IDF description similarity"
```

---

## Task 10: Model Trainer

**Files:**
- Create: `src/reconciliacao/models/trainer.py`
- Create: `tests/test_models/test_trainer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models/test_trainer.py
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from reconciliacao.models.trainer import get_pipelines, train_all


@pytest.fixture
def small_dataset():
    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        "delta_days": rng.uniform(0, 30, 60).tolist(),
        "delta_valor_pct": rng.uniform(0, 20, 60).tolist(),
        "valor_pago": rng.uniform(500, 50000, 60).tolist(),
        "nfse_valor_iss": rng.uniform(10, 500, 60).tolist(),
        "nfse_aliquota": rng.uniform(2, 5, 60).tolist(),
        "cnpj_match": [1] * 40 + [0] * 20,
        "descricao_similarity": rng.uniform(0, 1, 60).tolist(),
    })
    y = pd.Series([0] * 20 + [1] * 20 + [2] * 20)
    return X, y


@pytest.fixture
def mini_config():
    return {
        "simulation": {"random_seed": 42},
        "models": {
            "random_forest": {"n_estimators": 10, "class_weight": "balanced_subsample"},
            "svm": {"kernel": "rbf", "class_weight": "balanced"},
            "logistic_regression": {"penalty": "l2", "class_weight": "balanced"},
            "cv_folds": 2,
            "scoring": "f1_macro",
            "test_size": 0.25,
        },
    }


def test_get_pipelines_returns_three_algorithms(mini_config):
    pipelines = get_pipelines(mini_config)
    assert set(pipelines.keys()) == {"random_forest", "svm", "logistic_regression"}


def test_train_all_saves_model_files(small_dataset, mini_config, tmp_path):
    X, y = small_dataset
    train_all(X, y, mini_config, tmp_path / "models")
    assert (tmp_path / "models" / "random_forest.joblib").exists()
    assert (tmp_path / "models" / "svm.joblib").exists()
    assert (tmp_path / "models" / "logistic_regression.joblib").exists()


def test_train_all_returns_results_dict(small_dataset, mini_config, tmp_path):
    X, y = small_dataset
    results = train_all(X, y, mini_config, tmp_path / "models")
    assert set(results.keys()) == {"random_forest", "svm", "logistic_regression"}
    for r in results.values():
        assert "estimator" in r
        assert "X_test" in r
        assert "y_test" in r
        assert "cv_score_mean" in r


def test_train_all_estimator_can_predict(small_dataset, mini_config, tmp_path):
    X, y = small_dataset
    results = train_all(X, y, mini_config, tmp_path / "models")
    for r in results.values():
        preds = r["estimator"].predict(r["X_test"])
        assert len(preds) == len(r["y_test"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_models/test_trainer.py -v
```

Expected: `ImportError` for `reconciliacao.models.trainer`

- [ ] **Step 3: Write `src/reconciliacao/models/trainer.py`**

```python
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_pipelines(cfg: dict) -> dict[str, tuple]:
    seed = cfg["simulation"]["random_seed"]
    mcfg = cfg["models"]

    return {
        "random_forest": (
            Pipeline([
                ("clf", RandomForestClassifier(
                    n_estimators=mcfg["random_forest"]["n_estimators"],
                    class_weight=mcfg["random_forest"]["class_weight"],
                    random_state=seed,
                )),
            ]),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10]},
        ),
        "svm": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(
                    kernel=mcfg["svm"]["kernel"],
                    class_weight=mcfg["svm"]["class_weight"],
                    probability=True,
                    random_state=seed,
                )),
            ]),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "logistic_regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty=mcfg["logistic_regression"]["penalty"],
                    class_weight=mcfg["logistic_regression"]["class_weight"],
                    max_iter=1000,
                    random_state=seed,
                )),
            ]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
    }


def train_all(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    output_dir: Path,
) -> dict:
    seed = cfg["simulation"]["random_seed"]
    mcfg = cfg["models"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=mcfg["test_size"],
        stratify=y,
        random_state=seed,
    )

    cv = StratifiedKFold(n_splits=mcfg["cv_folds"], shuffle=True, random_state=seed)
    results = {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, (estimator, param_grid) in get_pipelines(cfg).items():
        gs = GridSearchCV(
            estimator,
            param_grid,
            cv=cv,
            scoring=mcfg["scoring"],
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X_train, y_train)
        joblib.dump(gs.best_estimator_, output_dir / f"{name}.joblib")
        best_idx = gs.best_index_
        results[name] = {
            "estimator": gs.best_estimator_,
            "X_test": X_test,
            "y_test": y_test,
            "cv_score_mean": float(gs.cv_results_["mean_test_score"][best_idx]),
            "cv_score_std": float(gs.cv_results_["std_test_score"][best_idx]),
        }

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_models/test_trainer.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/reconciliacao/models/trainer.py tests/test_models/test_trainer.py
git commit -m "feat: model trainer — GridSearchCV for Random Forest, SVM, and Logistic Regression"
```

---

## Task 11: Model Evaluator

**Files:**
- Create: `src/reconciliacao/models/evaluator.py`
- Create: `tests/test_models/test_evaluator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models/test_evaluator.py
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.dummy import DummyClassifier
from reconciliacao.models.evaluator import evaluate_all


@pytest.fixture
def dummy_results(tmp_path):
    rng = np.random.RandomState(42)
    X_test = pd.DataFrame({"a": rng.rand(30), "b": rng.rand(30)})
    y_test = pd.Series([0] * 15 + [1] * 15)
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_test, y_test)
    return {
        "dummy": {
            "estimator": clf,
            "X_test": X_test,
            "y_test": y_test,
            "cv_score_mean": 0.5,
            "cv_score_std": 0.05,
        }
    }


def test_evaluate_all_returns_dataframe(dummy_results, tmp_path):
    result = evaluate_all(dummy_results, n_classes=2, output_dir=tmp_path)
    assert isinstance(result, pd.DataFrame)


def test_evaluate_all_has_expected_columns(dummy_results, tmp_path):
    result = evaluate_all(dummy_results, n_classes=2, output_dir=tmp_path)
    for col in ("accuracy", "f1_macro", "precision_macro", "recall_macro", "cv_mean"):
        assert col in result.columns


def test_evaluate_all_saves_metrics_csv(dummy_results, tmp_path):
    evaluate_all(dummy_results, n_classes=2, output_dir=tmp_path)
    assert (tmp_path / "metrics_summary.csv").exists()


def test_evaluate_all_saves_confusion_matrix_png(dummy_results, tmp_path):
    evaluate_all(dummy_results, n_classes=2, output_dir=tmp_path)
    assert (tmp_path / "confusion_matrix_dummy.png").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_models/test_evaluator.py -v
```

Expected: `ImportError` for `reconciliacao.models.evaluator`

- [ ] **Step 3: Write `src/reconciliacao/models/evaluator.py`**

```python
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_all(results: dict, n_classes: int, output_dir: Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    roc_fig, roc_ax = plt.subplots() if n_classes == 2 else (None, None)

    for name, r in results.items():
        est = r["estimator"]
        X_test, y_test = r["X_test"], r["y_test"]
        y_pred = est.predict(X_test)

        row = {
            "algorithm": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "cv_mean": r["cv_score_mean"],
            "cv_std": r["cv_score_std"],
        }

        if n_classes == 2 and hasattr(est, "predict_proba"):
            y_prob = est.predict_proba(X_test)[:, 1]
            try:
                row["roc_auc"] = roc_auc_score(y_test, y_prob)
                RocCurveDisplay.from_estimator(est, X_test, y_test, ax=roc_ax, name=name)
            except ValueError:
                row["roc_auc"] = float("nan")

        rows.append(row)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        fig, ax = plt.subplots()
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"Confusion Matrix — {name}")
        fig.savefig(output_dir / f"confusion_matrix_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Feature importances (Random Forest only)
        clf = getattr(est, "named_steps", {}).get("clf", est)
        if hasattr(clf, "feature_importances_") and hasattr(X_test, "columns"):
            imp = pd.Series(clf.feature_importances_, index=X_test.columns)
            fig, ax = plt.subplots()
            imp.sort_values().plot.barh(ax=ax)
            ax.set_title("Feature Importances — Random Forest")
            fig.savefig(output_dir / "feature_importance_rf.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    if roc_fig is not None:
        roc_ax.set_title("ROC Curve — Exact Scenario")
        roc_fig.savefig(output_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close(roc_fig)

    df_metrics = pd.DataFrame(rows).set_index("algorithm")
    df_metrics.to_csv(output_dir / "metrics_summary.csv")
    return df_metrics
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_models/test_evaluator.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/reconciliacao/models/evaluator.py tests/test_models/test_evaluator.py
git commit -m "feat: model evaluator — classification metrics, confusion matrices, ROC curves, artifacts"
```

---

## Task 12: Pipeline Runner

**Files:**
- Create: `run_pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing integration test**

```python
# tests/test_pipeline.py
import subprocess
import sys
from pathlib import Path
import pytest


@pytest.mark.integration
def test_pipeline_exact_scenario_runs_end_to_end(tmp_path, monkeypatch):
    """Smoke test: run the full pipeline on a tiny dataset."""
    import yaml
    cfg = {
        "simulation": {"n_records": 80, "conciliation_rate": 0.70, "random_seed": 42},
        "scenarios": {
            "exact": {"date_tolerance_days": 0, "value_tolerance_pct": 0.0, "n_classes": 2},
            "fuzzy": {"date_tolerance_days": 5, "value_tolerance_pct": 2.0, "n_classes": 3},
        },
        "models": {
            "random_forest": {"n_estimators": 5, "class_weight": "balanced_subsample"},
            "svm": {"kernel": "rbf", "class_weight": "balanced"},
            "logistic_regression": {"penalty": "l2", "class_weight": "balanced"},
            "cv_folds": 2,
            "scoring": "f1_macro",
            "test_size": 0.20,
        },
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(cfg))
    monkeypatch.chdir(tmp_path)

    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent / "run_pipeline.py"), "--scenario", "exact"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "data" / "results" / "exact" / "metrics_summary.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_pipeline.py -v -m integration
```

Expected: `FAILED` — `run_pipeline.py` does not exist

- [ ] **Step 3: Write `run_pipeline.py`**

```python
import argparse
from pathlib import Path

from reconciliacao.etl.cleaner import clean_nfse, clean_pagamentos
from reconciliacao.etl.labeler import label_records
from reconciliacao.etl.loader import load_nfse, load_pagamentos
from reconciliacao.models.evaluator import evaluate_all
from reconciliacao.models.features import build_features
from reconciliacao.models.trainer import train_all
from reconciliacao.simulation.excel_generator import generate_payment_records, write_excel
from reconciliacao.simulation.xml_generator import generate_nfse, write_xml
from reconciliacao.utils.config import load_config


def run(scenario: str, cfg: dict) -> None:
    scfg = cfg["scenarios"][scenario]
    sim = cfg["simulation"]

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    results_dir = Path(f"data/results/{scenario}")

    print(f"[1/5] Simulating {sim['n_records']} records...")
    df_pag = generate_payment_records(sim["n_records"], sim["random_seed"])
    write_excel(df_pag, raw_dir / "pagamentos.xlsx")
    df_nfse_sim = generate_nfse(df_pag, sim["conciliation_rate"], sim["random_seed"])
    write_xml(df_nfse_sim, raw_dir / "nfse.xml")

    print("[2/5] Running ETL...")
    df_pag_raw = load_pagamentos(raw_dir / "pagamentos.xlsx")
    df_nfse_raw = load_nfse(raw_dir / "nfse.xml")
    df_pag_clean = clean_pagamentos(df_pag_raw)
    df_nfse_clean = clean_nfse(df_nfse_raw)
    df_reconciled = label_records(
        df_pag_clean,
        df_nfse_clean,
        date_tolerance_days=scfg["date_tolerance_days"],
        value_tolerance_pct=scfg["value_tolerance_pct"],
        n_classes=scfg["n_classes"],
    )
    df_reconciled["scenario"] = scenario
    processed_dir.mkdir(parents=True, exist_ok=True)
    df_reconciled.to_csv(processed_dir / f"{scenario}_reconciliado.csv", index=False)
    print(f"    Label distribution:\n{df_reconciled['label'].value_counts().to_string()}")

    print("[3/5] Building features...")
    X, y = build_features(df_reconciled)

    print("[4/5] Training models (this may take a few minutes)...")
    training_results = train_all(X, y, cfg, results_dir / "models")

    print("[5/5] Evaluating models...")
    df_metrics = evaluate_all(training_results, scfg["n_classes"], results_dir)
    print("\n" + df_metrics.to_string())
    print(f"\nArtifacts saved to {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ML reconciliation pipeline.")
    parser.add_argument("--scenario", choices=["exact", "fuzzy"], required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.scenario, load_config(args.config))
```

- [ ] **Step 4: Run the integration test**

```bash
pytest tests/test_pipeline.py -v -m integration
```

Expected: `1 passed` (may take ~60s)

- [ ] **Step 5: Run the full pipeline manually on both scenarios to verify end-to-end output**

```bash
python run_pipeline.py --scenario exact
python run_pipeline.py --scenario fuzzy
```

Expected: both complete without errors and print metrics tables. Verify:
- `data/results/exact/metrics_summary.csv` exists and has 3 rows
- `data/results/fuzzy/metrics_summary.csv` exists and has 3 rows
- `data/results/exact/confusion_matrix_random_forest.png` exists

- [ ] **Step 6: Run the full test suite**

```bash
pytest -v --ignore=tests/test_pipeline.py
```

Expected: all unit tests pass

- [ ] **Step 7: Commit**

```bash
git add run_pipeline.py tests/test_pipeline.py
git commit -m "feat: pipeline runner — end-to-end CLI connecting simulation, ETL, training, and evaluation"
```

---

## Task 13: Jupyter Notebooks

**Files:**
- Create: `notebooks/01_simulacao.ipynb`
- Create: `notebooks/02_etl.ipynb`
- Create: `notebooks/03_modelagem.ipynb`
- Create: `notebooks/04_avaliacao.ipynb`

Each notebook imports from the `reconciliacao` package and adds narrative markdown cells. Create them via `jupyter notebook` or by writing the JSON directly.

- [ ] **Step 1: Create `notebooks/01_simulacao.ipynb`**

The notebook should contain these cells in order:

**Cell 1 (Markdown):**
```
# Capítulo 1 — Simulação de Dados
Este notebook descreve e executa a geração dos datasets sintéticos utilizados na pesquisa:
um arquivo Excel com registros de pagamentos a fornecedores e um XML no padrão NFS-e ABRASF
com notas fiscais de serviços eletrônicas.
```

**Cell 2 (Code):**
```python
from reconciliacao.utils.config import load_config
from reconciliacao.simulation.excel_generator import generate_payment_records, write_excel
from reconciliacao.simulation.xml_generator import generate_nfse, write_xml
import pandas as pd

cfg = load_config("../config.yaml")
sim = cfg["simulation"]
```

**Cell 3 (Code — generate and display payment records):**
```python
df_pag = generate_payment_records(sim["n_records"], sim["random_seed"])
write_excel(df_pag, "../data/raw/pagamentos.xlsx")
print(f"Registros de pagamento gerados: {len(df_pag)}")
df_pag.head(10)
```

**Cell 4 (Code — generate NFS-e and show mismatch distribution):**
```python
df_nfse = generate_nfse(df_pag, sim["conciliation_rate"], sim["random_seed"])
write_xml(df_nfse, "../data/raw/nfse.xml")
print(f"NFS-e geradas: {len(df_nfse)}")
print(f"Conciliação esperada: {sim['conciliation_rate']*100:.0f}%")
df_nfse.head(10)
```

**Cell 5 (Code — basic statistics):**
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df_pag["valor_pago"].hist(bins=40, ax=axes[0])
axes[0].set_title("Distribuição de valor_pago")
axes[0].set_xlabel("BRL")
df_nfse["nfse_valor_servicos"].hist(bins=40, ax=axes[1])
axes[1].set_title("Distribuição de ValorServicos (NFS-e)")
axes[1].set_xlabel("BRL")
plt.tight_layout()
plt.show()
```

- [ ] **Step 2: Create `notebooks/02_etl.ipynb`**

**Cell 1 (Markdown):**
```
# Capítulo 2 — ETL: Carga, Limpeza e Rotulagem
Exploramos os dados brutos, aplicamos normalização de CNPJ, datas e valores,
e geramos os datasets rotulados para os cenários de conciliação exata e fuzzy.
```

**Cell 2 (Code):**
```python
from reconciliacao.utils.config import load_config
from reconciliacao.etl.loader import load_pagamentos, load_nfse
from reconciliacao.etl.cleaner import clean_pagamentos, clean_nfse
from reconciliacao.etl.labeler import label_records
import pandas as pd

cfg = load_config("../config.yaml")
df_pag_raw = load_pagamentos("../data/raw/pagamentos.xlsx")
df_nfse_raw = load_nfse("../data/raw/nfse.xml")
```

**Cell 3 (Code — show raw data quality):**
```python
print("=== Pagamentos Brutos ===")
print(df_pag_raw.dtypes)
print(df_pag_raw.head(5).to_string())
```

**Cell 4 (Code — clean and show validation results):**
```python
df_pag = clean_pagamentos(df_pag_raw)
df_nfse = clean_nfse(df_nfse_raw)

print(f"CNPJs inválidos (pagamentos): {(~df_pag['cnpj_valid']).sum()}")
print(f"CNPJs inválidos (NFS-e): {(~df_nfse['nfse_cnpj_valid']).sum()}")
print(f"Duplicatas (pagamentos): {df_pag['is_duplicate'].sum()}")
```

**Cell 5 (Code — label both scenarios):**
```python
for scenario in ["exact", "fuzzy"]:
    scfg = cfg["scenarios"][scenario]
    df = label_records(df_pag, df_nfse,
                       date_tolerance_days=scfg["date_tolerance_days"],
                       value_tolerance_pct=scfg["value_tolerance_pct"],
                       n_classes=scfg["n_classes"])
    df["scenario"] = scenario
    df.to_csv(f"../data/processed/{scenario}_reconciliado.csv", index=False)
    print(f"\n=== {scenario.upper()} ===")
    print(df["label"].value_counts().sort_index())
```

- [ ] **Step 3: Create `notebooks/03_modelagem.ipynb`**

**Cell 1 (Markdown):**
```
# Capítulo 3 — Modelagem: Treinamento e Validação Cruzada
Treinamos e ajustamos Random Forest, SVM e Regressão Logística para cada cenário,
usando GridSearchCV com validação cruzada estratificada de 5 folds.
```

**Cell 2 (Code):**
```python
import pandas as pd
from reconciliacao.utils.config import load_config
from reconciliacao.models.features import build_features
from reconciliacao.models.trainer import train_all
from pathlib import Path

cfg = load_config("../config.yaml")

for scenario in ["exact", "fuzzy"]:
    print(f"\n{'='*40}")
    print(f"Treinando cenário: {scenario.upper()}")
    df = pd.read_csv(f"../data/processed/{scenario}_reconciliado.csv")
    X, y = build_features(df)
    results = train_all(X, y, cfg, Path(f"../data/results/{scenario}/models"))
    print(f"Modelos treinados e salvos em data/results/{scenario}/models/")
    for name, r in results.items():
        print(f"  {name}: CV F1-macro = {r['cv_score_mean']:.4f} ± {r['cv_score_std']:.4f}")
```

- [ ] **Step 4: Create `notebooks/04_avaliacao.ipynb`**

**Cell 1 (Markdown):**
```
# Capítulo 4 — Avaliação: Métricas e Comparação entre Cenários
Calculamos as métricas de classificação para cada algoritmo em cada cenário,
exibimos matrizes de confusão e produzimos a tabela comparativa central do trabalho.
```

**Cell 2 (Code):**
```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from reconciliacao.utils.config import load_config
from reconciliacao.models.features import build_features
from reconciliacao.models.trainer import train_all
from reconciliacao.models.evaluator import evaluate_all

cfg = load_config("../config.yaml")
all_metrics = []

for scenario in ["exact", "fuzzy"]:
    df = pd.read_csv(f"../data/processed/{scenario}_reconciliado.csv")
    X, y = build_features(df)
    results = train_all(X, y, cfg, Path(f"../data/results/{scenario}/models"))
    scfg = cfg["scenarios"][scenario]
    metrics = evaluate_all(results, scfg["n_classes"], Path(f"../data/results/{scenario}"))
    metrics["scenario"] = scenario
    all_metrics.append(metrics.reset_index())

summary = pd.concat(all_metrics).set_index(["scenario", "algorithm"])
print(summary[["accuracy", "f1_macro", "precision_macro", "recall_macro", "cv_mean"]].to_string())
```

**Cell 3 (Code — display confusion matrices):**
```python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
algorithms = ["random_forest", "svm", "logistic_regression"]
for row_idx, scenario in enumerate(["exact", "fuzzy"]):
    for col_idx, alg in enumerate(algorithms):
        img_path = Path(f"../data/results/{scenario}/confusion_matrix_{alg}.png")
        if img_path.exists():
            axes[row_idx][col_idx].imshow(mpimg.imread(str(img_path)))
            axes[row_idx][col_idx].axis("off")
            axes[row_idx][col_idx].set_title(f"{scenario} / {alg}")
plt.tight_layout()
plt.show()
```

- [ ] **Step 5: Run all four notebooks end-to-end**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_simulacao.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_etl.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_modelagem.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_avaliacao.ipynb
```

Expected: all four complete without errors

- [ ] **Step 6: Run full test suite one final time**

```bash
pytest -v
```

Expected: all tests pass

- [ ] **Step 7: Final commit**

```bash
git add notebooks/
git commit -m "feat: Jupyter notebooks — simulation, ETL, modeling, and evaluation narratives"
```

---

## Running Both Scenarios

After all tasks are complete, run the full pipeline:

```bash
python run_pipeline.py --scenario exact
python run_pipeline.py --scenario fuzzy
```

Results will be in:
- `data/results/exact/metrics_summary.csv`
- `data/results/fuzzy/metrics_summary.csv`
- `data/results/exact/confusion_matrix_*.png`
- `data/results/fuzzy/confusion_matrix_*.png`
- `data/results/exact/roc_curve.png`
- `data/results/exact/feature_importance_rf.png`
