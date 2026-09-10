# Comparação entre Algoritmos — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir um experimento em que a comparação entre Random Forest, SVM e Regressão Logística tenha poder discriminante real — rótulo desacoplado das features por construção, sinal genuíno vindo de retenções tributárias, e decisão por recall de exceção sob precisão mínima.

**Architecture:** Um gerador novo produz pagamentos, notas e uma tabela de verdade; o rótulo vem exclusivamente dessa tabela. Cada pagamento é pareado com exatamente uma nota candidata do mesmo fornecedor, e o rótulo diz se esse pareamento é genuíno. As features são evidências observáveis do par. Um guarda anti-vazamento falha o pipeline se qualquer feature isolada separar as classes. A avaliação roda 10 sementes e compara os algoritmos por Wilcoxon pareado.

**Tech Stack:** Python 3.13+, pandas, scikit-learn, scipy, Faker (pt_BR), lxml, openpyxl, pytest.

**Spec:** [`docs/superpowers/specs/2026-09-10-comparacao-algoritmos-design.md`](../specs/2026-09-10-comparacao-algoritmos-design.md)

## Global Constraints

- **Nada nos módulos existentes é modificado.** `simulation/excel_generator.py`, `simulation/xml_generator.py`, `etl/labeler.py`, `etl/loader.py`, `etl/cleaner.py`, `models/features.py`, `models/evaluator.py` e `run_pipeline.py` ficam intactos — o experimento original precisa continuar reprodutível, porque é ele que sustenta o achado do vazamento. `models/trainer.py` é apenas **lido** (reuso de `get_pipelines`), nunca alterado.
- **Todo parâmetro vem de `config.yaml`**, bloco `comparison`. Nenhum valor fixo em código, seguindo a convenção do projeto.
- **Nenhuma feature pode participar da definição do rótulo.** O rótulo vem de `verdade.csv` e de mais nada.
- **Estatísticas ajustadas em dados de treino apenas.** A mediana de prazo por fornecedor é ajustada no treino e aplicada às demais partições.
- **Rótulo binário:** `1` = par verdadeiro, `0` = exceção (não é par). A classe de exceção é a `0`.
- **Semente:** toda função geradora recebe `seed` explícito e é determinística.
- **Testes:** pytest, nomes descritivos em inglês seguindo `tests/test_*/`, fixtures em `tests/conftest.py`. Testes lentos marcados `@pytest.mark.integration`.
- **Idioma:** nomes de código e docstrings em inglês (convenção do repositório); nomes de colunas de dados em português (convenção do domínio).

---

## Desvios do spec, decididos durante o planejamento

Dois pontos do spec foram refinados. Ambos exigem atualizar o spec — feito na Task 4.

1. **`iss_declarado_consistente` é substituída por `valor_retido_brl`.** O spec §5.2 definia
   `valor_iss ≈ valor_servicos × aliquota / 100`, que é uma propriedade **interna da nota** e não diz
   nada sobre o pareamento com o pagamento. Não teria sinal. No lugar entra o valor retido em reais —
   evidência de nível de par, que ainda interage com o limite de R$ 5.000 da CSRF.
2. **Não existem pagamentos sem nota candidata.** Cada pagamento é pareado com exatamente uma nota do
   mesmo fornecedor; o rótulo diz se esse par é genuíno. Consequência positiva: **as deltas são sempre
   computáveis, não há valores-sentinela e o `has_match` do spec §3 deixa de ser necessário** — a
   patologia de escala documentada em §2.2 do documento de discussão desaparece por construção.

---

## Nota de calibração para quem implementar

Com `match_rate: 0.70` e `hard_negative_rate: 0.30`, um classificador que use só a compatibilidade de
retenção acerta ≈91% dos casos: todos os pares verdadeiros (70%) mais os *soft negatives*
(30% × 0,70 = 21%). Os *hard negatives* (9%) são onde os algoritmos competem.

Isso passa no guarda anti-vazamento (limite 0,95) com folga pequena e proposital. **Se a Task 9 concluir
empate entre os três algoritmos, o parâmetro a aumentar é `hard_negative_rate`** — está em `config.yaml`
exatamente para isso (spec §12). Aumentá-lo torna a tarefa mais difícil e amplia a região onde as
capacidades dos algoritmos diferem. Não altere a `match_rate` para forçar um vencedor: calibrar a base até
aparecer um vencedor é a mesma classe de erro que este trabalho denuncia.

---

## Estrutura de arquivos

| Arquivo | Responsabilidade |
| --- | --- |
| `src/reconciliacao/simulation/retentions.py` | Aritmética de retenções: quais percentuais totais são legalmente plausíveis, e se um percentual observado bate com algum |
| `src/reconciliacao/simulation/truth_generator.py` | Gera pagamentos, notas e `verdade.csv` |
| `src/reconciliacao/etl/truth_labeler.py` | Junta pagamento ↔ nota candidata e aplica o rótulo vindo da verdade |
| `src/reconciliacao/models/features_v2.py` | Features de evidência, com ajuste treino/aplicação |
| `src/reconciliacao/models/leak_guard.py` | Falha o pipeline se alguma feature isolada separar as classes |
| `src/reconciliacao/models/decision.py` | Escolha de limiar sob restrição de precisão e métricas de exceção |
| `src/reconciliacao/models/comparison_stats.py` | Wilcoxon pareado + correção de Holm + tamanho de efeito |
| `src/reconciliacao/models/comparison_trainer.py` | Ajusta os três algoritmos numa partição dada |
| `src/reconciliacao/comparison.py` | Orquestra 10 sementes × 3 algoritmos e escreve os artefatos |
| `run_comparison.py` | CLI fino sobre `reconciliacao.comparison` |

---

### Task 1: Aritmética de retenções

**Files:**
- Create: `src/reconciliacao/simulation/retentions.py`
- Test: `tests/test_simulation/test_retentions.py`

**Interfaces:**
- Consumes: nada.
- Produces:
  - `legal_combos(valor_servicos: float, aliquota_iss: float, rates: dict, csrf_threshold: float) -> dict[str, float]`
  - `matches_any(retencao_pct: float, combos: dict[str, float], tol_pp: float) -> bool`
  - `compatibility_flags(retencao_pct: float, aliquota_iss: float, rates: dict, tol_pp: float) -> dict[str, int]`

**Contexto de desenho.** `legal_combos` é *condicionada* ao limite da CSRF — é usada pelo gerador para
sortear uma retenção legítima e pela definição de *hard negative*. `compatibility_flags` é
**deliberadamente não condicionada**: ela só compara números. Quem aprende a interação entre
`valor_servicos > 5.000` e a legitimidade da CSRF é o modelo, a partir da feature `acima_limite_csrf`.
Essa interação é a fonte esperada de poder discriminante entre os algoritmos — não a codifique nas
features.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_simulation/test_retentions.py
import pytest
from reconciliacao.simulation.retentions import (
    legal_combos,
    matches_any,
    compatibility_flags,
)

RATES = {"irrf": 1.50, "csrf": 4.65, "inss": 11.00}


def test_legal_combos_excludes_csrf_below_threshold():
    combos = legal_combos(4000.0, 3.00, RATES, csrf_threshold=5000.0)
    assert "csrf" not in combos
    assert "irrf_csrf" not in combos
    assert "irrf_csrf_iss" not in combos


def test_legal_combos_includes_csrf_above_threshold():
    combos = legal_combos(6000.0, 3.00, RATES, csrf_threshold=5000.0)
    assert combos["csrf"] == pytest.approx(4.65)
    assert combos["irrf_csrf"] == pytest.approx(6.15)
    assert combos["irrf_csrf_iss"] == pytest.approx(9.15)


def test_legal_combos_always_offers_no_withholding_and_iss():
    combos = legal_combos(1000.0, 2.50, RATES, csrf_threshold=5000.0)
    assert combos["nenhuma"] == 0.0
    assert combos["iss"] == pytest.approx(2.50)
    assert combos["irrf_iss"] == pytest.approx(4.00)
    assert combos["inss"] == pytest.approx(11.00)


def test_matches_any_respects_tolerance():
    combos = {"irrf": 1.50}
    assert matches_any(1.53, combos, tol_pp=0.05) is True
    assert matches_any(1.70, combos, tol_pp=0.05) is False


def test_compatibility_flags_are_not_gated_by_csrf_threshold():
    # 4.65% on a small invoice is not legal, but the flag still fires.
    # The model learns the interaction from acima_limite_csrf; we do not bake it in.
    flags = compatibility_flags(4.65, aliquota_iss=3.00, rates=RATES, tol_pp=0.05)
    assert flags["compat_csrf"] == 1


def test_compatibility_flags_detect_combined_iss():
    flags = compatibility_flags(4.50, aliquota_iss=3.00, rates=RATES, tol_pp=0.05)
    assert flags["compat_combo_iss"] == 1  # irrf 1.50 + iss 3.00
    assert flags["compat_irrf"] == 0


def test_compatibility_flags_all_zero_for_illegitimate_value():
    flags = compatibility_flags(7.30, aliquota_iss=3.00, rates=RATES, tol_pp=0.05)
    assert sum(flags.values()) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_simulation/test_retentions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reconciliacao.simulation.retentions'`

- [ ] **Step 3: Write the implementation**

```python
# src/reconciliacao/simulation/retentions.py
"""Brazilian withholding-tax arithmetic for the comparison experiment.

Rates and the CSRF threshold come from config.yaml. This module only computes
which total withholding percentages are legally plausible for a given invoice,
and whether an observed percentage matches one of them.
"""


def legal_combos(
    valor_servicos: float,
    aliquota_iss: float,
    rates: dict,
    csrf_threshold: float,
) -> dict[str, float]:
    """Total withholding percentages that are legally plausible for this invoice.

    CSRF (PIS + COFINS + CSLL) only applies above csrf_threshold, so combos
    including it are omitted for smaller invoices.
    """
    combos = {
        "nenhuma": 0.0,
        "irrf": rates["irrf"],
        "inss": rates["inss"],
        "iss": aliquota_iss,
        "irrf_iss": rates["irrf"] + aliquota_iss,
    }
    if valor_servicos > csrf_threshold:
        combos["csrf"] = rates["csrf"]
        combos["irrf_csrf"] = rates["irrf"] + rates["csrf"]
        combos["irrf_csrf_iss"] = rates["irrf"] + rates["csrf"] + aliquota_iss
    return combos


def matches_any(retencao_pct: float, combos: dict[str, float], tol_pp: float) -> bool:
    """True when retencao_pct is within tol_pp percentage points of any combo."""
    return any(abs(retencao_pct - valor) <= tol_pp for valor in combos.values())


def compatibility_flags(
    retencao_pct: float,
    aliquota_iss: float,
    rates: dict,
    tol_pp: float,
) -> dict[str, int]:
    """Per-combination compatibility indicators, as 0/1 features.

    Deliberately NOT gated by the CSRF threshold: these are numeric comparisons
    only. The interaction between invoice size and CSRF legitimacy is left for
    the model to learn from the acima_limite_csrf feature -- that interaction is
    the intended source of discriminative power between algorithms.
    """
    def near(valor: float) -> int:
        return int(abs(retencao_pct - valor) <= tol_pp)

    return {
        "compat_irrf": near(rates["irrf"]),
        "compat_csrf": near(rates["csrf"]),
        "compat_irrf_csrf": near(rates["irrf"] + rates["csrf"]),
        "compat_iss": near(aliquota_iss),
        "compat_combo_iss": max(
            near(rates["irrf"] + aliquota_iss),
            near(rates["irrf"] + rates["csrf"] + aliquota_iss),
        ),
        "compat_inss": near(rates["inss"]),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_simulation/test_retentions.py -v`
Expected: PASS — 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/reconciliacao/simulation/retentions.py tests/test_simulation/test_retentions.py
git commit -m "feat: withholding-tax arithmetic for the comparison experiment"
```

---

### Task 2: Gerador com verdade de origem

**Files:**
- Create: `src/reconciliacao/simulation/truth_generator.py`
- Modify: `config.yaml` (adiciona o bloco `comparison` ao final; nada existente muda)
- Modify: `tests/conftest.py` (adiciona a fixture `comparison_config`)
- Test: `tests/test_simulation/test_truth_generator.py`

**Interfaces:**
- Consumes: `legal_combos` da Task 1; `generate_cnpj` de `reconciliacao.utils.cnpj`.
- Produces:
  - `generate_comparison_dataset(cmp_cfg: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
    devolvendo `(df_pagamentos, df_nfse, df_verdade)`.
  - `df_pagamentos`: `id_pagamento, cnpj_fornecedor, data_pagamento, valor_pago, descricao, centro_custo, municipio_centro_custo`
  - `df_nfse`: as mesmas colunas do gerador existente (`nfse_numero, nfse_codigo_verificacao, nfse_data_emissao, nfse_competencia, nfse_cnpj_prestador, nfse_razao_social_prestador, nfse_cnpj_tomador, nfse_razao_social_tomador, nfse_valor_servicos, nfse_valor_iss, nfse_aliquota, nfse_valor_liquido, nfse_item_lista_servico, nfse_discriminacao, nfse_codigo_municipio`)
  - `df_verdade`: `id_pagamento, nfse_numero, tipo_negativo` — `nfse_numero` vazio quando não há par; `tipo_negativo` ∈ `{"", "hard", "soft"}`

**Contexto de desenho.** Uma nota por fornecedor, um pagamento por fornecedor, sempre pareáveis por CNPJ.
O rótulo diz se o par é genuíno. `tipo_negativo` existe só para asserções de teste e análise de erro —
**nunca vira feature**. A propriedade que sustenta o experimento é a sobreposição: nenhuma dimensão de
evidência pode separar as classes sozinha, e é por isso que pares verdadeiros também fogem do prazo e
divergem de município parte do tempo.

- [ ] **Step 1: Add the config block**

Acrescente ao final de `config.yaml`, sem tocar nos blocos existentes:

```yaml
comparison:
  n_seeds: 10
  n_records: 7500
  match_rate: 0.70              # fração de pagamentos com par verdadeiro
  hard_negative_rate: 0.30      # fração DOS NÃO-PARES que são hard negatives (não do total)
  atypical_term_rate: 0.10      # fração dos pares verdadeiros que fogem do prazo do fornecedor
  same_municipality_rate: 0.85  # fração dos pares verdadeiros com centro de custo no município da nota
  payment_terms_days: [0, 15, 30, 45, 60]
  csrf_threshold_brl: 5000.0
  retention_rates:              # em pontos percentuais do valor dos serviços
    irrf: 1.50
    csrf: 4.65
    inss: 11.00
  retention_tolerance_pp: 0.05  # tolerância em PONTOS PERCENTUAIS para compat_* e p/ definir hard
  split: { train: 0.60, val: 0.20, test: 0.20 }
  min_precision: 0.90
  cv_folds: 5
  scoring: average_precision
  leak_guard_max_stump_accuracy: 0.95
```

- [ ] **Step 2: Add the test fixture**

Acrescente ao final de `tests/conftest.py`:

```python
@pytest.fixture
def comparison_config():
    return {
        "n_seeds": 2,
        "n_records": 400,
        "match_rate": 0.70,
        "hard_negative_rate": 0.30,
        "atypical_term_rate": 0.10,
        "same_municipality_rate": 0.85,
        "payment_terms_days": [0, 15, 30, 45, 60],
        "csrf_threshold_brl": 5000.0,
        "retention_rates": {"irrf": 1.50, "csrf": 4.65, "inss": 11.00},
        "retention_tolerance_pp": 0.05,
        "split": {"train": 0.60, "val": 0.20, "test": 0.20},
        "min_precision": 0.90,
        "cv_folds": 2,
        "scoring": "average_precision",
        "leak_guard_max_stump_accuracy": 0.95,
    }
```

- [ ] **Step 3: Write the failing tests**

```python
# tests/test_simulation/test_truth_generator.py
import pandas as pd
import pytest

from reconciliacao.simulation.retentions import legal_combos, matches_any
from reconciliacao.simulation.truth_generator import generate_comparison_dataset


def test_truth_table_has_one_row_per_payment(comparison_config):
    pag, nfse, verdade = generate_comparison_dataset(comparison_config, seed=42)
    assert len(pag) == comparison_config["n_records"]
    assert len(nfse) == comparison_config["n_records"]
    assert len(verdade) == comparison_config["n_records"]
    assert set(verdade["id_pagamento"]) == set(pag["id_pagamento"])


def test_match_rate_is_respected(comparison_config):
    _, _, verdade = generate_comparison_dataset(comparison_config, seed=42)
    matched = (verdade["nfse_numero"] != "").sum()
    expected = comparison_config["n_records"] * comparison_config["match_rate"]
    assert matched == pytest.approx(expected, abs=2)


def test_at_least_thirty_percent_of_negatives_are_hard(comparison_config):
    _, _, verdade = generate_comparison_dataset(comparison_config, seed=42)
    negatives = verdade[verdade["nfse_numero"] == ""]
    hard = (negatives["tipo_negativo"] == "hard").sum()
    assert hard / len(negatives) >= comparison_config["hard_negative_rate"] - 0.01


def test_hard_negatives_have_a_legitimate_value_ratio(comparison_config):
    """A hard negative is precisely one the value ratio cannot resolve."""
    pag, nfse, verdade = generate_comparison_dataset(comparison_config, seed=42)
    merged = pag.merge(
        nfse, left_on="cnpj_fornecedor", right_on="nfse_cnpj_prestador"
    ).merge(verdade, on="id_pagamento", suffixes=("", "_verdade"))
    hard = merged[merged["tipo_negativo"] == "hard"]
    assert len(hard) > 0
    for row in hard.itertuples(index=False):
        retencao = (1 - row.valor_pago / row.nfse_valor_servicos) * 100
        combos = legal_combos(
            row.nfse_valor_servicos,
            row.nfse_aliquota,
            comparison_config["retention_rates"],
            comparison_config["csrf_threshold_brl"],
        )
        assert matches_any(retencao, combos, tol_pp=0.5)


def test_soft_negatives_have_an_illegitimate_value_ratio(comparison_config):
    pag, nfse, verdade = generate_comparison_dataset(comparison_config, seed=42)
    merged = pag.merge(
        nfse, left_on="cnpj_fornecedor", right_on="nfse_cnpj_prestador"
    ).merge(verdade, on="id_pagamento", suffixes=("", "_verdade"))
    soft = merged[merged["tipo_negativo"] == "soft"]
    assert len(soft) > 0
    off_band = 0
    for row in soft.itertuples(index=False):
        retencao = (1 - row.valor_pago / row.nfse_valor_servicos) * 100
        combos = legal_combos(
            row.nfse_valor_servicos,
            row.nfse_aliquota,
            comparison_config["retention_rates"],
            comparison_config["csrf_threshold_brl"],
        )
        if not matches_any(retencao, combos, tol_pp=0.15):
            off_band += 1
    assert off_band == len(soft)


def test_true_pairs_sometimes_break_the_supplier_term(comparison_config):
    """No evidence dimension may separate the classes on its own."""
    pag, nfse, verdade = generate_comparison_dataset(comparison_config, seed=42)
    merged = pag.merge(nfse, left_on="cnpj_fornecedor", right_on="nfse_cnpj_prestador").merge(
        verdade, on="id_pagamento", suffixes=("", "_verdade")
    )
    matched = merged[merged["nfse_numero_verdade"] != ""]
    delta = (
        pd.to_datetime(matched["data_pagamento"])
        - pd.to_datetime(matched["nfse_data_emissao"])
    ).dt.days
    assert not delta.isin(comparison_config["payment_terms_days"]).all()


def test_generation_is_deterministic(comparison_config):
    a = generate_comparison_dataset(comparison_config, seed=7)[0]
    b = generate_comparison_dataset(comparison_config, seed=7)[0]
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_produce_different_data(comparison_config):
    a = generate_comparison_dataset(comparison_config, seed=1)[0]
    b = generate_comparison_dataset(comparison_config, seed=2)[0]
    assert not a["valor_pago"].equals(b["valor_pago"])
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_simulation/test_truth_generator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reconciliacao.simulation.truth_generator'`

- [ ] **Step 5: Write the implementation**

```python
# src/reconciliacao/simulation/truth_generator.py
"""Generate a reconciliation dataset whose label comes from ground truth.

Every payment is paired with exactly one candidate invoice from the same
supplier, and the truth table says whether that pairing is genuine. Labels
therefore never derive from the observable deltas -- the decoupling is by
construction, which is what the original experiment lacked.
"""

import random
from datetime import timedelta

import pandas as pd
from faker import Faker

from reconciliacao.simulation.retentions import legal_combos
from reconciliacao.utils.cnpj import generate_cnpj

_LC116_CODES = ["1.01", "1.02", "1.03", "1.04", "1.05", "7.01", "7.02", "14.01"]
_MUNICIPIOS = ["3550308", "3304557", "4106902", "2304400", "5300108"]
_ACCENTS = str.maketrans("áàâãéêíóôõúüç", "aaaaeeiooouuc")


def _corrupt_text(texto: str, rng: random.Random) -> str:
    """Abbreviate, transpose characters and strip accents, as a payment memo would."""
    out = []
    for palavra in texto.split():
        r = rng.random()
        if r < 0.25 and len(palavra) > 4:
            out.append(palavra[:3])
        elif r < 0.40 and len(palavra) > 3:
            i = rng.randrange(len(palavra) - 1)
            out.append(palavra[:i] + palavra[i + 1] + palavra[i] + palavra[i + 2:])
        else:
            out.append(palavra)
    return " ".join(out).translate(_ACCENTS)


def _illegitimate_retention(
    rng: random.Random, combos: dict[str, float], tol_pp: float
) -> float:
    """A withholding percentage that is clearly outside every legal band."""
    for _ in range(200):
        candidato = rng.uniform(0.0, 20.0)
        if all(abs(candidato - v) > tol_pp * 3 for v in combos.values()):
            return candidato
    return 17.3


def generate_comparison_dataset(
    cmp_cfg: dict, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (df_pagamentos, df_nfse, df_verdade) for one seed."""
    rng = random.Random(seed)
    Faker.seed(seed)
    fake = Faker("pt_BR")

    n = cmp_cfg["n_records"]
    rates = cmp_cfg["retention_rates"]
    tol = cmp_cfg["retention_tolerance_pp"]
    limite_csrf = cmp_cfg["csrf_threshold_brl"]

    cnpjs: list[str] = []
    vistos: set[str] = set()
    while len(vistos) < n:
        c = generate_cnpj(rng)
        if c not in vistos:
            vistos.add(c)
            cnpjs.append(c)

    prazo_fornecedor = {c: rng.choice(cmp_cfg["payment_terms_days"]) for c in cnpjs}
    municipio_fornecedor = {c: rng.choice(_MUNICIPIOS) for c in cnpjs}

    centros = [f"CC-{i:03d}" for i in range(100, 200)]
    municipio_centro = {cc: rng.choice(_MUNICIPIOS) for cc in centros}
    centros_por_municipio: dict[str, list[str]] = {}
    for cc, m in municipio_centro.items():
        centros_por_municipio.setdefault(m, []).append(cc)

    n_match = int(n * cmp_cfg["match_rate"])
    eh_par = [True] * n_match + [False] * (n - n_match)
    rng.shuffle(eh_par)

    n_neg = n - n_match
    n_hard = int(n_neg * cmp_cfg["hard_negative_rate"])
    flags_hard = [True] * n_hard + [False] * (n_neg - n_hard)
    rng.shuffle(flags_hard)
    iter_hard = iter(flags_hard)

    cnpj_tomador = generate_cnpj(rng)
    razao_tomador = fake.company()

    pagamentos, notas, verdade = [], [], []

    for i, cnpj in enumerate(cnpjs):
        valor_servicos = round(rng.uniform(500.0, 50_000.0), 2)
        aliquota = round(rng.uniform(2.0, 5.0), 2)
        emissao = fake.date_between(start_date="-1y", end_date="-2m")
        discriminacao = fake.bs()
        numero = f"{i + 1:06d}"
        valor_iss = round(valor_servicos * aliquota / 100, 2)

        notas.append({
            "nfse_numero": numero,
            "nfse_codigo_verificacao": "".join(
                rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=8)
            ),
            "nfse_data_emissao": emissao.isoformat() + "T00:00:00",
            "nfse_competencia": emissao.replace(day=1).isoformat() + "T00:00:00",
            "nfse_cnpj_prestador": cnpj,
            "nfse_razao_social_prestador": fake.company(),
            "nfse_cnpj_tomador": cnpj_tomador,
            "nfse_razao_social_tomador": razao_tomador,
            "nfse_valor_servicos": valor_servicos,
            "nfse_valor_iss": valor_iss,
            "nfse_aliquota": aliquota,
            "nfse_valor_liquido": round(valor_servicos - valor_iss, 2),
            "nfse_item_lista_servico": rng.choice(_LC116_CODES),
            "nfse_discriminacao": discriminacao,
            "nfse_codigo_municipio": municipio_fornecedor[cnpj],
        })

        combos = legal_combos(valor_servicos, aliquota, rates, limite_csrf)
        no_municipio = centros_por_municipio.get(municipio_fornecedor[cnpj], centros)

        if eh_par[i]:
            retencao = combos[rng.choice(list(combos))]
            prazo = prazo_fornecedor[cnpj]
            if rng.random() < cmp_cfg["atypical_term_rate"]:
                prazo = max(0, prazo + rng.choice([-10, -5, 7, 14, 21]))
            descricao = _corrupt_text(discriminacao, rng)
            if rng.random() < cmp_cfg["same_municipality_rate"]:
                centro = rng.choice(no_municipio)
            else:
                centro = rng.choice(centros)
            tipo_negativo = ""
            nfse_verdadeira = numero
        else:
            hard = next(iter_hard)
            if hard:
                retencao = combos[rng.choice(list(combos))]
            else:
                retencao = _illegitimate_retention(rng, combos, tol)

            dimensoes = rng.sample(["prazo", "texto", "municipio"], k=rng.randint(1, 3))
            prazo = prazo_fornecedor[cnpj]
            if "prazo" in dimensoes:
                prazo = max(0, prazo + rng.choice([-30, -20, 25, 40, 60]))
            descricao = fake.bs() if "texto" in dimensoes else _corrupt_text(discriminacao, rng)
            centro = rng.choice(centros) if "municipio" in dimensoes else rng.choice(no_municipio)
            tipo_negativo = "hard" if hard else "soft"
            nfse_verdadeira = ""

        pagamentos.append({
            "id_pagamento": f"PAG-{i + 1:06d}",
            "cnpj_fornecedor": cnpj,
            "data_pagamento": emissao + timedelta(days=prazo),
            "valor_pago": round(valor_servicos * (1 - retencao / 100), 2),
            "descricao": descricao,
            "centro_custo": centro,
            "municipio_centro_custo": municipio_centro[centro],
        })
        verdade.append({
            "id_pagamento": f"PAG-{i + 1:06d}",
            "nfse_numero": nfse_verdadeira,
            "tipo_negativo": tipo_negativo,
        })

    return pd.DataFrame(pagamentos), pd.DataFrame(notas), pd.DataFrame(verdade)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_simulation/test_truth_generator.py -v`
Expected: PASS — 8 passed

- [ ] **Step 7: Confirm nothing existing broke**

Run: `pytest -q`
Expected: toda a suíte existente continua passando.

- [ ] **Step 8: Commit**

```bash
git add config.yaml tests/conftest.py src/reconciliacao/simulation/truth_generator.py tests/test_simulation/test_truth_generator.py
git commit -m "feat: ground-truth dataset generator for the comparison experiment"
```

---

### Task 3: Rotulagem pela verdade de origem

**Files:**
- Create: `src/reconciliacao/etl/truth_labeler.py`
- Test: `tests/test_etl/test_truth_labeler.py`

**Interfaces:**
- Consumes: os três DataFrames da Task 2.
- Produces: `label_from_truth(df_pag: pd.DataFrame, df_nfse: pd.DataFrame, df_verdade: pd.DataFrame) -> pd.DataFrame`
  — devolve o par pagamento↔nota candidata com a coluna `label` (1 = par verdadeiro, 0 = exceção).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_etl/test_truth_labeler.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_etl/test_truth_labeler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reconciliacao.etl.truth_labeler'`

- [ ] **Step 3: Write the implementation**

```python
# src/reconciliacao/etl/truth_labeler.py
"""Pair each payment with its candidate invoice and label from ground truth.

The label comes from the truth table and from nothing else. No observable
quantity -- date delta, value ratio, text similarity -- takes part in it.
"""

import pandas as pd


def label_from_truth(
    df_pag: pd.DataFrame,
    df_nfse: pd.DataFrame,
    df_verdade: pd.DataFrame,
) -> pd.DataFrame:
    """Join payment to its candidate invoice and label the pairing.

    Returns one row per payment with label 1 when the candidate invoice is the
    one the truth table records, 0 otherwise. tipo_negativo is dropped: it is
    diagnostic metadata, never a feature.
    """
    merged = df_pag.merge(
        df_nfse,
        left_on="cnpj_fornecedor",
        right_on="nfse_cnpj_prestador",
        how="inner",
    )

    truth = df_verdade[["id_pagamento", "nfse_numero"]].rename(
        columns={"nfse_numero": "nfse_verdadeira"}
    )
    merged = merged.merge(truth, on="id_pagamento", how="left")

    candidata = merged["nfse_numero"].fillna("").astype(str)
    verdadeira = merged["nfse_verdadeira"].fillna("").astype(str)
    merged["label"] = ((verdadeira != "") & (candidata == verdadeira)).astype(int)

    return merged.drop(columns=["nfse_verdadeira"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_etl/test_truth_labeler.py -v`
Expected: PASS — 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/reconciliacao/etl/truth_labeler.py tests/test_etl/test_truth_labeler.py
git commit -m "feat: label reconciliation pairs from generator ground truth"
```

---

### Task 4: Features de evidência

**Files:**
- Create: `src/reconciliacao/models/features_v2.py`
- Modify: `docs/superpowers/specs/2026-09-10-comparacao-algoritmos-design.md` (§5.2, registrar o desvio)
- Test: `tests/test_models/test_features_v2.py`

**Interfaces:**
- Consumes: `compatibility_flags` da Task 1; a saída de `label_from_truth` da Task 3.
- Produces:
  - `fit_supplier_terms(df_train: pd.DataFrame) -> tuple[dict[str, float], float]` — `(mediana por CNPJ, mediana global)`
  - `build_features_v2(df: pd.DataFrame, cmp_cfg: dict, supplier_terms: dict[str, float], global_term: float) -> tuple[pd.DataFrame, pd.Series]`

**Colunas produzidas, nesta ordem:** `delta_dias`, `razao_valor`, `retencao_implicita_pct`,
`valor_retido_brl`, `compat_irrf`, `compat_csrf`, `compat_irrf_csrf`, `compat_iss`, `compat_combo_iss`,
`compat_inss`, `acima_limite_csrf`, `desvio_prazo_fornecedor`, `similaridade_descricao`,
`mesmo_municipio`.

**Contexto de desenho.** `fit_supplier_terms` roda **só na partição de treino**. Fornecedores ausentes do
treino recebem a mediana global do treino. É a correção da falha transdutiva que o próprio estudo criticou
em §3.5 do documento de discussão — não a contorne calculando a mediana sobre o conjunto todo.

- [ ] **Step 1: Record the spec deviation**

Em `docs/superpowers/specs/2026-09-10-comparacao-algoritmos-design.md`, substitua a linha da tabela §5.2:

```
| `iss_declarado_consistente` | `valor_iss ≈ valor_servicos × aliquota / 100` |
```

por:

```
| `valor_retido_brl` | `valor_servicos − valor_pago`, em reais |
```

E acrescente logo abaixo da tabela:

```
> **Revisão de 2026-09-10.** `iss_declarado_consistente` foi removida durante o planejamento: era uma
> propriedade interna da nota, não do par pagamento↔nota, e portanto não carregaria sinal sobre a
> genuinidade do pareamento. Entrou no lugar `valor_retido_brl`, evidência de nível de par que preserva
> a interação com o limite de R$ 5.000 da CSRF.
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_models/test_features_v2.py
import pandas as pd
import pytest

from reconciliacao.models.features_v2 import build_features_v2, fit_supplier_terms

CFG = {
    "retention_rates": {"irrf": 1.50, "csrf": 4.65, "inss": 11.00},
    "retention_tolerance_pp": 0.05,
    "csrf_threshold_brl": 5000.0,
}


def _frame():
    return pd.DataFrame({
        "id_pagamento": ["PAG-000001", "PAG-000002", "PAG-000003"],
        "cnpj_fornecedor": ["11222333000181", "11222333000181", "44555666000195"],
        "data_pagamento": pd.to_datetime(["2026-02-14", "2026-03-02", "2026-01-20"]),
        "valor_pago": [9850.00, 5631.00, 970.00],  # 1.50%, 6.15% (irrf+csrf), 3.00% (iss)
        "descricao": ["consultoria de TI", "consultoria de TI", "algo diferente"],
        "centro_custo": ["CC-101", "CC-102", "CC-103"],
        "municipio_centro_custo": ["3550308", "3304557", "3550308"],
        "nfse_numero": ["000001", "000002", "000003"],
        "nfse_data_emissao": ["2026-01-15T00:00:00", "2026-01-31T00:00:00", "2026-01-20T00:00:00"],
        "nfse_valor_servicos": [10000.00, 6000.00, 1000.00],
        "nfse_valor_iss": [500.00, 300.00, 30.00],
        "nfse_aliquota": [5.00, 5.00, 3.00],
        "nfse_discriminacao": ["consultoria de TI", "consultoria de TI", "servico X"],
        "nfse_codigo_municipio": ["3550308", "3550308", "3550308"],
        "label": [1, 1, 0],
    })


def test_retention_and_ratio_are_computed():
    df = _frame()
    terms, global_term = fit_supplier_terms(df)
    X, y = build_features_v2(df, CFG, terms, global_term)
    assert X.loc[0, "retencao_implicita_pct"] == pytest.approx(1.50)
    assert X.loc[0, "razao_valor"] == pytest.approx(0.985)
    assert X.loc[0, "valor_retido_brl"] == pytest.approx(150.00)


def test_compatibility_flag_fires_for_legal_withholding():
    df = _frame()
    terms, global_term = fit_supplier_terms(df)
    X, _ = build_features_v2(df, CFG, terms, global_term)
    assert X.loc[0, "compat_irrf"] == 1          # 1.50%
    assert X.loc[1, "compat_irrf_csrf"] == 1     # 6.15% on a 6000 invoice


def test_csrf_threshold_flag_tracks_invoice_size():
    df = _frame()
    terms, global_term = fit_supplier_terms(df)
    X, _ = build_features_v2(df, CFG, terms, global_term)
    assert X.loc[0, "acima_limite_csrf"] == 1    # 10000 > 5000
    assert X.loc[2, "acima_limite_csrf"] == 0    # 1000 < 5000


def test_municipality_flag_compares_cost_centre_to_invoice():
    df = _frame()
    terms, global_term = fit_supplier_terms(df)
    X, _ = build_features_v2(df, CFG, terms, global_term)
    assert X.loc[0, "mesmo_municipio"] == 1
    assert X.loc[1, "mesmo_municipio"] == 0


def test_supplier_terms_are_fitted_on_training_data_only():
    """A supplier unseen in training falls back to the global training median."""
    df = _frame()
    train = df[df["cnpj_fornecedor"] == "11222333000181"]
    terms, global_term = fit_supplier_terms(train)
    assert "44555666000195" not in terms

    X, _ = build_features_v2(df, CFG, terms, global_term)
    # PAG-000003: delta 0 days, supplier unseen -> deviation from the global median
    assert X.loc[2, "desvio_prazo_fornecedor"] == pytest.approx(0.0 - global_term)


def test_no_feature_is_the_label():
    df = _frame()
    terms, global_term = fit_supplier_terms(df)
    X, y = build_features_v2(df, CFG, terms, global_term)
    assert "label" not in X.columns
    assert "tipo_negativo" not in X.columns
    assert list(y) == [1, 1, 0]


def test_text_similarity_is_higher_for_matching_descriptions():
    df = _frame()
    terms, global_term = fit_supplier_terms(df)
    X, _ = build_features_v2(df, CFG, terms, global_term)
    assert X.loc[0, "similaridade_descricao"] > X.loc[2, "similaridade_descricao"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_models/test_features_v2.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reconciliacao.models.features_v2'`

- [ ] **Step 4: Write the implementation**

```python
# src/reconciliacao/models/features_v2.py
"""Observable evidence about a payment-invoice pairing.

Every feature is computable by someone who does not know whether the pairing is
genuine. Nothing here participates in defining the label.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reconciliacao.simulation.retentions import compatibility_flags

FEATURE_ORDER = [
    "delta_dias",
    "razao_valor",
    "retencao_implicita_pct",
    "valor_retido_brl",
    "compat_irrf",
    "compat_csrf",
    "compat_irrf_csrf",
    "compat_iss",
    "compat_combo_iss",
    "compat_inss",
    "acima_limite_csrf",
    "desvio_prazo_fornecedor",
    "similaridade_descricao",
    "mesmo_municipio",
]


def _delta_dias(df: pd.DataFrame) -> pd.Series:
    pagamento = pd.to_datetime(df["data_pagamento"])
    emissao = pd.to_datetime(df["nfse_data_emissao"])
    return (pagamento - emissao).dt.days.astype(float)


def fit_supplier_terms(df_train: pd.DataFrame) -> tuple[dict[str, float], float]:
    """Median payment term per supplier, fitted on the training partition only."""
    delta = _delta_dias(df_train)
    por_fornecedor = delta.groupby(df_train["cnpj_fornecedor"]).median()
    return por_fornecedor.to_dict(), float(delta.median())


def build_features_v2(
    df: pd.DataFrame,
    cmp_cfg: dict,
    supplier_terms: dict[str, float],
    global_term: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the evidence matrix and the target from a labeled pairing frame."""
    rates = cmp_cfg["retention_rates"]
    tol = cmp_cfg["retention_tolerance_pp"]
    limite = cmp_cfg["csrf_threshold_brl"]

    df = df.reset_index(drop=True)
    feat = pd.DataFrame(index=df.index)

    valor_servicos = df["nfse_valor_servicos"].astype(float)
    valor_pago = df["valor_pago"].astype(float)

    feat["delta_dias"] = _delta_dias(df)
    feat["razao_valor"] = valor_pago / valor_servicos
    feat["retencao_implicita_pct"] = (1.0 - feat["razao_valor"]) * 100.0
    feat["valor_retido_brl"] = valor_servicos - valor_pago

    aliquotas = df["nfse_aliquota"].astype(float)
    flags = [
        compatibility_flags(ret, aliq, rates, tol)
        for ret, aliq in zip(feat["retencao_implicita_pct"], aliquotas)
    ]
    for nome in ["compat_irrf", "compat_csrf", "compat_irrf_csrf",
                 "compat_iss", "compat_combo_iss", "compat_inss"]:
        feat[nome] = [f[nome] for f in flags]

    feat["acima_limite_csrf"] = (valor_servicos > limite).astype(int)

    prazo_esperado = df["cnpj_fornecedor"].map(supplier_terms).fillna(global_term)
    feat["desvio_prazo_fornecedor"] = feat["delta_dias"] - prazo_esperado.astype(float)

    desc_pag = df["descricao"].fillna("").astype(str).tolist()
    desc_nfse = df["nfse_discriminacao"].fillna("").astype(str).tolist()
    try:
        vectorizer = TfidfVectorizer(min_df=1)
        tfidf = vectorizer.fit_transform(desc_pag + desc_nfse)
        n = len(df)
        feat["similaridade_descricao"] = cosine_similarity(tfidf[:n], tfidf[n:]).diagonal()
    except ValueError:
        feat["similaridade_descricao"] = 0.0

    feat["mesmo_municipio"] = (
        df["municipio_centro_custo"].astype(str)
        == df["nfse_codigo_municipio"].astype(str)
    ).astype(int)

    y = df["label"].astype(int)
    return feat[FEATURE_ORDER], y
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_models/test_features_v2.py -v`
Expected: PASS — 7 passed

- [ ] **Step 6: Commit**

```bash
git add src/reconciliacao/models/features_v2.py tests/test_models/test_features_v2.py docs/superpowers/specs/2026-09-10-comparacao-algoritmos-design.md
git commit -m "feat: pair-level evidence features with train-only term fitting"
```

---

### Task 5: Guarda anti-vazamento

**Files:**
- Create: `src/reconciliacao/models/leak_guard.py`
- Test: `tests/test_models/test_leak_guard.py`

**Interfaces:**
- Consumes: `(X, y)` de `build_features_v2`.
- Produces:
  - `class LeakDetected(Exception)`
  - `stump_accuracies(X: pd.DataFrame, y: pd.Series, cv: int = 3, seed: int = 42) -> pd.Series`
  - `assert_no_leak(X: pd.DataFrame, y: pd.Series, max_accuracy: float, cv: int = 3, seed: int = 42) -> pd.Series`

**Contexto de desenho.** Este é o artefato que teria capturado o bug original. Ele converte
"acurácia elevada deve ser tratada como sinal de alarme" de recomendação em verificação executável — a
contribuição 5.5 do documento de discussão. Ele **falha o pipeline**, não emite aviso.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models/test_leak_guard.py
import numpy as np
import pandas as pd
import pytest

from reconciliacao.models.leak_guard import LeakDetected, assert_no_leak, stump_accuracies


def test_detects_a_deliberately_circular_feature():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, size=300))
    X = pd.DataFrame({
        "ruido": rng.normal(size=300),
        "copia_do_rotulo": y.astype(float),
    })
    with pytest.raises(LeakDetected) as exc:
        assert_no_leak(X, y, max_accuracy=0.95)
    assert "copia_do_rotulo" in str(exc.value)


def test_passes_when_no_single_feature_separates():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, size=300))
    X = pd.DataFrame({
        "a": rng.normal(size=300),
        "b": rng.normal(size=300),
    })
    report = assert_no_leak(X, y, max_accuracy=0.95)
    assert (report < 0.95).all()


def test_report_covers_every_feature():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, size=200))
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    report = stump_accuracies(X, y)
    assert set(report.index) == {"a", "b"}


def test_error_names_every_offending_feature():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, size=300))
    X = pd.DataFrame({
        "copia_1": y.astype(float),
        "copia_2": y.astype(float) * 3.0,
        "ruido": rng.normal(size=300),
    })
    with pytest.raises(LeakDetected) as exc:
        assert_no_leak(X, y, max_accuracy=0.95)
    mensagem = str(exc.value)
    assert "copia_1" in mensagem and "copia_2" in mensagem and "ruido" not in mensagem
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models/test_leak_guard.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reconciliacao.models.leak_guard'`

- [ ] **Step 3: Write the implementation**

```python
# src/reconciliacao/models/leak_guard.py
"""Fail the pipeline when any single feature separates the classes.

A label that is a deterministic function of one feature is recoverable by a
depth-1 tree. This guard is the regression test the original experiment lacked:
accuracy above the threshold means the label leaked into the features, not that
the model learned something.
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier


class LeakDetected(Exception):
    """Raised when a single feature predicts the label too well."""


def stump_accuracies(
    X: pd.DataFrame, y: pd.Series, cv: int = 3, seed: int = 42
) -> pd.Series:
    """Cross-validated accuracy of a depth-1 tree fitted on each feature alone."""
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    scores = {}
    for coluna in X.columns:
        stump = DecisionTreeClassifier(max_depth=1, random_state=seed)
        scores[coluna] = float(
            cross_val_score(stump, X[[coluna]], y, cv=folds, scoring="accuracy").mean()
        )
    return pd.Series(scores).sort_values(ascending=False)


def assert_no_leak(
    X: pd.DataFrame,
    y: pd.Series,
    max_accuracy: float,
    cv: int = 3,
    seed: int = 42,
) -> pd.Series:
    """Return the per-feature report, raising LeakDetected if any feature exceeds max_accuracy."""
    report = stump_accuracies(X, y, cv=cv, seed=seed)
    culpadas = report[report > max_accuracy]
    if not culpadas.empty:
        detalhe = ", ".join(f"{nome}={valor:.4f}" for nome, valor in culpadas.items())
        raise LeakDetected(
            f"single-feature accuracy above {max_accuracy}: {detalhe}. "
            "The label is recoverable from a feature in isolation."
        )
    return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models/test_leak_guard.py -v`
Expected: PASS — 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/reconciliacao/models/leak_guard.py tests/test_models/test_leak_guard.py
git commit -m "feat: leak guard failing the pipeline on single-feature separability"
```

---

### Task 6: Limiar de decisão e métricas de exceção

**Files:**
- Create: `src/reconciliacao/models/decision.py`
- Test: `tests/test_models/test_decision.py`

**Interfaces:**
- Consumes: nada dos módulos anteriores.
- Produces:
  - `choose_threshold(y_val: pd.Series, proba_excecao: np.ndarray, min_precision: float) -> float | None`
  - `exception_metrics(y_true: pd.Series, proba_excecao: np.ndarray, threshold: float) -> dict[str, float]`
    com chaves `recall_excecao`, `precisao_excecao`, `taxa_encaminhamento`, `pr_auc_excecao`, `f1_macro`.

**Contexto de desenho.** A classe de exceção é o rótulo **0**. `proba_excecao` é
`estimator.predict_proba(X)[:, 0]`. O limiar é escolhido na validação e aplicado ao teste — nunca
calibrado no teste. `taxa_encaminhamento` é a fração do lote sinalizada como exceção: é o custo
operacional que acompanha o ganho de detecção, e sem ele a métrica primária não sustenta recomendação.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models/test_decision.py
import numpy as np
import pandas as pd
import pytest

from reconciliacao.models.decision import choose_threshold, exception_metrics


def test_chooses_threshold_that_meets_the_precision_floor():
    y = pd.Series([0, 0, 0, 1, 1, 1, 1, 1])
    proba = np.array([0.95, 0.90, 0.85, 0.40, 0.30, 0.20, 0.10, 0.05])
    limiar = choose_threshold(y, proba, min_precision=0.90)
    assert limiar is not None
    metricas = exception_metrics(y, proba, limiar)
    assert metricas["precisao_excecao"] >= 0.90


def test_returns_none_when_the_precision_floor_is_unreachable():
    y = pd.Series([0, 1, 1, 1, 1, 1, 1, 1])
    proba = np.array([0.10, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89])
    assert choose_threshold(y, proba, min_precision=0.90) is None


def test_maximises_recall_subject_to_the_constraint():
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    proba = np.array([0.99, 0.80, 0.70, 0.60, 0.20, 0.15, 0.10, 0.05])
    limiar = choose_threshold(y, proba, min_precision=0.90)
    metricas = exception_metrics(y, proba, limiar)
    # every exception is separable above 0.60 with no false alarm
    assert metricas["recall_excecao"] == pytest.approx(1.0)
    assert metricas["precisao_excecao"] == pytest.approx(1.0)


def test_referral_rate_is_the_flagged_fraction():
    y = pd.Series([0, 0, 1, 1, 1, 1, 1, 1])
    proba = np.array([0.99, 0.98, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    metricas = exception_metrics(y, proba, threshold=0.50)
    assert metricas["taxa_encaminhamento"] == pytest.approx(2 / 8)


def test_metrics_include_pr_auc_and_f1():
    y = pd.Series([0, 0, 1, 1])
    proba = np.array([0.9, 0.8, 0.2, 0.1])
    metricas = exception_metrics(y, proba, threshold=0.5)
    assert 0.0 <= metricas["pr_auc_excecao"] <= 1.0
    assert 0.0 <= metricas["f1_macro"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models/test_decision.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reconciliacao.models.decision'`

- [ ] **Step 3: Write the implementation**

```python
# src/reconciliacao/models/decision.py
"""Decision rule derived from the control objective, not from convention.

The exception class is label 0 -- the payment whose pairing is not genuine.
Selecting a model by F1-macro treats a silenced divergence and a redundant
manual review as equally costly; in reconciliation they are not. The rule here
maximises exception recall subject to a precision floor, and reports the
referral rate that comes with it.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def choose_threshold(
    y_val: pd.Series, proba_excecao: np.ndarray, min_precision: float
) -> float | None:
    """Highest-recall threshold whose exception precision meets the floor.

    Returns None when no threshold reaches min_precision -- the caller must then
    report the algorithm as unable to satisfy the operating constraint.
    """
    es_excecao = (np.asarray(y_val) == 0).astype(int)
    precisao, recall, limiares = precision_recall_curve(es_excecao, proba_excecao)

    # precision_recall_curve returns one more point than thresholds
    melhor_limiar, melhor_recall = None, -1.0
    for p, r, t in zip(precisao[:-1], recall[:-1], limiares):
        if p >= min_precision and r > melhor_recall:
            melhor_limiar, melhor_recall = float(t), float(r)
    return melhor_limiar


def exception_metrics(
    y_true: pd.Series, proba_excecao: np.ndarray, threshold: float
) -> dict[str, float]:
    """Exception-class metrics at a fixed threshold, plus threshold-free context."""
    y_true = np.asarray(y_true)
    es_excecao = (y_true == 0).astype(int)
    sinalizado = (np.asarray(proba_excecao) >= threshold).astype(int)

    y_pred = np.where(sinalizado == 1, 0, 1)

    return {
        "recall_excecao": float(recall_score(es_excecao, sinalizado, zero_division=0)),
        "precisao_excecao": float(precision_score(es_excecao, sinalizado, zero_division=0)),
        "taxa_encaminhamento": float(sinalizado.mean()),
        "pr_auc_excecao": float(average_precision_score(es_excecao, proba_excecao)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models/test_decision.py -v`
Expected: PASS — 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/reconciliacao/models/decision.py tests/test_models/test_decision.py
git commit -m "feat: exception-recall decision rule under a precision floor"
```

---

### Task 7: Comparação estatística pareada

**Files:**
- Create: `src/reconciliacao/models/comparison_stats.py`
- Test: `tests/test_models/test_comparison_stats.py`

**Interfaces:**
- Consumes: um DataFrame com colunas `seed`, `algorithm` e a métrica.
- Produces: `paired_comparisons(per_seed: pd.DataFrame, metric: str) -> pd.DataFrame` com colunas
  `algorithm_a`, `algorithm_b`, `median_diff`, `statistic`, `p_value`, `p_holm`, `significant`.

**Contexto de desenho.** `median_diff` é o tamanho de efeito — a diferença mediana pareada de `a` menos
`b`. Reportá-lo ao lado do p é obrigatório: significância sem magnitude não sustenta recomendação
prática, e com n=10 o teste tem pouco poder para diferenças pequenas.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models/test_comparison_stats.py
import pandas as pd
import pytest

from reconciliacao.models.comparison_stats import paired_comparisons


def _per_seed(valores: dict[str, list[float]]) -> pd.DataFrame:
    linhas = []
    for algoritmo, serie in valores.items():
        for seed, valor in enumerate(serie):
            linhas.append({"seed": seed, "algorithm": algoritmo, "recall_excecao": valor})
    return pd.DataFrame(linhas)


def test_produces_one_row_per_pair():
    df = _per_seed({
        "random_forest": [0.80] * 10,
        "svm": [0.70] * 10,
        "logistic_regression": [0.60] * 10,
    })
    out = paired_comparisons(df, metric="recall_excecao")
    assert len(out) == 3
    assert set(out.columns) == {
        "algorithm_a", "algorithm_b", "median_diff",
        "statistic", "p_value", "p_holm", "significant",
    }


def test_detects_a_consistent_difference():
    df = _per_seed({
        "random_forest": [0.80, 0.81, 0.79, 0.82, 0.80, 0.83, 0.78, 0.81, 0.80, 0.82],
        "svm": [0.70, 0.71, 0.69, 0.72, 0.70, 0.73, 0.68, 0.71, 0.70, 0.72],
        "logistic_regression": [0.60] * 10,
    })
    out = paired_comparisons(df, metric="recall_excecao")
    linha = out[(out["algorithm_a"] == "random_forest") & (out["algorithm_b"] == "svm")].iloc[0]
    assert linha["median_diff"] == pytest.approx(0.10, abs=0.005)
    assert linha["p_holm"] < 0.05
    assert bool(linha["significant"]) is True


def test_reports_no_significance_for_identical_performance():
    df = _per_seed({
        "random_forest": [0.80, 0.70, 0.75, 0.82, 0.68, 0.79, 0.71, 0.77, 0.73, 0.76],
        "svm": [0.80, 0.70, 0.75, 0.82, 0.68, 0.79, 0.71, 0.77, 0.73, 0.76],
        "logistic_regression": [0.60] * 10,
    })
    out = paired_comparisons(df, metric="recall_excecao")
    linha = out[(out["algorithm_a"] == "random_forest") & (out["algorithm_b"] == "svm")].iloc[0]
    assert linha["median_diff"] == pytest.approx(0.0)
    assert bool(linha["significant"]) is False


def test_holm_correction_is_no_smaller_than_the_raw_p():
    df = _per_seed({
        "random_forest": [0.80, 0.81, 0.79, 0.82, 0.80, 0.83, 0.78, 0.81, 0.80, 0.82],
        "svm": [0.70, 0.71, 0.69, 0.72, 0.70, 0.73, 0.68, 0.71, 0.70, 0.72],
        "logistic_regression": [0.60, 0.61, 0.59, 0.62, 0.60, 0.63, 0.58, 0.61, 0.60, 0.62],
    })
    out = paired_comparisons(df, metric="recall_excecao")
    assert (out["p_holm"] >= out["p_value"] - 1e-12).all()
    assert (out["p_holm"] <= 1.0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models/test_comparison_stats.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reconciliacao.models.comparison_stats'`

- [ ] **Step 3: Write the implementation**

```python
# src/reconciliacao/models/comparison_stats.py
"""Paired comparison of algorithms across seeds.

Wilcoxon signed-rank on the paired per-seed metric, with Holm correction for the
three pairwise comparisons. The median paired difference is reported alongside
the p-value: with ten seeds the test has little power for small differences, so
significance without magnitude would not support a recommendation.
"""

from itertools import combinations

import pandas as pd
from scipy.stats import wilcoxon

ALPHA = 0.05


def _holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    m = len(p_values)
    ordenados = sorted(range(m), key=lambda i: p_values[i])
    ajustados = [0.0] * m
    corrente = 0.0
    for posicao, indice in enumerate(ordenados):
        valor = (m - posicao) * p_values[indice]
        corrente = max(corrente, min(valor, 1.0))
        ajustados[indice] = corrente
    return ajustados


def paired_comparisons(per_seed: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pairwise Wilcoxon over seeds, Holm-corrected, with effect size."""
    largo = per_seed.pivot(index="seed", columns="algorithm", values=metric)
    algoritmos = sorted(largo.columns)

    linhas, brutos = [], []
    for a, b in combinations(algoritmos, 2):
        diferenca = largo[a] - largo[b]
        if diferenca.abs().sum() == 0:
            estatistica, p = 0.0, 1.0
        else:
            resultado = wilcoxon(largo[a], largo[b], zero_method="wilcox")
            estatistica, p = float(resultado.statistic), float(resultado.pvalue)
        brutos.append(p)
        linhas.append({
            "algorithm_a": a,
            "algorithm_b": b,
            "median_diff": float(diferenca.median()),
            "statistic": estatistica,
            "p_value": p,
        })

    for linha, p_holm in zip(linhas, _holm(brutos)):
        linha["p_holm"] = p_holm
        linha["significant"] = p_holm < ALPHA

    return pd.DataFrame(linhas)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models/test_comparison_stats.py -v`
Expected: PASS — 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/reconciliacao/models/comparison_stats.py tests/test_models/test_comparison_stats.py
git commit -m "feat: paired Wilcoxon comparison with Holm correction"
```

---

### Task 8: Treinador da comparação

**Files:**
- Create: `src/reconciliacao/models/comparison_trainer.py`
- Test: `tests/test_models/test_comparison_trainer.py`

**Interfaces:**
- Consumes: `get_pipelines` de `reconciliacao.models.trainer` (apenas leitura, o módulo não muda).
- Produces: `fit_algorithms(X_train: pd.DataFrame, y_train: pd.Series, cfg: dict, seed: int) -> dict[str, object]`
  — mapeia nome do algoritmo para o `best_estimator_` já ajustado.

**Contexto de desenho.** Reutilizar `get_pipelines(cfg)` é deliberado: os três algoritmos e suas grades
permanecem os do experimento original, para que a diferença de resultado seja atribuível à representação
dos dados e não a uma mudança no espaço de busca. O que muda é `scoring` (`average_precision`, coerente
com o critério de decisão) e a semente, que varia por execução.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models/test_comparison_trainer.py
import numpy as np
import pandas as pd
import pytest

from reconciliacao.models.comparison_trainer import fit_algorithms


@pytest.fixture
def tiny_cfg(sample_config, comparison_config):
    cfg = dict(sample_config)
    cfg["comparison"] = comparison_config
    return cfg


def _xy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.normal(size=n),
    })
    y = pd.Series((X["a"] + X["b"] > 0).astype(int))
    return X, y


def test_fits_all_three_algorithms(tiny_cfg):
    X, y = _xy()
    modelos = fit_algorithms(X, y, tiny_cfg, seed=42)
    assert set(modelos) == {"random_forest", "svm", "logistic_regression"}


def test_every_estimator_can_produce_probabilities(tiny_cfg):
    X, y = _xy()
    modelos = fit_algorithms(X, y, tiny_cfg, seed=42)
    for nome, modelo in modelos.items():
        proba = modelo.predict_proba(X)
        assert proba.shape == (len(X), 2), nome
        assert np.allclose(proba.sum(axis=1), 1.0), nome


def test_seed_is_threaded_through_to_the_estimators(tiny_cfg):
    X, y = _xy()
    a = fit_algorithms(X, y, tiny_cfg, seed=1)["random_forest"]
    b = fit_algorithms(X, y, tiny_cfg, seed=1)["random_forest"]
    assert np.array_equal(a.predict(X), b.predict(X))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models/test_comparison_trainer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reconciliacao.models.comparison_trainer'`

- [ ] **Step 3: Write the implementation**

```python
# src/reconciliacao/models/comparison_trainer.py
"""Fit the three algorithms on one training partition.

The pipelines and hyperparameter grids are reused verbatim from the original
experiment, so any difference in outcome is attributable to the data
representation rather than to a changed search space. Only the scoring function
and the seed differ: average_precision, to match the decision criterion.
"""

import copy

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from reconciliacao.models.trainer import get_pipelines


def fit_algorithms(
    X_train: pd.DataFrame, y_train: pd.Series, cfg: dict, seed: int
) -> dict[str, object]:
    """Return {algorithm_name: fitted best estimator} for one seed."""
    cmp_cfg = cfg["comparison"]

    cfg_seed = copy.deepcopy(cfg)
    cfg_seed["simulation"] = dict(cfg_seed.get("simulation", {}))
    cfg_seed["simulation"]["random_seed"] = seed

    folds = StratifiedKFold(
        n_splits=cmp_cfg["cv_folds"], shuffle=True, random_state=seed
    )

    modelos = {}
    for nome, (estimator, grade) in get_pipelines(cfg_seed).items():
        busca = GridSearchCV(
            estimator,
            grade,
            cv=folds,
            scoring=cmp_cfg["scoring"],
            n_jobs=-1,
            refit=True,
        )
        busca.fit(X_train, y_train)
        modelos[nome] = busca.best_estimator_
    return modelos
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models/test_comparison_trainer.py -v`
Expected: PASS — 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/reconciliacao/models/comparison_trainer.py tests/test_models/test_comparison_trainer.py
git commit -m "feat: comparison trainer reusing the original pipelines and grids"
```

---

### Task 9: Orquestração e artefatos

**Files:**
- Create: `src/reconciliacao/comparison.py` (orquestração testável)
- Create: `run_comparison.py` (CLI fino)
- Test: `tests/test_comparison.py`

**Interfaces:**
- Consumes: tudo das Tasks 1-8.
- Produces, em `reconciliacao.comparison`:
  - `run_seed(cfg: dict, seed: int) -> tuple[list[dict], pd.Series, dict[str, tuple]]` — devolve `(uma linha por algoritmo, relatório do guarda, {algoritmo: (y_teste, proba_excecao_teste)})`.
  - `run(cfg: dict, n_seeds: int, output_dir: Path) -> pd.DataFrame`

**Por que a lógica não fica no script.** `tests/test_pipeline.py` invoca `run_pipeline.py` por
`subprocess` justamente porque um script na raiz não é importável de forma confiável a partir de
`tests/` (que tem `__init__.py`). Colocar a orquestração dentro do pacote instalado torna o teste um
import comum, e deixa `run_comparison.py` como CLI fino — mais fácil de testar e de reusar no notebook.

**Runtime esperado.** O SVM com núcleo RBF é o gargalo: `SVC` escala com o quadrado do número de
amostras, e são 4.500 linhas de treino × 3 valores de `C` × 5 dobras × 10 sementes. Conte com **dezenas
de minutos**. O sinalizador `--seeds` existe para ensaiar com 2 antes de rodar as 10.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_comparison.py
from pathlib import Path

import pandas as pd
import pytest

from reconciliacao.comparison import run


@pytest.mark.integration
def test_end_to_end_produces_the_decision_artifacts(tmp_path, sample_config, comparison_config):
    cfg = dict(sample_config)
    comparison_config["n_records"] = 400
    comparison_config["cv_folds"] = 2
    cfg["comparison"] = comparison_config

    resumo = run(cfg, n_seeds=2, output_dir=tmp_path)

    assert len(resumo) == 6  # 2 seeds x 3 algorithms
    assert set(resumo["algorithm"]) == {"random_forest", "svm", "logistic_regression"}
    for coluna in ["recall_excecao", "precisao_excecao", "taxa_encaminhamento"]:
        assert coluna in resumo.columns

    for nome in ["per_seed_metrics.csv", "wilcoxon.csv", "decision_summary.csv",
                 "leak_guard_report.csv", "pr_curves.png", "recall_boxplot.png"]:
        assert (tmp_path / nome).exists(), nome


@pytest.mark.integration
def test_threshold_is_chosen_on_validation_not_test(tmp_path, sample_config, comparison_config):
    """The reported test precision may fall below the floor; the validation one may not."""
    cfg = dict(sample_config)
    comparison_config["n_records"] = 400
    comparison_config["cv_folds"] = 2
    cfg["comparison"] = comparison_config

    resumo = run(cfg, n_seeds=1, output_dir=tmp_path)
    aplicaveis = resumo[resumo["threshold"].notna()]
    assert len(aplicaveis) > 0
    assert (aplicaveis["precisao_validacao"] >= cfg["comparison"]["min_precision"] - 1e-9).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_comparison.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reconciliacao.comparison'`

- [ ] **Step 3: Write the implementation**

```python
# src/reconciliacao/comparison.py
"""Answer the first research question: which algorithm best automates reconciliation.

Ten independent datasets, three algorithms, one decision criterion derived from
the control objective -- exception recall under a precision floor. The label
comes from generator ground truth, so the comparison measures the algorithms
rather than the labeling rule.
"""

from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
from sklearn.metrics import PrecisionRecallDisplay  # noqa: E402

from reconciliacao.etl.truth_labeler import label_from_truth
from reconciliacao.models.comparison_stats import paired_comparisons
from reconciliacao.models.comparison_trainer import fit_algorithms
from reconciliacao.models.decision import choose_threshold, exception_metrics
from reconciliacao.models.features_v2 import build_features_v2, fit_supplier_terms
from reconciliacao.models.leak_guard import assert_no_leak
from reconciliacao.simulation.truth_generator import generate_comparison_dataset

PRIMARY_METRIC = "recall_excecao"


def _split_three_ways(df: pd.DataFrame, cmp_cfg: dict, seed: int):
    fracoes = cmp_cfg["split"]
    treino, resto = train_test_split(
        df,
        train_size=fracoes["train"],
        stratify=df["label"],
        random_state=seed,
    )
    proporcao_val = fracoes["val"] / (fracoes["val"] + fracoes["test"])
    validacao, teste = train_test_split(
        resto,
        train_size=proporcao_val,
        stratify=resto["label"],
        random_state=seed,
    )
    return treino, validacao, teste


def run_seed(cfg: dict, seed: int) -> tuple[list[dict], pd.Series, dict[str, tuple]]:
    """Run one seed end to end.

    Returns the per-algorithm metric rows, the leak-guard report, and the test
    partition predictions kept for plotting.
    """
    cmp_cfg = cfg["comparison"]

    df_pag, df_nfse, df_verdade = generate_comparison_dataset(cmp_cfg, seed)
    pares = label_from_truth(df_pag, df_nfse, df_verdade)

    treino, validacao, teste = _split_three_ways(pares, cmp_cfg, seed)
    termos, termo_global = fit_supplier_terms(treino)

    X_tr, y_tr = build_features_v2(treino, cmp_cfg, termos, termo_global)
    X_val, y_val = build_features_v2(validacao, cmp_cfg, termos, termo_global)
    X_te, y_te = build_features_v2(teste, cmp_cfg, termos, termo_global)

    relatorio = assert_no_leak(
        X_tr, y_tr, max_accuracy=cmp_cfg["leak_guard_max_stump_accuracy"], seed=seed
    ).rename(seed)

    modelos = fit_algorithms(X_tr, y_tr, cfg, seed)

    linhas, curvas = [], {}
    for nome, modelo in modelos.items():
        proba_val = modelo.predict_proba(X_val)[:, 0]
        proba_te = modelo.predict_proba(X_te)[:, 0]
        curvas[nome] = ((y_te.to_numpy() == 0).astype(int), proba_te)

        limiar = choose_threshold(y_val, proba_val, cmp_cfg["min_precision"])
        if limiar is None:
            linhas.append({
                "seed": seed, "algorithm": nome, "threshold": None,
                "precisao_validacao": None, "recall_excecao": 0.0,
                "precisao_excecao": 0.0, "taxa_encaminhamento": 0.0,
                "pr_auc_excecao": float("nan"), "f1_macro": float("nan"),
            })
            continue

        metricas_val = exception_metrics(y_val, proba_val, limiar)
        metricas_te = exception_metrics(y_te, proba_te, limiar)
        linhas.append({
            "seed": seed,
            "algorithm": nome,
            "threshold": limiar,
            "precisao_validacao": metricas_val["precisao_excecao"],
            **metricas_te,
        })
    return linhas, relatorio, curvas


def _plot_pr_curves(curvas: dict[str, tuple], destino: Path) -> None:
    """Precision-recall curves for the exception class, from the first seed."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for nome, (es_excecao, proba) in curvas.items():
        PrecisionRecallDisplay.from_predictions(es_excecao, proba, name=nome, ax=ax)
    ax.set_title("Precisão-recall da classe de exceção — primeira semente")
    ax.set_xlabel("Recall da exceção")
    ax.set_ylabel("Precisão da exceção")
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_recall_boxplot(per_seed: pd.DataFrame, destino: Path) -> None:
    """Spread of exception recall across seeds, one box per algorithm."""
    algoritmos = sorted(per_seed["algorithm"].unique())
    dados = [per_seed.loc[per_seed["algorithm"] == a, PRIMARY_METRIC] for a in algoritmos]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(dados, tick_labels=algoritmos)
    ax.set_ylabel("Recall da exceção (precisão ≥ mínimo)")
    ax.set_title(f"Dispersão entre {per_seed['seed'].nunique()} sementes")
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(cfg: dict, n_seeds: int, output_dir: Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    linhas, leak_reports, curvas_primeira = [], [], {}
    for seed in range(n_seeds):
        print(f"[seed {seed + 1}/{n_seeds}] generating, guarding, training...")
        linhas_seed, relatorio, curvas = run_seed(cfg, seed)
        linhas.extend(linhas_seed)
        leak_reports.append(relatorio)
        if seed == 0:
            curvas_primeira = curvas

    per_seed = pd.DataFrame(linhas)
    per_seed.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    pd.concat(leak_reports, axis=1).to_csv(output_dir / "leak_guard_report.csv")

    wilcoxon = paired_comparisons(per_seed, metric=PRIMARY_METRIC)
    wilcoxon.to_csv(output_dir / "wilcoxon.csv", index=False)

    resumo = (
        per_seed.groupby("algorithm")
        .agg(
            recall_excecao_medio=("recall_excecao", "mean"),
            recall_excecao_dp=("recall_excecao", "std"),
            precisao_excecao_media=("precisao_excecao", "mean"),
            taxa_encaminhamento_media=("taxa_encaminhamento", "mean"),
            pr_auc_media=("pr_auc_excecao", "mean"),
        )
        .sort_values("recall_excecao_medio", ascending=False)
    )
    resumo.to_csv(output_dir / "decision_summary.csv")

    _plot_pr_curves(curvas_primeira, output_dir / "pr_curves.png")
    _plot_recall_boxplot(per_seed, output_dir / "recall_boxplot.png")

    print("\n=== Recall de exceção sob precisão >= "
          f"{cfg['comparison']['min_precision']} ===")
    print(resumo.to_string())
    print("\n=== Wilcoxon pareado (Holm) ===")
    print(wilcoxon.to_string(index=False))

    return per_seed
```

E o CLI fino, na raiz do repositório, seguindo o padrão de `run_pipeline.py`:

```python
# run_comparison.py
import argparse
from pathlib import Path

from reconciliacao.comparison import run
from reconciliacao.utils.config import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare RF, SVM and Logistic Regression on the leak-free task."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seeds", type=int, default=None,
                        help="override comparison.n_seeds (use 2 for a dry run)")
    parser.add_argument("--output", default="data/results/comparison")
    args = parser.parse_args()

    configuracao = load_config(args.config)
    total = args.seeds or configuracao["comparison"]["n_seeds"]
    run(configuracao, total, Path(args.output))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_comparison.py -v`
Expected: PASS — 2 passed

- [ ] **Step 5: Run the full suite**

Run: `pytest -q`
Expected: toda a suíte passa, incluindo os testes do experimento original.

- [ ] **Step 6: Dry run with two seeds**

Run: `python run_comparison.py --seeds 2`
Expected: o guarda anti-vazamento **não** dispara; a tabela de resumo é impressa.

Se `LeakDetected` for levantada, alguma feature está separando as classes sozinha — leia
`leak_guard_report.csv` e corrija a sobreposição no gerador (Task 2), nunca removendo a feature.

- [ ] **Step 7: Commit**

```bash
git add src/reconciliacao/comparison.py run_comparison.py tests/test_comparison.py
git commit -m "feat: end-to-end algorithm comparison over ten seeds"
```

---

### Task 10: Execução final e redação dos resultados

**Files:**
- Modify: `docs/discussao-limitacoes-contribuicoes.md` (nova seção; atualizar limitações e agenda)
- Modify: `README.md` (documentar `run_comparison.py`)

**Interfaces:**
- Consumes: os artefatos em `data/results/comparison/`.
- Produces: texto. Nenhuma interface de código.

- [ ] **Step 1: Run the full experiment**

Run: `python run_comparison.py`
Expected: 10 sementes concluídas; `data/results/comparison/` populado.

- [ ] **Step 2: Check the success criteria from spec §13**

Verifique, contra `decision_summary.csv` e `wilcoxon.csv`:

1. O guarda anti-vazamento aprovou todas as sementes.
2. Ao menos 30% dos não-pares são *hard negatives* (já coberto pelos testes da Task 2).
3. O melhor recall de exceção está **abaixo de 0,98** e **acima do acaso**.
4. O Wilcoxon produziu decisão explícita para cada par.

Se o critério 3 falhar por excesso de facilidade (recall ≥ 0,98), aumente `hard_negative_rate` para 0,50 e
repita. **Não ajuste `match_rate`.** Registre no texto qual valor foi usado e por quê — calibrar em
silêncio até aparecer um vencedor é a falha que este trabalho denuncia.

- [ ] **Step 3: Write the results section**

Acrescente a `docs/discussao-limitacoes-contribuicoes.md` uma seção "Experimento de comparação válida",
com a tabela de `decision_summary.csv`, a de `wilcoxon.csv`, e a conclusão no formato do spec §13:

> Sob rotulagem independente das features e critério derivado do objetivo de controle, o algoritmo *X*
> detecta *N* pontos percentuais a mais de divergências que o *Y* mantendo precisão de 0,90 (Wilcoxon
> pareado, *p* = …, 10 execuções), ao custo de encaminhar *M*% do lote para revisão manual.

Se o resultado for empate, redija-o como equivalência prática — é conclusão legítima, e a recomendação
passa a ser decidir por interpretabilidade e custo.

- [ ] **Step 4: Update limitations and agenda**

No mesmo documento:

- limitações **4** (ausência de teste de significância), **7** (semente única), **12** (ausência de ruído
  textual) e **20** (classe parcial arbitrária) passam a "resolvida no experimento de comparação";
- limitações **1**, **2** e **23** ganham a nota de que o novo experimento não as herda;
- itens **3**, **4**, **6**, **8** e **9** da agenda passam a executados;
- acrescente às contribuições práticas o guarda anti-vazamento como artefato reutilizável.

- [ ] **Step 5: Document the runner in the README**

Acrescente a `README.md`, após a seção "Running the Pipeline":

````markdown
### Algorithm Comparison (leak-free)

```bash
python run_comparison.py --seeds 2    # dry run
python run_comparison.py              # full: 10 seeds
```

Labels come from generator ground truth rather than from a threshold rule, so the comparison measures
the algorithms and not the labeling. A leak guard fails the run if any single feature separates the
classes. Selection is by exception recall under precision >= 0.90, compared across seeds with a paired
Wilcoxon test.
````

- [ ] **Step 6: Commit**

```bash
git add docs/discussao-limitacoes-contribuicoes.md README.md
git commit -m "docs: results of the leak-free algorithm comparison"
```

---

## Notas de execução

**Ordem.** As tasks 1-8 são sequencialmente dependentes por interface. A 9 depende de todas. A 10 depende
de rodar a 9.

**Se o guarda anti-vazamento disparar**, a correção é sempre no gerador (Task 2), aumentando a
sobreposição da dimensão acusada. Remover a feature acusada resolveria o sintoma e destruiria o
experimento — foi exatamente o que a configuração `no_leak` do estudo de ablação mostrou.

**Se os três algoritmos empatarem**, isso é resultado publicável, não falha. Antes de concluir empate,
confirme pelo critério 3 do spec §13 que a tarefa não ficou fácil demais.
