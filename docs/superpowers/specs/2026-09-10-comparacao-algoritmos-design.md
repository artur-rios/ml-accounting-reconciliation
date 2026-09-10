---
name: comparacao-algoritmos-design
description: Design spec for the corrective experiment that gives the Random Forest / SVM / Logistic Regression comparison real discriminative power — leak-free labeling, genuine signal from Brazilian tax withholding rules, and a control-derived decision criterion
metadata:
  type: project
---

# Design: Experimento Complementar de Comparação entre Algoritmos

**Date:** 2026-09-10
**Project:** MBA TCC — Applied Research
**Status:** Design aprovado, aguardando plano de implementação
**Antecedente:** [`2026-06-14-reconciliacao-financeira-ml-design.md`](2026-06-14-reconciliacao-financeira-ml-design.md)

---

## 1. Motivação

O experimento principal e o estudo de ablação (`run_ablation.py`,
`docs/discussao-limitacoes-contribuicoes.md`) estabeleceram que a comparação original entre Random
Forest, SVM e Regressão Logística não tem poder discriminante:

- O rótulo é função determinística de duas das sete variáveis preditoras. A Random Forest atinge
  F1-macro 1,0000 com desvio-padrão de validação cruzada nulo.
- A diferença de 9,0 pontos percentuais entre a floresta e os modelos de fronteira suave era
  integralmente artefato de escala: corrigida a codificação por sentinela, os três algoritmos
  empatam em 1,0000.
- Removido o vazamento, os modelos detectam 203/203 pagamentos sem nota fiscal e 1/251 divergências
  de data ou valor. Não resta sinal aprendível na base.

Este experimento **complementa** o trabalho: o achado do vazamento permanece como resultado central, e
esta etapa fornece a validação corretiva que finalmente responde à primeira pergunta de pesquisa.
Narrativa da monografia: diagnostiquei, corrigi, respondi.

---

## 2. Princípio de desenho

> **O rótulo vem do gerador. As features são observações. Nenhuma feature participa da definição do
> rótulo.**

O desacoplamento é obtido **por construção**, não por remoção de variáveis. O gerador conhece a verdade
— qual pagamento corresponde a qual nota — e essa verdade é persistida. As features são reconstruídas a
partir do que um analista de conciliação observaria, sem acesso a essa verdade.

---

## 3. Arquitetura

```
config.yaml  (novo bloco `comparison`)
    │
    ▼
simulation/truth_generator.py
    ├─► data/raw/comparison/pagamentos.xlsx
    ├─► data/raw/comparison/nfse/*.xml
    └─► data/raw/comparison/verdade.csv      # id_pagamento → nfse_numero | vazio
    │
    ▼
etl/truth_labeler.py        # junta pagamento ↔ nota candidata; rótulo vem de verdade.csv
    │
    ▼
models/features_v2.py       # evidência observável apenas
    │
    ▼
models/leak_guard.py        # FALHA o pipeline se alguma feature isolada separar as classes
    │
    ▼
run_comparison.py           # 10 sementes × 3 algoritmos, seleção de limiar, Wilcoxon
    │
    ▼
data/results/comparison/
```

Os módulos existentes (`simulation/excel_generator.py`, `simulation/xml_generator.py`,
`etl/labeler.py`, `models/features.py`) **não são modificados**. O experimento original precisa
continuar reprodutível, porque é ele que sustenta o achado do vazamento.

---

## 4. Geração de dados

### 4.1 Verdade de origem

`xml_generator.py:26` já calcula `conciliated_indices` e descarta a informação. O novo gerador
persiste o pareamento verdadeiro em `verdade.csv`:

| coluna | conteúdo |
| --- | --- |
| `id_pagamento` | chave do pagamento |
| `nfse_numero` | número da nota verdadeiramente correspondente, ou vazio |
| `tipo_negativo` | vazio para pares verdadeiros; `hard` ou `soft` para não-pares |

`tipo_negativo` existe apenas para asserções de teste e análise de erro. **Não é feature.**

### 4.2 Pareamentos legítimos não são idênticos

Para um par verdadeiro, o pagamento é derivado da nota por transformação legal, não por cópia.

**Prazo de pagamento.** `data_pagamento = data_emissao + prazo`, com
`prazo ∈ {0, 15, 30, 45, 60}` dias corridos. O prazo é sorteado **uma vez por fornecedor** (CNPJ) e não
por registro, de modo que cada fornecedor tenha um prazo característico — estrutura latente que o modelo
pode aprender a partir do histórico. Um pagamento verdadeiro ocasionalmente foge do prazo típico do
fornecedor (ver 4.4).

> **Revisão de 2026-09-10 (commit `4277230`).** A primeira implementação emitia **um pagamento por
> fornecedor**, o que tornava esta afirmação falsa: sem vários pagamentos, não há histórico, a mediana por
> fornecedor é o próprio valor da linha e `desvio_prazo_fornecedor` fica identicamente zero na partição de
> treino. Isso produziu um artefato de escala que colapsou o núcleo RBF e derrubou o SVM de 0,80 para 0,22
> de recall de exceção — o mesmo tipo de defeito que este experimento existe para evitar. O gerador passou
> a sortear fornecedores de um pool (`payments_per_supplier: 5`), e o pagamento carrega referência
> explícita à nota candidata, já que a junção por CNPJ deixou de ser 1:1. Ver §4.6 do documento de
> discussão.

**Retenções tributárias.** `valor_pago = valor_servicos × (1 − Σ retenções)`, com as retenções sorteadas
de um conjunto de combinações plausíveis:

| Combinação | Retenção total | Condição |
| --- | --- | --- |
| nenhuma | 0% | — |
| IRRF | 1,50% | — |
| CSRF (PIS+COFINS+CSLL) | 4,65% | `valor_servicos > 5.000` |
| IRRF + CSRF | 6,15% | `valor_servicos > 5.000` |
| ISS retido | `aliquota` (2–5%) | — |
| IRRF + ISS retido | `1,50 + aliquota` | — |
| IRRF + CSRF + ISS retido | `6,15 + aliquota` | `valor_servicos > 5.000` |
| INSS | 11,00% | — |

Somado a arredondamento de centavos. O resultado é um conjunto **discreto** de deltas legítimos, ao
passo que as divergências são sorteadas de faixa **contínua** sobreposta a ele.

> **Verificação obrigatória antes da monografia.** As alíquotas e o limite de R$ 5.000,00 devem ser
> conferidos contra a legislação vigente (IN RFB 1.234/2012 e alterações; Lei 10.833/2003 arts. 30-31;
> Lei 8.212/1991 art. 31; LC 116/2003 art. 6º) antes de qualquer afirmação normativa no texto. Para o
> experimento, são parâmetros de `config.yaml` — sua exatidão jurídica não afeta a validade metodológica,
> mas afeta a defensabilidade da narrativa de domínio.

### 4.3 O que isso cria

`delta_valor_pct = 4,65%` deixa de significar "não conciliado" e passa a significar "compatível com
retenção de PIS/COFINS/CSLL". Como essa retenção só incide acima de R$ 5.000, surge uma **interação
condicional** genuína entre `valor_servicos` e o delta legítimo: uma fronteira não linear sobre
combinações discretas de alíquotas, condicionada a um limiar em outra variável.

É precisamente o tipo de estrutura em que particionamento axial (floresta), fronteira de margem com
núcleo RBF (SVM) e fronteira linear (regressão logística) têm capacidades diferentes. **É daí que se
espera poder discriminante.**

### 4.4 Não-pares e *hard negatives*

**Requisito explícito:** ao menos **30% dos não-pares** devem ser *hard negatives* — razão de valor cuja
retenção implícita cai dentro de ±0,05 p.p. de uma combinação legal. Esses casos não são resolvíveis pela
razão de valor e exigem outra evidência: ISS declarado inconsistente, prazo atípico para o fornecedor,
similaridade textual baixa, município divergente.

**Requisito de sobreposição, igualmente obrigatório:** nenhuma dimensão de evidência pode separar as
classes sozinha. Se todo *hard negative* tivesse exatamente uma dimensão corrompida e todo par verdadeiro
nenhuma, essa dimensão viraria a nova regra determinística e reproduziríamos o problema original. Portanto:

- pares verdadeiros ocasionalmente fogem do prazo típico do fornecedor — pagamento adiantado ou atrasado
  legítimo, na fração `atypical_term_rate`;
- pares verdadeiros ocasionalmente têm ISS declarado com divergência de centavos por arredondamento;
- *hard negatives* corrompem um subconjunto **sorteado** das dimensões, não uma fixa.

Essa propriedade é verificada pelo guarda anti-vazamento (§6), que é o mecanismo de execução do
requisito, não apenas uma checagem de sanidade.

### 4.5 Ruído textual

A `descricao` do pagamento passa a ser derivada da `discriminacao` da nota por corrupção: abreviação de
palavras, remoção de acentuação, truncamento e troca de caracteres adjacentes no teclado. Para não-pares,
texto não relacionado.

Com isso `similaridade_descricao` deixa de ser ruído por construção — limitação 12 do documento de
discussão — e passa a ser evidência genuína, porém imperfeita.

---

## 5. Rotulagem e features

### 5.1 Rótulo

Binário: `1` = par verdadeiro, `0` = não. Origem exclusiva: `verdade.csv`.

A classe "parcialmente conciliado" é **removida**. Era convenção arbitrária (limitação 20) e conflita com
o critério de recall de exceção, que exige uma classe de exceção bem definida.

### 5.2 Features (evidência observável)

| Feature | Derivação |
| --- | --- |
| `delta_dias` | `data_pagamento − data_emissao`, em dias corridos (com sinal) |
| `razao_valor` | `valor_pago / valor_servicos` |
| `retencao_implicita_pct` | `(1 − razao_valor) × 100` |
| `compat_irrf` | retenção implícita ≈ 1,50 (±`retention_tolerance_pp`) |
| `compat_csrf` | retenção implícita ≈ 4,65 |
| `compat_irrf_csrf` | retenção implícita ≈ 6,15 |
| `compat_iss` | retenção implícita ≈ `aliquota` declarada |
| `compat_combo_iss` | retenção implícita ≈ `1,50 + aliquota` ou `6,15 + aliquota` |
| `compat_inss` | retenção implícita ≈ 11,00 |
| `acima_limite_csrf` | `valor_servicos > csrf_threshold_brl` |
| `valor_retido_brl` | `valor_servicos − valor_pago`, em reais |
| `desvio_prazo_fornecedor` | `delta_dias − prazo mediano histórico do fornecedor` |
| `similaridade_descricao` | cosseno TF-IDF entre descrição e discriminação |
| `mesmo_municipio` | código IBGE do município da nota igual ao do tomador |

> **Revisão de 2026-09-10.** `iss_declarado_consistente` foi removida durante o planejamento: era uma
> propriedade interna da nota, não do par pagamento↔nota, e portanto não carregaria sinal sobre a
> genuinidade do pareamento. Entrou no lugar `valor_retido_brl`, evidência de nível de par que preserva
> a interação com o limite de R$ 5.000 da CSRF.

O gerador atual não emite município do tomador — apenas `nfse_codigo_municipio` do prestador
(`xml_generator.py:15`). O `truth_generator` passa a atribuir um município ao tomador (a empresa
pagadora) para que `mesmo_municipio` seja computável.

Todas computáveis sem acesso ao rótulo. As features `compat_*` codificam conhecimento de domínio a partir
de campos observáveis (`valor_pago`, `valor_servicos`, `aliquota`) — um não-par pode ser coincidentemente
compatível, e é exatamente isso que define um *hard negative*.

**`desvio_prazo_fornecedor` exige histórico.** A mediana do fornecedor é calculada **apenas sobre a
partição de treino** e aplicada às demais. Fornecedores ausentes do treino recebem a mediana global do
treino. Isso evita vazamento transdutivo — falha que o próprio estudo criticou em §3.5 do documento de
discussão.

---

## 6. Guarda anti-vazamento

`models/leak_guard.py` recebe `(X, y)` e, para cada coluna isoladamente, ajusta uma árvore de decisão de
profundidade 1 e mede acurácia em validação cruzada. **Falha o pipeline** se qualquer feature sozinha
ultrapassar `leak_guard_max_stump_accuracy` (padrão 0,95).

É a regressão que teria capturado o bug original. Roda antes do treino, no `run_comparison.py` e como
teste automatizado sobre uma base gerada com semente fixa.

Operacionaliza a contribuição 5.5 do documento de discussão: converte "acurácia elevada deve ser tratada
como sinal de alarme" de recomendação em verificação executável.

---

## 7. Protocolo de avaliação

### 7.1 Critério de decisão

**Métrica primária:** recall da classe de exceção (rótulo 0) sob **precisão ≥ 0,90** nessa mesma classe.

O limiar de decisão que satisfaz a restrição de precisão é escolhido na **partição de validação** e
aplicado à de teste. Nunca calibrado no teste.

Reporta-se junto a **taxa de encaminhamento** — fração do lote classificada como exceção e portanto
enviada à revisão manual. É o custo operacional que acompanha o ganho de detecção, e sem ele a métrica
primária não sustenta recomendação prática.

**Métricas secundárias:** PR-AUC (`average_precision`), F1-macro, matriz de confusão completa.

### 7.2 Partição e ajuste

Por semente: 60% treino / 20% validação / 20% teste, estratificado.
Hiperparâmetros por `GridSearchCV` de 5 dobras **dentro do treino**, com `scoring = average_precision` —
não F1-macro, coerente com o critério e com a crítica de Saito e Rehmsmeier já citada no documento.

Os três algoritmos e suas grades permanecem os do experimento original, para que a comparação seja
atribuível à representação dos dados e não a uma mudança de espaço de busca.

### 7.3 Repetição e comparação estatística

10 sementes geram 10 bases independentes. Cada algoritmo roda nas 10, produzindo 10 valores pareados por
métrica.

Comparação por **Wilcoxon de postos sinalizados pareado** entre os três pares de algoritmos, com correção
de **Holm** para as 3 comparações múltiplas. Reporta-se p-valor **e tamanho de efeito** — diferença
mediana em pontos percentuais de recall, com intervalo — porque significância sem magnitude não sustenta
recomendação.

Resolve as limitações 4 (ausência de teste de significância) e 7 (semente única).

---

## 8. Saídas

```
data/results/comparison/
├── per_seed_metrics.csv      # 30 linhas: 10 sementes × 3 algoritmos
├── wilcoxon.csv              # 3 comparações: p bruto, p corrigido (Holm), tamanho de efeito
├── decision_summary.csv      # tabela final que responde à pergunta de pesquisa
├── pr_curves.png             # curvas precisão-recall médias por algoritmo
├── recall_boxplot.png        # dispersão do recall de exceção entre as 10 sementes
└── leak_guard_report.csv     # acurácia do toco por feature
```

---

## 9. Configuração

Novo bloco em `config.yaml`, sem alterar os existentes:

```yaml
comparison:
  n_seeds: 10
  n_records: 7500
  match_rate: 0.70              # fração de pagamentos com par verdadeiro
  hard_negative_rate: 0.30      # fração DOS NÃO-PARES que são hard negatives (não do total)
  atypical_term_rate: 0.10      # fração dos pares verdadeiros que fogem do prazo do fornecedor
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

---

## 10. Testes

Seguindo a suíte existente (`tests/test_simulation`, `test_etl`, `test_models`):

| Teste | Asserção |
| --- | --- |
| `test_truth_generator.py` | `verdade.csv` tem uma linha por pagamento; `match_rate` respeitada dentro de tolerância amostral |
| `test_truth_generator.py` | ao menos 30% dos não-pares são `hard` pela definição de ±0,05 p.p. |
| `test_truth_generator.py` | prazo é constante por CNPJ, com fração declarada de exceções legítimas |
| `test_truth_labeler.py` | rótulo depende exclusivamente de `verdade.csv`; nenhuma coluna de delta o influencia |
| `test_features_v2.py` | mediana de prazo por fornecedor calculada só no treino |
| `test_leak_guard.py` | detecta vazamento em base construída com feature deliberadamente circular |
| `test_leak_guard.py` | aprova a base gerada pelo `truth_generator` com semente fixa |
| `test_comparison.py` | limiar escolhido na validação, nunca no teste (integração, marcada `integration`) |

---

## 11. Fora de escopo

Deliberadamente excluídos, permanecendo na agenda de pesquisa futura:

- **Pareamento N:M e geração de candidatos (*blocking*).** Muda o objeto de estudo do registro para o par
  candidato e exigiria reescrever metodologia e avaliação. É o caminho mais fiel a Fellegi-Sunter e
  continua sendo a extensão natural do trabalho.
- **Custo monetário explícito por tipo de erro.** Exigiria justificar valores de custo sem base empírica.
- **Cenário ternário.** Removido por conflitar com o critério de recall de exceção.
- **Validação temporal.** A dimensão temporal existe, mas tratá-la corretamente exigiria partição por
  período e análise de deriva — escopo próprio.

---

## 12. Riscos

| Risco | Mitigação |
| --- | --- |
| **A base fica fácil demais** e os três algoritmos voltam a empatar em ~1,0 | `hard_negative_rate` em `config.yaml`; guarda anti-vazamento falha o pipeline; critério de sucesso §13 exige recall < 0,98 |
| **A base fica difícil demais** e os três empatam perto do acaso | `hard_negative_rate` é ajustável; e o empate continua sendo resposta válida — ver abaixo |
| **Alíquotas juridicamente imprecisas** enfraquecem a narrativa de domínio | Parâmetros em `config.yaml`; verificação normativa marcada como obrigatória em §4.2 |
| **Wilcoxon com n=10 tem baixo poder** para diferenças pequenas | Reportar tamanho de efeito junto ao p; declarar equivalência prática quando a diferença mediana for desprezível |

**Sobre o empate.** Se os três algoritmos não diferirem significativamente, isso **não é falha do
experimento**. "Não há diferença estatisticamente significativa; a escolha deve ser decidida por
interpretabilidade e custo operacional" é conclusão legítima, acionável e coerente com a literatura de
*No Free Lunch* já citada. O experimento é desenhado para poder chegar a essa conclusão sem que ela
represente um resultado negativo.

---

## 13. Critérios de sucesso

O experimento é considerado bem-sucedido quando:

1. O guarda anti-vazamento aprova a base: nenhuma feature isolada ultrapassa 0,95 de acurácia.
2. Ao menos 30% dos não-pares são *hard negatives* pela definição de §4.4.
3. O melhor recall de exceção sob precisão ≥ 0,90 fica **abaixo de 0,98** — evidência de que a tarefa não
   é trivial — e **acima do acaso**, evidência de que há sinal.
4. O teste de Wilcoxon produz uma decisão explícita para cada par de algoritmos: diferença significativa
   com tamanho de efeito, ou equivalência prática declarada.
5. A conclusão pode ser redigida na forma:

   > Sob rotulagem independente das features e critério derivado do objetivo de controle, o algoritmo *X*
   > detecta *N* pontos percentuais a mais de divergências que o *Y* mantendo precisão de 0,90 (Wilcoxon
   > pareado, *p* = …, 10 execuções), ao custo de encaminhar *M*% do lote para revisão manual.

---

## 14. Impacto no documento de discussão

Ao final, `docs/discussao-limitacoes-contribuicoes.md` recebe:

- nova seção respondendo à primeira pergunta de pesquisa com o resultado válido;
- limitações 4, 7, 12 e 20 marcadas como resolvidas, com referência ao novo experimento;
- itens 3, 4, 6, 8 e 9 da agenda de pesquisa marcados como executados;
- contribuição prática adicional: o guarda anti-vazamento como artefato reutilizável.
