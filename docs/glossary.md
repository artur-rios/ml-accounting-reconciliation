# Glossary · Glossário

Terms used throughout this repository — in the code, in `config.yaml`, in the result artifacts and in the
write-up — explained in English and Brazilian Portuguese. Each entry says what the term means in general
and what it means *here*, with references at the end of the document.

Termos usados neste repositório — no código, no `config.yaml`, nos artefatos de resultado e no texto da
monografia — explicados em inglês e em português do Brasil. Cada verbete diz o que o termo significa em
geral e o que significa *aqui*, com as referências ao final do documento.

> **Note on references · Nota sobre as referências.** The works cited are real and standard in their
> fields, but confirm the full bibliographic record in an indexed database before citing them in the
> monograph, and format them per ABNT NBR 6023. · As obras citadas são reais e de referência em suas
> áreas, mas confirme o registro bibliográfico completo em base indexada antes de citá-las na monografia,
> e formate-as conforme a ABNT NBR 6023.

## Contents · Sumário

1. [Experimental methodology · Metodologia experimental](#1-experimental-methodology--metodologia-experimental)
2. [Data representation · Representação dos dados](#2-data-representation--representação-dos-dados)
3. [Models and training · Modelos e treinamento](#3-models-and-training--modelos-e-treinamento)
4. [Evaluation and decision · Avaliação e decisão](#4-evaluation-and-decision--avaliação-e-decisão)
5. [Statistical comparison · Comparação estatística](#5-statistical-comparison--comparação-estatística)
6. [Accounting and tax domain · Domínio contábil e tributário](#6-accounting-and-tax-domain--domínio-contábil-e-tributário)
7. [References · Referências](#7-references--referências)

### Term index · Índice de termos

| English | Português | Section · Seção |
| --- | --- | --- |
| Ablation | Ablação | 1 |
| Circularity | Circularidade | 1 |
| Data leakage / label leakage | Vazamento de dados / vazamento de rótulo | 1 |
| Decision stump | Toco de decisão | 1 |
| Determinism / hash seed | Determinismo / semente de *hash* | 1 |
| Ground truth | Verdade de referência (*ground truth*) | 1 |
| Indirect leakage / proxy | Vazamento indireto / *proxy* | 1 |
| Leak guard | Guarda anti-vazamento | 1 |
| Random seed | Semente aleatória | 1 |
| Synthetic data | Dados sintéticos | 1 |
| Transductive leakage | Vazamento transdutivo | 1 |
| Winner's curse / selection bias | Maldição do vencedor / viés de seleção | 1 |
| Affine redundancy | Redundância afim | 2 |
| Feature | Variável / atributo (*feature*) | 2 |
| Imputation | Imputação | 2 |
| Missing indicator | Indicador de ausência | 2 |
| Scale artifact | Artefato de escala | 2 |
| Sentinel value | Valor-sentinela | 2 |
| Standardization | Padronização | 2 |
| TF-IDF / cosine similarity | TF-IDF / similaridade do cosseno | 2 |
| Class weighting | Ponderação de classes | 3 |
| Cross-validation (stratified *k*-fold) | Validação cruzada (*k* dobras estratificadas) | 3 |
| Feature importance | Importância de variáveis | 3 |
| Grid search / hyperparameter | Busca em grade / hiperparâmetro | 3 |
| Logistic Regression | Regressão Logística | 3 |
| Random Forest | Floresta Aleatória (*Random Forest*) | 3 |
| Support Vector Machine (RBF kernel) | Máquina de Vetores de Suporte (núcleo RBF) | 3 |
| Train / validation / test split | Partição treino / validação / teste | 3 |
| Trivial (majority-class) classifier | Classificador trivial (classe majoritária) | 3 |
| Confusion matrix | Matriz de confusão | 4 |
| Decision threshold / operating point | Limiar de decisão / ponto de operação | 4 |
| Exception class | Classe de exceção | 4 |
| Exception recall | Recall da exceção | 4 |
| F1-macro | F1-macro | 4 |
| Precision / recall | Precisão / revocação (*recall*) | 4 |
| Precision floor | Piso de precisão | 4 |
| PR curve / PR-AUC / average precision | Curva PR / PR-AUC / precisão média | 4 |
| Referral rate | Taxa de encaminhamento | 4 |
| ROC curve / ROC-AUC | Curva ROC / ROC-AUC | 4 |
| Effect size | Tamanho de efeito | 5 |
| Friedman test (omnibus) | Teste de Friedman (omnibus) | 5 |
| Holm correction | Correção de Holm | 5 |
| Post-hoc test | Teste *post-hoc* | 5 |
| Wilcoxon signed-rank test | Teste dos postos sinalizados de Wilcoxon | 5 |
| CNPJ | CNPJ | 6 |
| Conciliated / partially / not conciliated | Conciliado / parcialmente / não conciliado | 6 |
| CSRF threshold | Limite da CSRF | 6 |
| ETL | ETL (extração, transformação e carga) | 6 |
| Exact and fuzzy scenarios | Cenários *exact* e *fuzzy* | 6 |
| Financial reconciliation | Conciliação financeira | 6 |
| Hard and soft negatives | Negativos difíceis e fáceis | 6 |
| Internal control / audit evidence | Controle interno / evidência de auditoria | 6 |
| ISS | ISS | 6 |
| NFS-e / ABRASF | NFS-e / ABRASF | 6 |
| Record linkage | Pareamento de registros | 6 |
| Supplier payment term | Prazo de pagamento do fornecedor | 6 |
| Withholding taxes (IRRF, CSRF, INSS) | Retenções tributárias (IRRF, CSRF, INSS) | 6 |

---

## 1. Experimental methodology · Metodologia experimental

### Ablation · Ablação

**EN.** An ablation study measures how much each component of a system contributes to its result by
removing or replacing that component, one at a time, while holding everything else constant, and
observing the change in performance. The term is borrowed from neuroscience, where ablation is the
surgical removal of tissue to infer its function; Newell (1975) used it for removing knowledge sources
from speech-understanding systems, and it is now a standard requirement for supporting claims in machine
learning papers (Cohen & Howe, 1988; Meyes et al., 2019; Lipton & Steinhardt, 2019). The key property is
*control*: if the data, split, seed, hyperparameter grid or cross-validation protocol also changed, the
difference could not be attributed to the ablated component.

In this project, `run_ablation.py` (experiment 2) keeps the data, the train/test split, the seed (42), the
grid and the 5-fold stratified F1-macro protocol fixed, and varies **only the feature representation**,
across four configurations written to `data/results/ablation/`:

| Configuration | Change | Question it answers |
| --- | --- | --- |
| `full` | The seven original features | Baseline; reproduces experiment 1 to the last digit |
| `no_leak` | Drops `delta_days` and `delta_valor_pct` | How much performance comes from [direct leakage](#data-leakage--label-leakage--vazamento-de-dados--vazamento-de-rótulo) |
| `no_leak_strict` | Also drops `nfse_valor_iss` and `nfse_aliquota` | How much comes from [indirect leakage](#indirect-leakage--proxy--vazamento-indireto--proxy) through ISS proxies |
| `sentinel_fixed` | Replaces [sentinels](#sentinel-value--valor-sentinela) by `has_match` plus neutral imputation | Whether the gap between algorithms is a [scale artifact](#scale-artifact--artefato-de-escala) |

It showed that the 11–13 p.p. F1-macro gap between Random Forest and the smooth-boundary models vanished
under `sentinel_fixed`, and that without the leaking features the synthetic data carried almost no
learnable signal.

**PT-BR.** Um estudo de ablação mede quanto cada componente de um sistema contribui para o resultado,
removendo ou substituindo esse componente, um de cada vez, com todo o resto mantido constante, e
observando a variação de desempenho. O termo vem da neurociência, em que ablação é a remoção cirúrgica de
tecido para inferir sua função; Newell (1975) o usou para a remoção de fontes de conhecimento em sistemas
de reconhecimento de fala, e hoje ele é exigência comum para sustentar afirmações em artigos de
aprendizado de máquina (Cohen & Howe, 1988; Meyes et al., 2019; Lipton & Steinhardt, 2019). A propriedade
essencial é o *controle*: se os dados, a partição, a semente, a grade de hiperparâmetros ou o protocolo de
validação cruzada também mudassem, a diferença não poderia ser atribuída ao componente removido.

Neste projeto, `run_ablation.py` (experimento 2) mantém fixos os dados, a partição treino/teste, a semente
(42), a grade e o protocolo de 5 dobras estratificadas com F1-macro, e varia **exclusivamente a
representação das variáveis**, em quatro configurações (tabela acima). A ablação mostrou que a diferença
de 11 a 13 p.p. de F1-macro entre a Random Forest e os modelos de fronteira suave desaparecia em
`sentinel_fixed`, e que, sem as variáveis que vazavam o rótulo, os dados sintéticos quase não continham
sinal aprendível. Detalhes em [`discussao-limitacoes-contribuicoes.md` §3](discussao-limitacoes-contribuicoes.md#3-experimento-complementar-de-ablação).

**References · Referências:** [Newell 1975](#ref-newell1975); [Cohen & Howe 1988](#ref-cohen1988);
[Meyes et al. 2019](#ref-meyes2019); [Lipton & Steinhardt 2019](#ref-lipton2019).

### Circularity · Circularidade

**EN.** A model evaluation is circular when the target is defined as a deterministic function of the
model's own inputs, so a high score measures the ability to rediscover the labeling rule rather than to
learn the phenomenon. In experiment 1 the label is a threshold over `delta_days` and `delta_valor_pct`,
which are also features, so Random Forest's accuracy of 1.0000 is a diagnostic, not a finding.

**PT-BR.** Uma avaliação é circular quando o alvo é definido como função determinística das próprias
entradas do modelo, de modo que um escore alto mede a capacidade de redescobrir a regra de rotulagem, e não
de aprender o fenômeno. No experimento 1, o rótulo é um limiar sobre `delta_days` e `delta_valor_pct`, que
também são variáveis do modelo; por isso a acurácia de 1,0000 da Random Forest é um diagnóstico, e não um
resultado.

**References · Referências:** [Kaufman et al. 2012](#ref-kaufman2012); [Kapoor & Narayanan 2023](#ref-kapoor2023).

### Data leakage / label leakage · Vazamento de dados / vazamento de rótulo

**EN.** Leakage is the presence, in the training data, of information that would not legitimately be
available at prediction time, or that encodes the target itself. It inflates evaluation metrics without
any corresponding real-world skill and is one of the main causes of irreproducible results in ML-based
science. *Label leakage* is the case where the target is recoverable from the features — here, directly
through the labeling rule's own inputs.

**PT-BR.** Vazamento é a presença, nos dados de treino, de informação que não estaria legitimamente
disponível no momento da predição, ou que codifica o próprio alvo. Ele infla as métricas de avaliação sem
capacidade real correspondente e é uma das principais causas de resultados irreprodutíveis em ciência
baseada em aprendizado de máquina. *Vazamento de rótulo* é o caso em que o alvo é recuperável a partir das
variáveis — aqui, diretamente pelas entradas da regra de rotulagem.

**References · Referências:** [Kaufman et al. 2012](#ref-kaufman2012); [Kapoor & Narayanan 2023](#ref-kapoor2023).

### Indirect leakage / proxy · Vazamento indireto / *proxy*

**EN.** A proxy is a feature that is not part of the labeling rule but correlates with it by construction,
leaking the label through a side channel. In experiments 1–2, `nfse_valor_iss` and `nfse_aliquota` are
`0` exactly when no invoice joined, so they reveal the missing-invoice case — which the `no_leak_strict`
ablation removes.

**PT-BR.** *Proxy* é uma variável que não participa da regra de rotulagem mas se correlaciona com ela por
construção, vazando o rótulo por um canal lateral. Nos experimentos 1 e 2, `nfse_valor_iss` e
`nfse_aliquota` valem `0` exatamente quando nenhuma nota foi associada, revelando o caso de ausência de
nota — que a configuração `no_leak_strict` da ablação remove.

**References · Referências:** [Kaufman et al. 2012](#ref-kaufman2012).

### Transductive leakage · Vazamento transdutivo

**EN.** Leakage of an aggregate statistic (mean, median, vocabulary, scaler parameters) computed over the
full dataset, test partition included, instead of over the training partition only. The `sentinel_fixed`
median was computed this way; the effect is immaterial because the median is exactly 0.0, but it is
recorded as a limitation. Experiment 3 fits supplier terms (`fit_supplier_terms`) on training data only.

**PT-BR.** Vazamento de uma estatística agregada (média, mediana, vocabulário, parâmetros de escala)
calculada sobre a base inteira, partição de teste incluída, em vez de apenas sobre a partição de treino. A
mediana de `sentinel_fixed` foi calculada assim; o efeito é imaterial porque a mediana é exatamente 0,0,
mas consta como limitação. O experimento 3 ajusta os prazos de fornecedor (`fit_supplier_terms`) apenas no
treino.

**References · Referências:** [Kaufman et al. 2012](#ref-kaufman2012); [Hastie et al. 2009](#ref-hastie2009) (§7.10.2).

### Leak guard · Guarda anti-vazamento

**EN.** An automated check, in `src/reconciliacao/models/leak_guard.py`, that fits a
[decision stump](#decision-stump--toco-de-decisão) on each feature alone and **fails the run** (raises
`LeakDetected`) if any single feature reaches cross-validated accuracy above
`leak_guard_max_stump_accuracy` (0.95). It is the regression test experiment 1 lacked; its per-feature
report is saved to `leak_guard_report.csv`.

**PT-BR.** Verificação automática, em `src/reconciliacao/models/leak_guard.py`, que ajusta um
[toco de decisão](#decision-stump--toco-de-decisão) sobre cada variável isolada e **interrompe a execução**
(lança `LeakDetected`) se alguma delas atingir acurácia em validação cruzada acima de
`leak_guard_max_stump_accuracy` (0,95). É o teste de regressão que faltava ao experimento 1; o relatório
por variável é gravado em `leak_guard_report.csv`.

**References · Referências:** [Kaufman et al. 2012](#ref-kaufman2012); [Holte 1993](#ref-holte1993).

### Decision stump · Toco de decisão

**EN.** A decision tree of depth one: a single split on a single feature. Simple one-feature rules are
surprisingly strong baselines; here the stump is used the other way round, as a detector — if one split on
one feature almost perfectly separates the classes, the label is recoverable from that feature in isolation.

**PT-BR.** Árvore de decisão de profundidade um: um único corte sobre uma única variável. Regras simples de
uma variável são linhas de base surpreendentemente fortes; aqui o toco é usado ao contrário, como detector —
se um corte em uma variável separa as classes quase perfeitamente, o rótulo é recuperável a partir dela
isoladamente.

**References · Referências:** [Iba & Langley 1992](#ref-iba1992); [Holte 1993](#ref-holte1993).

### Ground truth · Verdade de referência (*ground truth*)

**EN.** The known correct answer against which predictions are judged. In experiment 3 the simulator knows
which payment settles which invoice and persists that fact (`truth_generator.py`), so the label comes from
generation rather than from a rule over the features (`truth_labeler.py`). Decoupling is by construction,
not by dropping features.

**PT-BR.** A resposta correta conhecida contra a qual as predições são julgadas. No experimento 3, o
simulador sabe qual pagamento quita qual nota e persiste esse fato (`truth_generator.py`), de modo que o
rótulo vem da geração, e não de uma regra sobre as variáveis (`truth_labeler.py`). O desacoplamento é por
construção, não por remoção de variáveis.

**References · Referências:** [Patki et al. 2016](#ref-patki2016); [Christen 2012](#ref-christen2012).

### Synthetic data · Dados sintéticos

**EN.** Data produced by a generator instead of collected from operations. It makes the experiment
shareable and gives exact ground truth, but results hold only for the generator's assumptions — the main
external-validity limitation of this study. Generators live in `src/reconciliacao/simulation/` and use
Faker with the `pt_BR` locale.

**PT-BR.** Dados produzidos por um gerador, em vez de coletados da operação. Tornam o experimento
compartilhável e fornecem verdade de referência exata, mas os resultados valem apenas para as premissas do
gerador — principal limitação de validade externa deste estudo. Os geradores estão em
`src/reconciliacao/simulation/` e usam Faker com a localidade `pt_BR`.

**References · Referências:** [Patki et al. 2016](#ref-patki2016).

### Random seed · Semente aleatória

**EN.** The initial value of a pseudorandom generator; fixing it makes every random draw reproducible.
Experiments 1–2 use a single seed (42); experiment 3 generates **10 independent datasets** (`n_seeds: 10`)
and reports dispersion across them, which is what makes paired statistical tests possible.

**PT-BR.** Valor inicial de um gerador pseudoaleatório; fixá-lo torna reprodutível todo sorteio. Os
experimentos 1 e 2 usam semente única (42); o experimento 3 gera **10 bases independentes** (`n_seeds: 10`)
e reporta a dispersão entre elas, o que viabiliza os testes estatísticos pareados.

**References · Referências:** [Peng 2011](#ref-peng2011); [Demšar 2006](#ref-demsar2006).

### Determinism / hash seed · Determinismo / semente de *hash*

**EN.** A pipeline is deterministic when the same configuration produces bit-identical outputs. Seeding
the RNGs is not enough: Python randomizes string hashing per process (`PYTHONHASHSEED`), so iterating a
`set` of strings can change order between runs. That silently reassigned supplier CNPJs in an earlier
version; `tests/test_simulation/test_excel_generator.py` now checks the property across two subprocesses
with different hash seeds. Dates are anchored to `simulation.reference_date`, not the wall clock.

**PT-BR.** Um pipeline é determinístico quando a mesma configuração produz saídas idênticas bit a bit.
Fixar as sementes não basta: o Python aleatoriza o *hash* de *strings* por processo (`PYTHONHASHSEED`), e a
iteração sobre um `set` de *strings* pode mudar de ordem entre execuções. Isso reatribuía silenciosamente
os CNPJs dos fornecedores numa versão anterior; `tests/test_simulation/test_excel_generator.py` agora
verifica a propriedade em dois subprocessos com sementes de *hash* diferentes. As datas são ancoradas em
`simulation.reference_date`, e não no relógio.

**References · Referências:** [Python docs](#ref-python-hashseed); [Peng 2011](#ref-peng2011).

### Winner's curse / selection bias · Maldição do vencedor / viés de seleção

**EN.** When the best of many candidates is chosen by a noisy estimate, the winner's estimate is biased
upward, because it tends to be the candidate that was lucky. `choose_threshold` picks the highest-recall
cut point among thousands whose *validation* precision clears 0.90, so test precision tends to land just
under the floor. `precision_margin` in `config.yaml` exists to compensate (default `0.0` reproduces the
published artifacts).

**PT-BR.** Quando o melhor entre muitos candidatos é escolhido por uma estimativa ruidosa, a estimativa do
vencedor é enviesada para cima, porque ele tende a ser o candidato que teve sorte. `choose_threshold`
escolhe, entre milhares de pontos de corte, o de maior recall cuja precisão de *validação* atinge 0,90; por
isso a precisão de teste tende a ficar logo abaixo do piso. O parâmetro `precision_margin` do `config.yaml`
existe para compensar (o padrão `0.0` reproduz os artefatos publicados).

**References · Referências:** [Thaler 1988](#ref-thaler1988); [Cawley & Talbot 2010](#ref-cawley2010).

---

## 2. Data representation · Representação dos dados

### Feature · Variável / atributo (*feature*)

**EN.** A measurable input given to the model. Experiments 1–2 use 7 features (`features.py`);
experiment 3 uses 14 (`features_v2.py`), all computable by someone who does not know whether the pairing is
genuine, and none participating in the label definition. Both lists are in the README.

**PT-BR.** Entrada mensurável fornecida ao modelo. Os experimentos 1 e 2 usam 7 variáveis (`features.py`);
o experimento 3 usa 14 (`features_v2.py`), todas computáveis por quem não sabe se o pareamento é genuíno, e
nenhuma participando da definição do rótulo. As duas listas estão no README.

**References · Referências:** [Hastie et al. 2009](#ref-hastie2009).

### Sentinel value · Valor-sentinela

**EN.** An out-of-range placeholder used to mark "not available" inside a numeric column. In experiments
1–2, `delta_days = 9999` and `delta_valor_pct = 100` mark payments with no joined invoice. Tree models
isolate such values with one split; distance- and scale-based models do not — see
[scale artifact](#scale-artifact--artefato-de-escala).

**PT-BR.** Marcador fora da faixa usado para indicar "indisponível" dentro de uma coluna numérica. Nos
experimentos 1 e 2, `delta_days = 9999` e `delta_valor_pct = 100` marcam pagamentos sem nota associada.
Modelos de árvore isolam esses valores com um corte; modelos baseados em distância e escala, não — ver
[artefato de escala](#scale-artifact--artefato-de-escala).

**References · Referências:** [Little & Rubin 2019](#ref-little2019).

### Missing indicator · Indicador de ausência

**EN.** A binary feature that carries the fact that a value is missing, so the numeric column can be imputed
with a neutral value instead of an extreme one. `sentinel_fixed` adds `has_match` (1 when an invoice joined).

**PT-BR.** Variável binária que carrega o fato de um valor estar ausente, permitindo imputar a coluna
numérica com um valor neutro em vez de extremo. A configuração `sentinel_fixed` adiciona `has_match` (1
quando uma nota foi associada).

**References · Referências:** [Little & Rubin 2019](#ref-little2019).

### Imputation · Imputação

**EN.** Replacing missing values with estimates. `sentinel_fixed` imputes `delta_days` and
`delta_valor_pct` by the median over joined records, which in this dataset is exactly 0.0.

**PT-BR.** Substituição de valores ausentes por estimativas. A configuração `sentinel_fixed` imputa
`delta_days` e `delta_valor_pct` pela mediana dos registros pareados, que nesta base é exatamente 0,0.

**References · Referências:** [Little & Rubin 2019](#ref-little2019).

### Standardization · Padronização

**EN.** Rescaling each feature to zero mean and unit variance (scikit-learn's `StandardScaler`), required
by SVM and regularized logistic regression so that no feature dominates by its units. A column with zero
training variance is left unscaled (`scale_ = 1`), which is how a degenerate feature turned into a scale
artifact in an early version of experiment 3.

**PT-BR.** Reescala de cada variável para média zero e variância unitária (`StandardScaler`, do
scikit-learn), necessária ao SVM e à regressão logística regularizada para que nenhuma variável domine por
sua unidade. Uma coluna com variância zero no treino não é reescalada (`scale_ = 1`), e foi assim que uma
variável degenerada se tornou artefato de escala numa versão inicial do experimento 3.

**References · Referências:** [scikit-learn preprocessing](#ref-sklearn-preprocessing); [Hastie et al. 2009](#ref-hastie2009).

### Scale artifact · Artefato de escala

**EN.** A performance difference caused by how values are scaled rather than by what the data contain. The
sentinel 9999, once standardized, compresses every real `delta_days` into a tiny band near the origin,
hiding a boundary that the linear and RBF models could otherwise find. The ablation's `sentinel_fixed`
configuration made all three algorithms reach identical, perfect confusion matrices in the `exact` scenario.

**PT-BR.** Diferença de desempenho causada pela forma de escalar os valores, e não pelo que os dados
contêm. A sentinela 9999, depois de padronizada, comprime todos os `delta_days` reais numa faixa estreita
junto à origem, escondendo uma fronteira que os modelos linear e RBF encontrariam. Na configuração
`sentinel_fixed` da ablação, os três algoritmos chegaram a matrizes de confusão idênticas e perfeitas no
cenário `exact`.

**References · Referências:** [Hastie et al. 2009](#ref-hastie2009).

### Affine redundancy · Redundância afim

**EN.** Two features where one is a linear transform of the other, so they carry one direction of
information, not two. `retencao_implicita_pct = (1 − razao_valor) × 100`, so the 14-feature matrix of
experiment 3 spans 13 directions. It is kept deliberately and asserted in the test suite.

**PT-BR.** Duas variáveis em que uma é transformação linear da outra, carregando uma direção de informação,
e não duas. `retencao_implicita_pct = (1 − razao_valor) × 100`; assim, a matriz de 14 variáveis do
experimento 3 cobre 13 direções. A redundância é mantida deliberadamente e verificada na suíte de testes.

**References · Referências:** [Hastie et al. 2009](#ref-hastie2009).

### TF-IDF / cosine similarity · TF-IDF / similaridade do cosseno

**EN.** TF-IDF weights each term of a text by its frequency in the document, discounted by how common it
is across documents; cosine similarity measures the angle between two such vectors (1 = same direction).
Used for `descricao_similarity` / `similaridade_descricao`, comparing the payment memo with the NFS-e
service description.

**PT-BR.** O TF-IDF pondera cada termo de um texto pela sua frequência no documento, descontada por quão
comum ele é entre documentos; a similaridade do cosseno mede o ângulo entre dois desses vetores (1 = mesma
direção). É usada em `descricao_similarity` / `similaridade_descricao`, comparando o histórico do
pagamento com a discriminação do serviço na NFS-e.

**References · Referências:** [Salton & Buckley 1988](#ref-salton1988); [Manning et al. 2008](#ref-manning2008).

---

## 3. Models and training · Modelos e treinamento

### Random Forest · Floresta Aleatória (*Random Forest*)

**EN.** An ensemble of decision trees, each trained on a bootstrap sample with a random subset of features
at each split, whose votes are averaged. Insensitive to monotone rescaling, which is why it was immune to
the sentinel scale artifact. Configured with 200 trees and `class_weight: balanced_subsample`.

**PT-BR.** Conjunto de árvores de decisão, cada uma treinada em amostra *bootstrap* com subconjunto
aleatório de variáveis a cada corte, cujos votos são agregados. É insensível a reescalas monótonas, razão
pela qual ficou imune ao artefato de escala da sentinela. Configurada com 200 árvores e
`class_weight: balanced_subsample`.

**References · Referências:** [Breiman 2001](#ref-breiman2001).

### Support Vector Machine (RBF kernel) · Máquina de Vetores de Suporte (núcleo RBF)

**EN.** A maximum-margin classifier; the radial basis function kernel lets it draw non-linear boundaries
from distances between points, which makes it sensitive to feature scale. Configured with
`kernel: rbf` and `class_weight: balanced`.

**PT-BR.** Classificador de margem máxima; o núcleo de função de base radial permite fronteiras não
lineares a partir de distâncias entre pontos, o que o torna sensível à escala das variáveis. Configurado
com `kernel: rbf` e `class_weight: balanced`.

**References · Referências:** [Cortes & Vapnik 1995](#ref-cortes1995).

### Logistic Regression · Regressão Logística

**EN.** A linear model of the log-odds of the class, fitted by maximum likelihood (here with
regularization). Its probabilities are directly usable for [threshold](#decision-threshold--operating-point--limiar-de-decisão--ponto-de-operação)
selection. Configured with `class_weight: balanced`.

**PT-BR.** Modelo linear do logaritmo da razão de chances da classe, ajustado por máxima verossimilhança
(aqui com regularização). Suas probabilidades servem diretamente à escolha do
[limiar](#decision-threshold--operating-point--limiar-de-decisão--ponto-de-operação). Configurada com
`class_weight: balanced`.

**References · Referências:** [Hosmer et al. 2013](#ref-hosmer2013).

### Grid search / hyperparameter · Busca em grade / hiperparâmetro

**EN.** Hyperparameters are settings chosen before training (number of trees, `C`, `gamma`). Grid search
(`GridSearchCV`) tries every combination of a predefined grid and keeps the best by cross-validated score —
F1-macro in experiments 1–2, exception average precision in experiment 3.

**PT-BR.** Hiperparâmetros são configurações escolhidas antes do treino (número de árvores, `C`, `gamma`). A
busca em grade (`GridSearchCV`) testa todas as combinações de uma grade predefinida e mantém a melhor pelo
escore em validação cruzada — F1-macro nos experimentos 1 e 2, precisão média da exceção no experimento 3.

**References · Referências:** [scikit-learn grid search](#ref-sklearn-grid); [Pedregosa et al. 2011](#ref-pedregosa2011).

### Cross-validation (stratified *k*-fold) · Validação cruzada (*k* dobras estratificadas)

**EN.** The training data are split into *k* folds (here `cv_folds: 5`); each fold serves once as
validation while the others train. *Stratified* folds preserve the class proportions in every fold.

**PT-BR.** Os dados de treino são divididos em *k* dobras (aqui `cv_folds: 5`); cada dobra serve uma vez
como validação enquanto as demais treinam. Dobras *estratificadas* preservam a proporção das classes em
cada dobra.

**References · Referências:** [Kohavi 1995](#ref-kohavi1995); [scikit-learn cross-validation](#ref-sklearn-cv).

### Train / validation / test split · Partição treino / validação / teste

**EN.** Experiments 1–2 hold out 20% for test. Experiment 3 splits 60/20/20: the model is fitted on
training data, the decision threshold is chosen on validation, and only then applied unchanged to test.

**PT-BR.** Os experimentos 1 e 2 reservam 20% para teste. O experimento 3 divide 60/20/20: o modelo é
ajustado no treino, o limiar de decisão é escolhido na validação e só então aplicado, sem reajuste, ao
teste.

**References · Referências:** [Hastie et al. 2009](#ref-hastie2009) (§7.2).

### Class weighting · Ponderação de classes

**EN.** Reweighting the loss so errors on the less frequent class cost more (`balanced`,
`balanced_subsample`), counteracting the 70/30 class imbalance.

**PT-BR.** Reponderação da função de perda para que erros na classe menos frequente custem mais
(`balanced`, `balanced_subsample`), compensando o desbalanceamento 70/30.

**References · Referências:** [King & Zeng 2001](#ref-king2001); [Elkan 2001](#ref-elkan2001).

### Trivial (majority-class) classifier · Classificador trivial (classe majoritária)

**EN.** A classifier that always predicts the most frequent class. Its accuracy equals that class's share
(0.6981 here), with zero recall on the exception class — the floor any useful model must beat. In
`no_leak_strict`, the SVM degenerated into exactly this.

**PT-BR.** Classificador que sempre prediz a classe mais frequente. Sua acurácia é a proporção dessa classe
(0,6981 aqui), com recall nulo na classe de exceção — o patamar que qualquer modelo útil precisa superar.
Em `no_leak_strict`, o SVM degenerou exatamente nisso.

**References · Referências:** [Saito & Rehmsmeier 2015](#ref-saito2015).

### Feature importance · Importância de variáveis

**EN.** A per-feature score of contribution to a fitted model; `feature_importance_rf.png` plots Random
Forest's impurity-based importance, which is known to be biased toward high-cardinality and continuous
features.

**PT-BR.** Escore, por variável, da contribuição para um modelo ajustado; `feature_importance_rf.png` mostra
a importância por impureza da Random Forest, sabidamente enviesada a favor de variáveis contínuas e de alta
cardinalidade.

**References · Referências:** [Breiman 2001](#ref-breiman2001); [Strobl et al. 2007](#ref-strobl2007).

---

## 4. Evaluation and decision · Avaliação e decisão

### Confusion matrix · Matriz de confusão

**EN.** A table counting predictions against true classes (true/false positives and negatives). Saved per
model as PNG/TXT in experiments 1–2 and as exception-class cells summed over seeds in
`confusion_matrices.csv`.

**PT-BR.** Tabela que conta as predições contra as classes verdadeiras (verdadeiros/falsos positivos e
negativos). Gravada por modelo em PNG/TXT nos experimentos 1 e 2, e como células da classe de exceção
somadas sobre as sementes em `confusion_matrices.csv`.

**References · Referências:** [Fawcett 2006](#ref-fawcett2006).

### Precision / recall · Precisão / revocação (*recall*)

**EN.** For a given class, precision is the share of predictions of that class that are correct; recall is
the share of true members of that class that are found. For the exception class: precision = how many
flagged payments really are exceptions; recall = how many real exceptions get flagged.

**PT-BR.** Para uma classe, precisão é a fração das predições daquela classe que estão corretas; recall
(revocação) é a fração dos membros verdadeiros da classe que são encontrados. Na classe de exceção:
precisão = quantos pagamentos sinalizados são de fato exceções; recall = quantas exceções reais são
sinalizadas.

**References · Referências:** [Manning et al. 2008](#ref-manning2008).

### F1-macro · F1-macro

**EN.** The harmonic mean of precision and recall, computed per class and averaged with equal weight across
classes. It was the selection metric of experiments 1–2; experiment 3 replaced it because it treats a
silenced divergence and a redundant manual review as equally costly.

**PT-BR.** Média harmônica de precisão e recall, calculada por classe e com média de peso igual entre as
classes. Foi a métrica de seleção dos experimentos 1 e 2; o experimento 3 a substituiu porque ela trata uma
divergência silenciada e uma revisão manual redundante como igualmente custosas.

**References · Referências:** [Manning et al. 2008](#ref-manning2008); [Elkan 2001](#ref-elkan2001).

### ROC curve / ROC-AUC · Curva ROC / ROC-AUC

**EN.** The ROC curve plots true-positive rate against false-positive rate across all thresholds; ROC-AUC is
the area under it. Reported for the binary `exact` scenario (`roc_curve.png`). Under class imbalance it can
look optimistic, which motivates PR curves.

**PT-BR.** A curva ROC traça a taxa de verdadeiros positivos contra a de falsos positivos em todos os
limiares; ROC-AUC é a área sob ela. Reportada para o cenário binário `exact` (`roc_curve.png`). Com classes
desbalanceadas pode parecer otimista, o que motiva as curvas PR.

**References · Referências:** [Fawcett 2006](#ref-fawcett2006).

### PR curve / PR-AUC / average precision · Curva PR / PR-AUC / precisão média

**EN.** The precision–recall curve plots precision against recall across thresholds for one class; PR-AUC
(estimated by average precision) summarizes it. More informative than ROC for the minority class.
Experiment 3 uses exception average precision as the grid-search score (`average_precision_excecao`) and
saves `pr_curves.png`.

**PT-BR.** A curva precisão–recall traça precisão contra recall em todos os limiares, para uma classe; a
PR-AUC (estimada pela precisão média) a resume. É mais informativa que a ROC para a classe minoritária. O
experimento 3 usa a precisão média da exceção como escore da busca em grade (`average_precision_excecao`)
e grava `pr_curves.png`.

**References · Referências:** [Saito & Rehmsmeier 2015](#ref-saito2015).

### Exception class · Classe de exceção

**EN.** Label 0 in experiment 3: a payment whose pairing with the candidate invoice is **not** genuine. In
audit terms it is the item that must be investigated; missing one (a false negative) is the costly error.

**PT-BR.** Rótulo 0 no experimento 3: pagamento cujo pareamento com a nota candidata **não** é genuíno. Em
termos de auditoria, é o item que precisa ser investigado; deixar de detectá-lo (falso negativo) é o erro
custoso.

**References · Referências:** [COSO 2013](#ref-coso2013); [CFC NBC TA 530](#ref-nbcta530).

### Exception recall · Recall da exceção

**EN.** The primary metric of experiment 3: the share of genuine exceptions the model flags, measured under
the precision floor. It is derived from the control objective rather than from convention.

**PT-BR.** Métrica primária do experimento 3: fração das exceções genuínas que o modelo sinaliza, medida sob
o piso de precisão. Deriva do objetivo de controle, e não de convenção.

**References · Referências:** [Elkan 2001](#ref-elkan2001).

### Precision floor · Piso de precisão

**EN.** The minimum acceptable exception precision (`min_precision: 0.90`): at least 9 of 10 flagged
payments must be real exceptions, bounding the manual-review load. Algorithms that cannot reach it on
validation are reported as unable to meet the operating constraint (`n_seeds_applicable`).

**PT-BR.** Precisão mínima aceitável na classe de exceção (`min_precision: 0.90`): ao menos 9 em cada 10
pagamentos sinalizados devem ser exceções reais, limitando a carga de revisão manual. Algoritmos que não o
atingem na validação são reportados como incapazes de cumprir a restrição operacional
(`n_seeds_applicable`).

**References · Referências:** [Elkan 2001](#ref-elkan2001).

### Decision threshold / operating point · Limiar de decisão / ponto de operação

**EN.** The probability cut-off above which a payment is flagged as an exception; the resulting
(precision, recall) pair is the operating point. `decision.choose_threshold` picks the highest-recall
threshold meeting the floor on validation.

**PT-BR.** Ponto de corte de probabilidade acima do qual um pagamento é sinalizado como exceção; o par
(precisão, recall) resultante é o ponto de operação. `decision.choose_threshold` escolhe, na validação, o
limiar de maior recall que atinge o piso.

**References · Referências:** [Elkan 2001](#ref-elkan2001); [Fawcett 2006](#ref-fawcett2006).

### Referral rate · Taxa de encaminhamento

**EN.** The share of payments sent to manual review at the chosen operating point
(`taxa_encaminhamento`), reported next to recall so the cost of catching exceptions is visible.

**PT-BR.** Fração dos pagamentos encaminhados à revisão manual no ponto de operação escolhido
(`taxa_encaminhamento`), reportada ao lado do recall para que o custo de capturar exceções fique visível.

**References · Referências:** [CFC NBC TA 530](#ref-nbcta530).

---

## 5. Statistical comparison · Comparação estatística

### Friedman test (omnibus) · Teste de Friedman (omnibus)

**EN.** A non-parametric rank test of whether *k* related samples (here, 3 algorithms over 10 seeds) come
from the same distribution. It is *omnibus*: it tests the global null "all algorithms are equivalent" and
must reject it before any pairwise comparison is interpreted (`friedman.csv`).

**PT-BR.** Teste não paramétrico de postos que verifica se *k* amostras relacionadas (aqui, 3 algoritmos
sobre 10 sementes) vêm da mesma distribuição. É *omnibus*: testa a hipótese nula global "todos os
algoritmos são equivalentes" e precisa rejeitá-la antes que qualquer comparação par a par seja
interpretada (`friedman.csv`).

**References · Referências:** [Friedman 1937](#ref-friedman1937); [Demšar 2006](#ref-demsar2006).

### Post-hoc test · Teste *post-hoc*

**EN.** A follow-up test run after an omnibus rejection to locate which pairs differ. Running pairwise
tests without the omnibus step first is the order Demšar (2006) warns against.

**PT-BR.** Teste complementar executado após a rejeição do omnibus para localizar quais pares diferem.
Executar testes par a par sem o omnibus antes é a ordem que Demšar (2006) desaconselha.

**References · Referências:** [Demšar 2006](#ref-demsar2006).

### Wilcoxon signed-rank test · Teste dos postos sinalizados de Wilcoxon

**EN.** A non-parametric paired test on the signed ranks of per-seed differences between two algorithms;
here the post-hoc step (`wilcoxon.csv`, and `wilcoxon_secundarias.csv` for PR-AUC and exception precision).

**PT-BR.** Teste pareado não paramétrico sobre os postos sinalizados das diferenças, por semente, entre dois
algoritmos; aqui é a etapa *post-hoc* (`wilcoxon.csv`, e `wilcoxon_secundarias.csv` para PR-AUC e precisão
da exceção).

**References · Referências:** [Wilcoxon 1945](#ref-wilcoxon1945); [Demšar 2006](#ref-demsar2006).

### Holm correction · Correção de Holm

**EN.** A step-down adjustment of *p*-values that controls the family-wise error rate across multiple
comparisons (here, the three algorithm pairs), uniformly more powerful than Bonferroni.

**PT-BR.** Ajuste sequencial dos valores-*p* que controla a taxa de erro da família de comparações (aqui,
os três pares de algoritmos), uniformemente mais poderoso que Bonferroni.

**References · Referências:** [Holm 1979](#ref-holm1979).

### Effect size · Tamanho de efeito

**EN.** The magnitude of a difference, independent of its significance. With only 10 seeds the tests have
little power, so the median paired difference is reported next to each *p*-value: significance without
magnitude does not support a recommendation.

**PT-BR.** A magnitude de uma diferença, independentemente de sua significância. Com apenas 10 sementes os
testes têm pouco poder, por isso a diferença pareada mediana é reportada ao lado de cada valor-*p*:
significância sem magnitude não sustenta uma recomendação.

**References · Referências:** [Demšar 2006](#ref-demsar2006).

---

## 6. Accounting and tax domain · Domínio contábil e tributário

### Financial reconciliation · Conciliação financeira

**EN.** The control activity of matching records from independent sources — here, supplier payments in an
Excel ledger against electronic service invoices (NFS-e) in XML — and investigating the divergences. It is
an internal control in the COSO sense and a source of audit evidence.

**PT-BR.** Atividade de controle que confronta registros de fontes independentes — aqui, pagamentos a
fornecedores numa planilha Excel contra notas fiscais de serviço eletrônicas (NFS-e) em XML — e investiga as
divergências. É controle interno no sentido do COSO e fonte de evidência de auditoria.

**References · Referências:** [COSO 2013](#ref-coso2013); [CFC NBC TA 500](#ref-nbcta500); [Vasarhelyi et al. 2015](#ref-vasarhelyi2015).

### Internal control / audit evidence · Controle interno / evidência de auditoria

**EN.** Internal control is the process that gives reasonable assurance about the reliability of reporting;
audit evidence is the information the auditor uses to reach conclusions. NBC TA 530 (audit sampling) frames
the asymmetry between a missed exception and an unnecessary review that motivates the decision rule.

**PT-BR.** Controle interno é o processo que fornece segurança razoável quanto à confiabilidade das
informações; evidência de auditoria é a informação que o auditor usa para chegar a conclusões. A NBC TA
530 (amostragem em auditoria) fundamenta a assimetria entre uma exceção não detectada e uma revisão
desnecessária, que motiva a regra de decisão.

**References · Referências:** [COSO 2013](#ref-coso2013); [CFC NBC TA 500](#ref-nbcta500); [CFC NBC TA 530](#ref-nbcta530).

### Conciliated / partially / not conciliated · Conciliado / parcialmente / não conciliado

**EN.** The labels of experiments 1–2, assigned by `labeler.py` from date and value tolerances: binary in
`exact` (conciliated / not conciliated), ternary in `fuzzy` (adds *partially conciliated*).

**PT-BR.** Os rótulos dos experimentos 1 e 2, atribuídos por `labeler.py` a partir das tolerâncias de data e
valor: binários no `exact` (conciliado / não conciliado), ternários no `fuzzy` (acrescenta *parcialmente
conciliado*).

**References · Referências:** [Christen 2012](#ref-christen2012).

### Exact and fuzzy scenarios · Cenários *exact* e *fuzzy*

**EN.** The two labeling regimes of experiments 1–2: `exact` tolerates 0 days and 0% of value difference;
`fuzzy` tolerates ±5 business days and ±2%. Because the label is a threshold either way, the fuzzy scenario
changes the rule's strictness, not the model's tolerance.

**PT-BR.** Os dois regimes de rotulagem dos experimentos 1 e 2: `exact` tolera 0 dia e 0% de diferença de
valor; `fuzzy` tolera ±5 dias úteis e ±2%. Como o rótulo é um limiar em ambos, o cenário *fuzzy* altera o
rigor da regra, e não a tolerância do modelo.

**References · Referências:** [Christen 2012](#ref-christen2012).

### Record linkage · Pareamento de registros

**EN.** The problem of deciding whether two records from different sources refer to the same entity or
event. Fellegi and Sunter's probabilistic framework — with its match / possible match / non-match decision
and a clerical-review region — is the classical antecedent of this project's referral-based decision rule.

**PT-BR.** Problema de decidir se dois registros de fontes diferentes se referem à mesma entidade ou
evento. O arcabouço probabilístico de Fellegi e Sunter — com a decisão par / par possível / não par e uma
região de revisão manual — é o antecedente clássico da regra de decisão com encaminhamento deste projeto.

**References · Referências:** [Fellegi & Sunter 1969](#ref-fellegi1969); [Christen 2012](#ref-christen2012).

### Hard and soft negatives · Negativos difíceis e fáceis

**EN.** In experiment 3, non-pairs whose value gap matches a **legal** withholding combination are *hard*
negatives (they look like true pairs on value); those with an illegitimate percentage are *soft*.
`hard_negative_rate: 0.30` is the share of non-pairs that are hard. Each non-pair also differs in 1–3 of
term, description and municipality.

**PT-BR.** No experimento 3, os não-pares cuja diferença de valor coincide com uma combinação **legal** de
retenções são negativos *difíceis* (parecem pares verdadeiros pelo valor); os com percentual ilegítimo são
*fáceis*. `hard_negative_rate: 0.30` é a fração dos não-pares que são difíceis. Cada não-par também
diverge em 1 a 3 dimensões entre prazo, descrição e município.

**References · Referências:** [Christen 2012](#ref-christen2012).

### NFS-e / ABRASF · NFS-e / ABRASF

**EN.** NFS-e (*Nota Fiscal de Serviços Eletrônica*) is the Brazilian electronic service invoice, issued
under municipal ISS rules. ABRASF (the association of municipal finance secretaries) publishes the national
conceptual model and XML layout that `xml_generator.py` follows.

**PT-BR.** A NFS-e (Nota Fiscal de Serviços Eletrônica) é o documento fiscal eletrônico de prestação de
serviços, emitido sob as regras municipais do ISS. A ABRASF (Associação Brasileira das Secretarias de
Finanças das Capitais) publica o modelo conceitual e o leiaute XML nacional que `xml_generator.py` segue.

**References · Referências:** [ABRASF](#ref-abrasf); [LC 116/2003](#ref-lc116).

### CNPJ · CNPJ

**EN.** *Cadastro Nacional da Pessoa Jurídica*, the Brazilian federal company registration number (14
digits, two check digits), validated in `utils/cnpj.py` and used as the join key between payments and
invoices.

**PT-BR.** Cadastro Nacional da Pessoa Jurídica, número de inscrição federal das empresas (14 dígitos, dois
verificadores), validado em `utils/cnpj.py` e usado como chave de junção entre pagamentos e notas.

**References · Referências:** [IN RFB 2.119/2022](#ref-inrfb2119).

### ISS · ISS

**EN.** *Imposto Sobre Serviços*, the municipal service tax, with rates between 2% and 5% (`iss_rate_range_pct`).
It may be withheld by the service taker, in which case it reduces the amount paid.

**PT-BR.** Imposto Sobre Serviços de Qualquer Natureza, tributo municipal com alíquotas entre 2% e 5%
(`iss_rate_range_pct`). Pode ser retido pelo tomador do serviço, caso em que reduz o valor pago.

**References · Referências:** [LC 116/2003](#ref-lc116).

### Withholding taxes (IRRF, CSRF, INSS) · Retenções tributárias (IRRF, CSRF, INSS)

**EN.** Amounts the payer must deduct from a service payment and remit to the government, so a genuine
payment is lower than its invoice. Modeled in `retentions.py` with rates from `config.yaml`: IRRF (income
tax withheld at source, 1.50%), CSRF (PIS + COFINS + CSLL, 4.65%) and INSS (social security, 11%), alone or
combined with withheld ISS. Feature columns `compat_*` flag which legal combination a payment's gap matches.

**PT-BR.** Valores que o tomador deve descontar do pagamento de um serviço e recolher ao fisco, de modo que
um pagamento genuíno é menor que sua nota. Modelados em `retentions.py` com alíquotas do `config.yaml`:
IRRF (imposto de renda retido na fonte, 1,50%), CSRF (PIS + COFINS + CSLL, 4,65%) e INSS (contribuição
previdenciária, 11%), isolados ou combinados com o ISS retido. As variáveis `compat_*` indicam qual
combinação legal a diferença do pagamento reproduz.

**References · Referências:** [Decreto 9.580/2018](#ref-rir2018); [Lei 10.833/2003](#ref-lei10833);
[Lei 8.212/1991](#ref-lei8212); [LC 116/2003](#ref-lc116).

### CSRF threshold · Limite da CSRF

**EN.** In the simulator, CSRF only applies to invoices above `csrf_threshold_brl` (R$ 5,000.00), and the
feature `acima_limite_csrf` lets the model learn that interaction. This follows the original wording of
Lei 10.833/2003, art. 31, §3; since Lei 13.137/2015 the waiver is based on the withheld amount
(≤ R$ 10.00) rather than the invoice value, so the parameter is a modeling simplification.

**PT-BR.** No simulador, a CSRF só se aplica a notas acima de `csrf_threshold_brl` (R$ 5.000,00), e a
variável `acima_limite_csrf` permite ao modelo aprender essa interação. A regra segue a redação original
da Lei 10.833/2003, art. 31, §3º; desde a Lei 13.137/2015 a dispensa se baseia no valor retido
(≤ R$ 10,00), e não no valor da nota, de modo que o parâmetro é uma simplificação de modelagem.

**References · Referências:** [Lei 10.833/2003](#ref-lei10833); [Lei 13.137/2015](#ref-lei13137).

### Supplier payment term · Prazo de pagamento do fornecedor

**EN.** The characteristic number of days between invoice issue and payment for each supplier (drawn from
`payment_terms_days`). `desvio_prazo_fornecedor` is the deviation from the supplier's median term, fitted on
training data only; each supplier issues five invoices (`payments_per_supplier: 5`) so that median exists.

**PT-BR.** Número característico de dias entre a emissão da nota e o pagamento, por fornecedor (sorteado de
`payment_terms_days`). `desvio_prazo_fornecedor` é o desvio em relação ao prazo mediano do fornecedor,
ajustado apenas no treino; cada fornecedor emite cinco notas (`payments_per_supplier: 5`) para que essa
mediana exista.

**References · Referências:** [Christen 2012](#ref-christen2012).

### ETL · ETL (extração, transformação e carga)

**EN.** Extract, Transform, Load: reading the raw sources (`loader.py`), cleaning and normalizing them
(`cleaner.py`) and producing the labeled dataset (`labeler.py` / `truth_labeler.py`) in `data/processed/`.

**PT-BR.** Extração, transformação e carga: leitura das fontes brutas (`loader.py`), limpeza e normalização
(`cleaner.py`) e produção da base rotulada (`labeler.py` / `truth_labeler.py`) em `data/processed/`.

**References · Referências:** [Kimball & Caserta 2004](#ref-kimball2004).

---

## 7. References · Referências

**Methodology, leakage and evaluation · Metodologia, vazamento e avaliação**

- <a id="ref-cawley2010"></a>CAWLEY, G. C.; TALBOT, N. L. C. On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research*, v. 11, p. 2079–2107, 2010. Available at: <https://jmlr.org/papers/v11/cawley10a.html>.
- <a id="ref-cohen1988"></a>COHEN, P. R.; HOWE, A. E. How evaluation guides AI research. *AI Magazine*, v. 9, n. 4, p. 35–43, 1988.
- <a id="ref-demsar2006"></a>DEMŠAR, J. Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, v. 7, p. 1–30, 2006. Available at: <https://jmlr.org/papers/v7/demsar06a.html>.
- <a id="ref-elkan2001"></a>ELKAN, C. The foundations of cost-sensitive learning. In: INTERNATIONAL JOINT CONFERENCE ON ARTIFICIAL INTELLIGENCE (IJCAI), 17., 2001. *Proceedings* […]. p. 973–978.
- <a id="ref-fawcett2006"></a>FAWCETT, T. An introduction to ROC analysis. *Pattern Recognition Letters*, v. 27, n. 8, p. 861–874, 2006. DOI: 10.1016/j.patrec.2005.10.010.
- <a id="ref-hastie2009"></a>HASTIE, T.; TIBSHIRANI, R.; FRIEDMAN, J. *The Elements of Statistical Learning*. 2. ed. New York: Springer, 2009.
- <a id="ref-kapoor2023"></a>KAPOOR, S.; NARAYANAN, A. Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, v. 4, n. 9, 100804, 2023. DOI: 10.1016/j.patter.2023.100804.
- <a id="ref-kaufman2012"></a>KAUFMAN, S.; ROSSET, S.; PERLICH, C.; STITELMAN, O. Leakage in data mining: formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, v. 6, n. 4, art. 15, 2012. DOI: 10.1145/2382577.2382579.
- <a id="ref-kohavi1995"></a>KOHAVI, R. A study of cross-validation and bootstrap for accuracy estimation and model selection. In: INTERNATIONAL JOINT CONFERENCE ON ARTIFICIAL INTELLIGENCE (IJCAI), 14., 1995. *Proceedings* […]. p. 1137–1145.
- <a id="ref-lipton2019"></a>LIPTON, Z. C.; STEINHARDT, J. Troubling trends in machine learning scholarship. *ACM Queue*, v. 17, n. 1, p. 45–77, 2019.
- <a id="ref-little2019"></a>LITTLE, R. J. A.; RUBIN, D. B. *Statistical Analysis with Missing Data*. 3. ed. Hoboken: Wiley, 2019.
- <a id="ref-meyes2019"></a>MEYES, R.; LU, M.; DE PUISEAU, C. W.; MEISEN, T. Ablation studies in artificial neural networks. *arXiv preprint*, arXiv:1901.08644, 2019. Available at: <https://arxiv.org/abs/1901.08644>.
- <a id="ref-newell1975"></a>NEWELL, A. A tutorial on speech understanding systems. In: REDDY, D. R. (ed.). *Speech Recognition: Invited Papers Presented at the 1974 IEEE Symposium*. New York: Academic Press, 1975. p. 3–54.
- <a id="ref-patki2016"></a>PATKI, N.; WEDGE, R.; VEERAMACHANENI, K. The Synthetic Data Vault. In: IEEE INTERNATIONAL CONFERENCE ON DATA SCIENCE AND ADVANCED ANALYTICS (DSAA), 2016. *Proceedings* […]. p. 399–410.
- <a id="ref-peng2011"></a>PENG, R. D. Reproducible research in computational science. *Science*, v. 334, n. 6060, p. 1226–1227, 2011. DOI: 10.1126/science.1213847.
- <a id="ref-saito2015"></a>SAITO, T.; REHMSMEIER, M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, v. 10, n. 3, e0118432, 2015. DOI: 10.1371/journal.pone.0118432.
- <a id="ref-thaler1988"></a>THALER, R. H. Anomalies: the winner's curse. *Journal of Economic Perspectives*, v. 2, n. 1, p. 191–202, 1988.

**Algorithms and text representation · Algoritmos e representação de texto**

- <a id="ref-breiman2001"></a>BREIMAN, L. Random forests. *Machine Learning*, v. 45, n. 1, p. 5–32, 2001. DOI: 10.1023/A:1010933404324.
- <a id="ref-cortes1995"></a>CORTES, C.; VAPNIK, V. Support-vector networks. *Machine Learning*, v. 20, n. 3, p. 273–297, 1995. DOI: 10.1007/BF00994018.
- <a id="ref-holte1993"></a>HOLTE, R. C. Very simple classification rules perform well on most commonly used datasets. *Machine Learning*, v. 11, n. 1, p. 63–90, 1993.
- <a id="ref-hosmer2013"></a>HOSMER, D. W.; LEMESHOW, S.; STURDIVANT, R. X. *Applied Logistic Regression*. 3. ed. Hoboken: Wiley, 2013.
- <a id="ref-iba1992"></a>IBA, W.; LANGLEY, P. Induction of one-level decision trees. In: INTERNATIONAL WORKSHOP ON MACHINE LEARNING, 9., 1992. *Proceedings* […]. San Mateo: Morgan Kaufmann, 1992. p. 233–240.
- <a id="ref-king2001"></a>KING, G.; ZENG, L. Logistic regression in rare events data. *Political Analysis*, v. 9, n. 2, p. 137–163, 2001.
- <a id="ref-manning2008"></a>MANNING, C. D.; RAGHAVAN, P.; SCHÜTZE, H. *Introduction to Information Retrieval*. Cambridge: Cambridge University Press, 2008.
- <a id="ref-salton1988"></a>SALTON, G.; BUCKLEY, C. Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, v. 24, n. 5, p. 513–523, 1988. DOI: 10.1016/0306-4573(88)90021-0.
- <a id="ref-strobl2007"></a>STROBL, C.; BOULESTEIX, A.-L.; ZEILEIS, A.; HOTHORN, T. Bias in random forest variable importance measures: illustrations, sources and a solution. *BMC Bioinformatics*, v. 8, art. 25, 2007. DOI: 10.1186/1471-2105-8-25.

**Statistics · Estatística**

- <a id="ref-friedman1937"></a>FRIEDMAN, M. The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *Journal of the American Statistical Association*, v. 32, n. 200, p. 675–701, 1937. DOI: 10.1080/01621459.1937.10503522.
- <a id="ref-holm1979"></a>HOLM, S. A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, v. 6, n. 2, p. 65–70, 1979.
- <a id="ref-wilcoxon1945"></a>WILCOXON, F. Individual comparisons by ranking methods. *Biometrics Bulletin*, v. 1, n. 6, p. 80–83, 1945. DOI: 10.2307/3001968.

**Record linkage, data engineering and accounting · Pareamento, engenharia de dados e contabilidade**

- <a id="ref-christen2012"></a>CHRISTEN, P. *Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection*. Berlin: Springer, 2012. DOI: 10.1007/978-3-642-31164-2.
- <a id="ref-coso2013"></a>COMMITTEE OF SPONSORING ORGANIZATIONS OF THE TREADWAY COMMISSION (COSO). *Internal Control — Integrated Framework*. 2013.
- <a id="ref-fellegi1969"></a>FELLEGI, I. P.; SUNTER, A. B. A theory for record linkage. *Journal of the American Statistical Association*, v. 64, n. 328, p. 1183–1210, 1969. DOI: 10.1080/01621459.1969.10501049.
- <a id="ref-kimball2004"></a>KIMBALL, R.; CASERTA, J. *The Data Warehouse ETL Toolkit*. Indianapolis: Wiley, 2004.
- <a id="ref-vasarhelyi2015"></a>VASARHELYI, M. A.; KOGAN, A.; TUTTLE, B. M. Big data in accounting: an overview. *Accounting Horizons*, v. 29, n. 2, p. 381–396, 2015.

**Brazilian standards and legislation · Normas e legislação brasileiras**

- <a id="ref-abrasf"></a>ASSOCIAÇÃO BRASILEIRA DAS SECRETARIAS DE FINANÇAS DAS CAPITAIS (ABRASF). *Nota Fiscal de Serviços Eletrônica (NFS-e): modelo conceitual*. Available at: <https://abrasf.org.br/biblioteca/arquivos-publicos/nfs-e>.
- <a id="ref-lc116"></a>BRASIL. Lei Complementar nº 116, de 31 de julho de 2003. Dispõe sobre o Imposto Sobre Serviços de Qualquer Natureza. Available at: <https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp116.htm>.
- <a id="ref-lei8212"></a>BRASIL. Lei nº 8.212, de 24 de julho de 1991 (art. 31 — retenção de 11% sobre serviços). Available at: <https://www.planalto.gov.br/ccivil_03/leis/l8212cons.htm>.
- <a id="ref-lei10833"></a>BRASIL. Lei nº 10.833, de 29 de dezembro de 2003 (arts. 30 a 36 — retenção de CSLL, COFINS e PIS/Pasep). Available at: <https://www.planalto.gov.br/ccivil_03/leis/2003/l10.833.htm>.
- <a id="ref-lei13137"></a>BRASIL. Lei nº 13.137, de 19 de junho de 2015. Available at: <https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2015/lei/l13137.htm>.
- <a id="ref-rir2018"></a>BRASIL. Decreto nº 9.580, de 22 de novembro de 2018 — Regulamento do Imposto sobre a Renda (RIR/2018). Available at: <https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/decreto/d9580.htm>.
- <a id="ref-inrfb2119"></a>BRASIL. Receita Federal do Brasil. Instrução Normativa RFB nº 2.119, de 6 de dezembro de 2022 — dispõe sobre o Cadastro Nacional da Pessoa Jurídica (CNPJ).
- <a id="ref-nbcta500"></a>CONSELHO FEDERAL DE CONTABILIDADE (CFC). *NBC TA 500 — Evidência de Auditoria*.
- <a id="ref-nbcta530"></a>CONSELHO FEDERAL DE CONTABILIDADE (CFC). *NBC TA 530 — Amostragem em Auditoria*. Aprovada pela Resolução CFC nº 1.222, de 27 de novembro de 2009.

**Software documentation · Documentação de software**

- <a id="ref-pedregosa2011"></a>PEDREGOSA, F. et al. Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, v. 12, p. 2825–2830, 2011.
- <a id="ref-sklearn-cv"></a>SCIKIT-LEARN. *Cross-validation: evaluating estimator performance*. Available at: <https://scikit-learn.org/stable/modules/cross_validation.html>.
- <a id="ref-sklearn-grid"></a>SCIKIT-LEARN. *Tuning the hyper-parameters of an estimator*. Available at: <https://scikit-learn.org/stable/modules/grid_search.html>.
- <a id="ref-sklearn-preprocessing"></a>SCIKIT-LEARN. *Preprocessing data*. Available at: <https://scikit-learn.org/stable/modules/preprocessing.html>.
- <a id="ref-python-hashseed"></a>PYTHON SOFTWARE FOUNDATION. *Command line and environment: `PYTHONHASHSEED`*. Available at: <https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED>.
