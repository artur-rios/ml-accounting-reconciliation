# Discussão, Limitações e Contribuições

Material de apoio para a redação dos capítulos finais do TCC — MBA em Engenharia de Software (USP/ESALQ).
Todos os números citados foram extraídos diretamente dos artefatos do repositório
(`data/results/*/metrics_summary.csv`, `data/results/ablation/ablation_summary.csv`,
`data/results/comparison/*.csv`, matrizes de confusão, `data/processed/*.csv`) e do código-fonte.

> **Aviso sobre as referências.** As obras citadas ao longo do texto são sugestões de ancoragem teórica.
> Elas são reais e amplamente conhecidas nas respectivas áreas, mas **você deve conferir autoria, ano,
> periódico, volume e páginas em uma base bibliográfica (Scopus, Web of Science, Google Scholar, Portal
> de Periódicos CAPES) antes de incorporá-las à monografia**. Trechos marcados com `[VERIFICAR]` exigem
> essa conferência. Não cite nenhuma obra que você não tenha lido pelo menos em resumo e conclusão.

---

## 1. Síntese dos achados que sustentam a discussão

### 1.1 Resultados consolidados do experimento principal

| Cenário | Algoritmo | Acurácia | F1-macro | Precisão-macro | Recall-macro | CV média ± dp | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exact | Random Forest | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 1,0000 ± 0,0000 | 1,0000 |
| exact | SVM (RBF) | 0,9136 | 0,8873 | 0,9449 | 0,8568 | 0,9080 ± 0,0092 | 0,9866 |
| exact | Regressão Logística | 0,9136 | 0,8873 | 0,9449 | 0,8568 | 0,9086 ± 0,0089 | 0,9713 |
| fuzzy | Random Forest | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 0,9996 ± 0,0007 | — |
| fuzzy | SVM (RBF) | 0,9209 | 0,8645 | 0,9567 | 0,8280 | 0,8636 ± 0,0101 | — |
| fuzzy | Regressão Logística | 0,9235 | 0,8684 | 0,9659 | 0,8304 | 0,8701 ± 0,0087 | — |

### 1.2 Composição das bases

- **Registros processados:** 7.518 pagamentos (de 7.500 solicitados; a diferença decorre de duplicidades
  de CNPJ resolvidas no *merge*). Conjunto de teste: 1.504 registros (20%, estratificado).
- **Cenário `exact`:** 5.250 conciliados (69,83%) e 2.268 não conciliados (30,17%).
- **Cenário `fuzzy`:** 5.250 conciliados (69,83%), 1.129 parcialmente conciliados (15,02%) e
  1.139 não conciliados (15,15%).
- **Composição da classe negativa (`exact`):** 1.119 registros sem correspondência na NFS-e (49,3% —
  erros de CNPJ e notas *ghost*), 582 divergências de data (25,7%) e 567 divergências de valor (25,0%).

### 1.3 Matrizes de confusão determinantes

**SVM e Regressão Logística, cenário `exact`** (matrizes idênticas nos dois modelos):

|  | Predito 0 | Predito 1 |
| --- | --- | --- |
| **Real 0** | 324 | 130 |
| **Real 1** | 0 | 1.050 |

Recall da classe negativa: 71,37%. Recall da classe positiva: 100,00%. **Zero falsos negativos, 130 falsos positivos.**

**Regressão Logística, cenário `fuzzy`:**

|  | Predito 0 | Predito 1 | Predito 2 |
| --- | --- | --- | --- |
| **Real 0** | 228 | 0 | 0 |
| **Real 1** | 1 | 111 | 114 |
| **Real 2** | 0 | 0 | 1.050 |

**114 dos 226 registros parcialmente conciliados (50,44%) foram promovidos à classe "conciliado".**
As classes extremas foram classificadas sem erro.

---

## 2. Discussão dos resultados à luz da literatura

### 2.1 O desempenho perfeito da Random Forest é um diagnóstico, não um resultado

O achado mais relevante deste trabalho não é que a Random Forest superou os demais classificadores, mas
**por que** ela alcançou 1,0000 em todas as métricas, nos dois cenários, com desvio-padrão de validação
cruzada exatamente igual a zero no cenário `exact`.

A inspeção do pipeline revela circularidade entre rotulagem e representação. O módulo de rotulagem
(`src/reconciliacao/etl/labeler.py`) define o rótulo como uma regra determinística de limiar sobre duas
grandezas:

```
label = 1  ⟺  delta_days ≤ tolerância_data  ∧  delta_valor_pct ≤ tolerância_valor
```

O módulo de engenharia de atributos (`src/reconciliacao/models/features.py`) entrega **essas mesmas duas
grandezas** ao classificador como variáveis preditoras. O alvo é, portanto, uma função exata e
inteiramente conhecida de duas das sete variáveis de entrada. O problema submetido aos algoritmos não é
"aprender a conciliar", mas "recuperar uma conjunção de dois limiares axialmente alinhados a partir de
exemplos" — tarefa para a qual o particionamento recursivo de árvores é a hipótese indutiva perfeitamente
adequada.

A evidência empírica é inequívoca: no gráfico de importância de variáveis do cenário `exact`,
`delta_valor_pct` (≈0,46) e `delta_days` (≈0,42) concentram aproximadamente **87% da importância total**.
As demais cinco variáveis somam ≈13%, e `cnpj_match` tem importância exatamente zero.

Este é o fenômeno que Kaufman, Rosset, Perlich e Stitelman descreveram como *leakage* — a introdução, no
conjunto de variáveis preditoras, de informação que não estaria legitimamente disponível no momento da
predição, ou que é derivada do próprio alvo `[VERIFICAR: Kaufman, S.; Rosset, S.; Perlich, C.;
Stitelman, O. "Leakage in Data Mining: Formulation, Detection, and Avoidance". ACM Transactions on
Knowledge Discovery from Data, v. 6, n. 4, 2012]`. Os autores destacam que o sintoma clássico do
vazamento é justamente o desempenho implausivelmente alto e estável — precisamente o padrão observado
aqui (acurácia unitária com dp de validação cruzada nulo).

O experimento de ablação apresentado na **seção 3** confirma esse diagnóstico de forma quantitativa: ao
remover as duas variáveis da regra, o F1-macro da Random Forest cai de 1,0000 para 0,7568 no cenário
`exact` e para 0,6538 no cenário `fuzzy`.

**Como redigir isso na monografia.** Não apresente o 1,0000 como sucesso. Apresente-o como *achado
metodológico*: o experimento demonstra que, quando a definição operacional de conciliação é convertida
em variável preditora, qualquer modelo com capacidade de particionamento axial reproduz a regra
trivialmente, e a comparação entre algoritmos perde poder discriminante. Essa é uma advertência
diretamente transferível para equipes que pretendam aplicar aprendizado supervisionado a conciliação
contábil a partir de rótulos gerados por regras internas — situação extremamente comum na prática, já
que poucas organizações dispõem de rótulos auditados independentemente.

A literatura de aprendizado de máquina aplicado à contabilidade e auditoria vem alertando para a
distância entre acurácia reportada e utilidade operacional
`[VERIFICAR: Bao, Y.; Ke, B.; Li, B.; Yu, Y. J.; Zhang, J. "Detecting Accounting Fraud in Publicly
Traded U.S. Firms Using a Machine Learning Approach". Journal of Accounting Research, v. 58, n. 1,
2020]` e `[VERIFICAR: Appelbaum, D.; Kogan, A.; Vasarhelyi, M. A. "Big Data and Analytics in the Modern
Audit Engagement: Research Needs". Auditing: A Journal of Practice & Theory, v. 36, n. 4, 2017]`. Este
trabalho oferece uma instância concreta, reprodutível e quantificada desse descolamento.

### 2.2 Por que SVM e Regressão Logística estacionam em ~0,91: uma explicação geométrica

O contraste entre a Random Forest (1,0000) e os modelos de fronteira suave (0,9136) **não decorre de
superioridade conceitual das florestas aleatórias para o domínio de conciliação**. Decorre de uma
interação entre a codificação de ausência de correspondência e a padronização de escala.

Registros de pagamento sem NFS-e correspondente (1.119 casos, 14,9% da base) recebem valores-sentinela:
`delta_days = 9999` e `delta_valor_pct = 100`. A Random Forest é invariante a transformações monotônicas
das variáveis — apenas a ordem importa para a escolha de pontos de corte. Já SVM e Regressão Logística
operam sobre variáveis padronizadas por `StandardScaler` (`src/reconciliacao/models/trainer.py:39,55`), e
a sentinela distorce severamente média e desvio-padrão.

Calculando sobre a base `exact`, `delta_days` apresenta média 1.490,08 e desvio-padrão 3.558,24. Após a
padronização:

| `delta_days` original | Valor padronizado (z) |
| --- | --- |
| 0 dias | −0,4188 |
| 5 dias | −0,4174 |
| 6 dias | −0,4171 |
| 30 dias | −0,4103 |
| 9999 (sentinela) | +2,3913 |

**Toda a faixa informativa de `delta_days` — de 0 a 30 dias, que é onde reside a decisão de conciliação —
comprime-se em um intervalo de 0,0085 desvios-padrão**, enquanto a sentinela ocupa sozinha 2,8 desvios.
Para um classificador de margem, a diferença entre um pagamento na data exata e um pagamento com 30 dias
de defasagem torna-se numericamente indistinguível de ruído.

A variável `delta_valor_pct` sofre distorção muito menor (0% → −0,4480; 6% → −0,2800; diferença de 0,168
desvios), o que explica precisamente o padrão de erros observado: **os modelos suaves continuam separando
divergências de valor, mas perdem quase completamente a capacidade de separar divergências de data.** Os
130 falsos positivos do cenário `exact` são compatíveis, em ordem de grandeza, com os ~116 registros de
divergência de data presentes no conjunto de teste. No cenário `fuzzy`, os 114 registros parciais
promovidos indevidamente à classe "conciliado" seguem a mesma lógica.

Note-se ainda que **SVM e Regressão Logística produziram matrizes de confusão rigorosamente idênticas no
cenário `exact`** — mesmos 324/130/0/1.050. Dois modelos de famílias distintas convergindo para a mesma
partição do espaço de teste é indício adicional de que ambos colapsaram para essencialmente a mesma
decisão univariada sobre `delta_valor_pct`.

**Esta hipótese foi testada diretamente e confirmada** (seção 3.3): substituindo a sentinela por
indicador explícito de correspondência, SVM e Regressão Logística saltam de 0,8873 para **1,0000** de
F1-macro no cenário `exact`, eliminando integralmente a diferença em relação à Random Forest. A conclusão
apropriada, portanto, não é "use Random Forest para conciliação", e sim: **a diferença de desempenho
entre os três algoritmos era, em sua totalidade, um artefato de pré-processamento.**

A literatura clássica de comparação empírica de classificadores oferece o enquadramento adequado:
o teorema *No Free Lunch* `[VERIFICAR: Wolpert, D. H. "The Lack of A Priori Distinctions Between Learning
Algorithms". Neural Computation, v. 8, n. 7, 1996]` e o levantamento de Fernández-Delgado e colegas sobre
121 conjuntos de dados `[VERIFICAR: Fernández-Delgado, M. et al. "Do we Need Hundreds of Classifiers to
Solve Real World Classification Problems?". Journal of Machine Learning Research, v. 15, 2014]` sustentam
que o desempenho relativo de algoritmos é condicionado à representação dos dados, não intrínseco.

### 2.3 A assimetria dos erros e o custo em contexto de auditoria

O achado de maior relevância prática está na **estrutura** dos erros, não em sua quantidade. No cenário
`exact`, SVM e Regressão Logística cometeram **130 falsos positivos e nenhum falso negativo**.

Em conciliação financeira, essas duas categorias têm consequências assimétricas:

- **Falso negativo** (pagamento legítimo classificado como não conciliado): gera trabalho manual
  desnecessário. Custo operacional, mas o controle permanece íntegro.
- **Falso positivo** (pagamento divergente classificado como conciliado): **a divergência é silenciada.**
  Um pagamento com valor 3% superior ao da nota fiscal, ou emitido 30 dias fora da competência, recebe
  aprovação automática e nunca chega à revisão humana.

O modelo, tal como configurado (`class_weight: balanced`), otimiza F1-macro e não incorpora custo
diferenciado. O resultado é um classificador que **erra sistematicamente na direção mais custosa para o
controle interno**. No cenário `fuzzy`, o efeito é ainda mais pronunciado: metade dos registros
parcialmente conciliados recebe atestado de conformidade plena.

A ablação mostra que esse viés **se agrava** quando o vazamento é removido: sem as variáveis da regra, a
Random Forest passa a cometer 250 falsos positivos contra zero falsos negativos no cenário `exact`
(seção 3.4). A tendência a silenciar exceções não é acidental — decorre da combinação entre desbalanço de
classes (69,8% positivos) e métrica de seleção simétrica.

Este ponto conecta o trabalho diretamente à teoria de aprendizado sensível a custo
`[VERIFICAR: Elkan, C. "The Foundations of Cost-Sensitive Learning". In: Proceedings of the 17th
International Joint Conference on Artificial Intelligence (IJCAI), 2001]` e às estruturas normativas de
controle interno, que tratam a detecção de exceções como objetivo de controle e não como métrica de
eficiência `[VERIFICAR: COSO. Internal Control — Integrated Framework, 2013]` e
`[VERIFICAR: NBC TA 530 — Amostragem em Auditoria, Conselho Federal de Contabilidade]`.

Recomendação derivada: **em conciliação, a métrica de seleção de modelo deve ser o recall da classe de
exceção, sob restrição de precisão mínima aceitável — não F1-macro nem acurácia.** Um sistema que
encaminha 15% dos registros para revisão manual mas não deixa passar nenhuma divergência é
operacionalmente superior a um que automatiza 91% silenciando 29% das exceções.

Observe-se também a discrepância entre ROC-AUC e F1-macro: o SVM no cenário `exact` atinge AUC de 0,9866
com F1-macro de apenas 0,8873. Com classes desbalanceadas, a curva ROC produz impressão otimista do
desempenho, fenômeno documentado por `[VERIFICAR: Saito, T.; Rehmsmeier, M. "The Precision-Recall Plot
Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets". PLoS
ONE, v. 10, n. 3, 2015]`. A ablação fornece ilustração ainda mais didática: no cenário `fuzzy` sem
vazamento, o SVM alcança **acurácia de 0,8497 sem jamais predizer a classe "parcialmente conciliado"**
(recall da classe 1 = 0,0000). Vale registrar essa observação como limitação metodológica do próprio
conjunto de métricas adotado.

### 2.4 O efeito do rigor do rótulo: o cenário *fuzzy* não testou tolerância

A segunda pergunta de pesquisa — como o rigor do rótulo afeta o ranking dos algoritmos — merece resposta
mais cuidadosa do que a tabela de métricas sugere.

O gerador de dados sintéticos (`src/reconciliacao/simulation/xml_generator.py:53,58`) produz divergências
de data por deslocamento sorteado no intervalo de **6 a 30 dias**, e divergências de valor por fator
sorteado entre **3% e 20%**. As tolerâncias do cenário `fuzzy` são de **±5 dias** e **±2%**. Logo,
**por construção nenhum registro cai dentro da janela de tolerância**.

A verificação empírica confirma: na base `fuzzy`, todos os 5.250 registros da classe "plenamente
conciliado" apresentam `delta_days = 0` e `delta_valor_pct = 0` exatos — valores máximos e mínimos
coincidentes. As janelas de tolerância estão **vazias**.

Isso significa que o cenário `fuzzy` **não avaliou robustez a tolerâncias**. O que ele efetivamente fez
foi subdividir a classe negativa em duas, aplicando a regra de proximidade (4× a tolerância de data *ou*
5× a tolerância de valor). A conclusão legítima é, portanto, mais restrita e mais interessante:

> O aumento da granularidade do rótulo — de binário para ternário — degradou o desempenho dos modelos de
> fronteira suave (F1-macro de 0,8873 para 0,8645 no SVM) sem afetar a Random Forest, porque a classe
> intermediária é definida por limiares sobre as mesmas variáveis vazadas. A dificuldade adicional recai
> integralmente sobre a fronteira entre "parcial" e "conciliado", que é justamente a região comprimida
> pela padronização.

Redija a resposta à segunda pergunta de pesquisa nesses termos. Afirmar que o estudo demonstrou robustez
a tolerâncias seria insustentável diante de uma banca que examine o gerador.

### 2.5 Variáveis inertes e o limite da informação disponível

Três das sete variáveis não carregam informação neste desenho experimental:

- **`cnpj_match`** é constante e igual a 1 em toda a base. A verificação `df["cnpj_fornecedor"].notna()`
  (`features.py:29`) nunca é falsa, pois o CNPJ do pagamento sempre existe. Importância medida: zero.
  Mais relevante: como o *merge* é feito sobre o CNPJ, **erros de CNPJ nunca chegam ao classificador** —
  são resolvidos a montante, transformando-se em ausência de correspondência.
- **`descricao_similarity`** é ruído por construção. As descrições de pagamento e de NFS-e são geradas
  por chamadas independentes a `Faker.bs()`, sem qualquer relação semântica. O primeiro registro da base,
  que é *conciliado*, tem descrição de pagamento "embrace proactive niches" e discriminação de NFS-e
  "enhance global platforms". Importância medida: ≈0,002.
- **`nfse_valor_iss` e `nfse_aliquota`** (importâncias ≈0,05 e ≈0,07) não representam sinal contábil
  genuíno: o ISS é calculado deterministicamente como `valor × alíquota / 100` e é preenchido com zero
  nos registros sem correspondência. Funcionam, portanto, como **proxies da própria existência da
  correspondência** — uma segunda via de vazamento, mais sutil que a primeira. A ablação confirma
  (seção 3.4): removidas as variáveis da regra, o que os modelos ainda detectam é *exclusivamente* a
  ausência de nota fiscal.

A consequência é que o modelo dispõe, na prática, de duas variáveis reais, ambas derivadas do alvo. Isso
delimita com precisão o alcance do que foi demonstrado.

Vale acrescentar que a importância por impureza da Random Forest, aqui utilizada, é reconhecidamente
enviesada em favor de variáveis contínuas de alta cardinalidade
`[VERIFICAR: Strobl, C.; Boulesteix, A.-L.; Zeileis, A.; Hothorn, T. "Bias in Random Forest Variable
Importance Measures: Illustrations, Sources and a Solution". BMC Bioinformatics, v. 8, 2007]` — o que
recomenda cautela adicional na leitura dos valores absolutos, ainda que a hierarquia observada seja
robusta o bastante para sustentar o argumento.

### 2.6 Diálogo com a literatura de *record linkage*

A tarefa aqui tratada é, formalmente, um problema de **pareamento de registros** (*record linkage*), campo
com fundamentação teórica consolidada desde o modelo probabilístico de Fellegi e Sunter
`[VERIFICAR: Fellegi, I. P.; Sunter, A. B. "A Theory for Record Linkage". Journal of the American
Statistical Association, v. 64, n. 328, p. 1183-1210, 1969]`, sistematizado em obra de referência por
Christen `[VERIFICAR: Christen, P. Data Matching: Concepts and Techniques for Record Linkage, Entity
Resolution and Duplicate Detection. Springer, 2012]`.

O confronto com essa literatura evidencia uma diferença estrutural relevante. No modelo de Fellegi-Sunter,
o objeto de decisão é o **par candidato** (pagamento *i*, nota *j*), e o problema central é a geração de
candidatos — o *blocking* — porque o espaço de pares cresce quadraticamente. Neste trabalho, a simulação
gera CNPJs únicos por pagamento e exatamente uma NFS-e por pagamento, de modo que a junção é **1:1 por
construção**. O problema de geração de candidatos, que é o núcleo computacional e conceitual do
*record linkage* real, foi eliminado antes da modelagem.

Esse é o ponto em que o trabalho mais se afasta da conciliação praticada em organizações, onde é rotineiro
que um pagamento quite várias notas, que uma nota seja quitada em parcelas, ou que existam múltiplos
fornecedores com dados cadastrais semelhantes. Reconhecer isso explicitamente fortalece a monografia, e
converte a limitação em agenda de pesquisa bem delimitada (seção 8).

---

## 3. Experimento complementar de ablação

Para submeter a teste as duas hipóteses levantadas nas seções 2.1 e 2.2, foi executado um experimento
complementar de ablação, implementado em `run_ablation.py`. O experimento mantém constantes os dados, a
divisão treino/teste, a semente aleatória (42), a grade de hiperparâmetros e o protocolo de validação
cruzada (5 dobras estratificadas, F1-macro), **variando exclusivamente a representação das variáveis**.
Os artefatos originais em `data/results/exact|fuzzy/` não foram sobrescritos; a ablação escreve em
`data/results/ablation/`.

### 3.1 Configurações avaliadas

| Configuração | Variáveis | Hipótese testada |
| --- | --- | --- |
| `full` | As sete originais | Linha de base (replica o experimento principal) |
| `no_leak` | Sem `delta_days` e `delta_valor_pct` | Quanto do desempenho decorre do vazamento direto (§2.1) |
| `no_leak_strict` | Também sem `nfse_valor_iss` e `nfse_aliquota` | Quanto decorre do vazamento indireto pelos proxies de ISS (§2.5) |
| `sentinel_fixed` | As sete originais + `has_match`, com imputação neutra no lugar das sentinelas | Se a diferença entre RF e modelos suaves é artefato de escala (§2.2) |

Na configuração `sentinel_fixed`, os registros sem correspondência recebem `has_match = 0` e têm
`delta_days` e `delta_valor_pct` imputados pela mediana dos registros pareados — que, nesta base, é
exatamente 0,0 para ambas as variáveis. A informação de ausência passa a ser carregada por uma variável
binária, e não por um valor extremo na escala contínua.

### 3.2 Resultados (F1-macro)

**Cenário `exact`:**

| Configuração | Random Forest | SVM (RBF) | Regressão Logística |
| --- | --- | --- | --- |
| `full` (linha de base) | **1,0000** | 0,8873 | 0,8873 |
| `sentinel_fixed` | **1,0000** | **1,0000** | **1,0000** |
| `no_leak` | 0,7568 | **0,7808** | 0,6799 |
| `no_leak_strict` | **0,4966** | 0,4111 | 0,4709 |

**Cenário `fuzzy`:**

| Configuração | Random Forest | SVM (RBF) | Regressão Logística |
| --- | --- | --- | --- |
| `full` (linha de base) | **1,0000** | 0,8645 | 0,8684 |
| `sentinel_fixed` | **1,0000** | 0,9985 | **1,0000** |
| `no_leak` | **0,6538** | 0,6337 | 0,6336 |
| `no_leak_strict` | **0,3517** | 0,2741 | 0,2526 |

A configuração `full` reproduziu os valores do experimento principal com precisão de quatro casas
decimais, apesar de executada sob versões mais recentes das bibliotecas (scikit-learn 1.9.0, pandas
3.0.5). Isso constitui verificação independente da reprodutibilidade do pipeline.

### 3.3 A diferença entre algoritmos era integralmente artefato de escala

Na configuração `sentinel_fixed`, **os três algoritmos atingem matrizes de confusão idênticas e perfeitas**
no cenário `exact`: 454 verdadeiros negativos, 1.050 verdadeiros positivos, **zero falsos positivos e zero
falsos negativos**. A validação cruzada da Regressão Logística registra 1,0000 ± 0,0000 e a do SVM
0,9998 ± 0,0004.

A diferença de 9,0 pontos percentuais de F1-macro entre a Random Forest e os modelos de fronteira suave —
apresentada no experimento principal como evidência de superioridade da floresta aleatória — **desaparece
por completo com uma única alteração de pré-processamento**, sem qualquer mudança de algoritmo, de
hiperparâmetro ou de dados.

Esta é a confirmação direta da hipótese da seção 2.2. A explicação é geometricamente transparente: uma vez
removida a sentinela, os registros conciliados ocupam exatamente a origem do plano
(`delta_days` = 0, `delta_valor_pct` = 0) e a classe é linearmente separável das demais com o auxílio de
`has_match`. O que impedia a Regressão Logística de encontrar essa fronteira não era sua natureza linear,
mas o esmagamento da escala provocado pelo valor 9999.

**Consequência para a monografia:** a resposta à primeira pergunta de pesquisa — qual algoritmo melhor
automatiza a conciliação — deve ser reformulada. Sob representação adequada, os três algoritmos são
equivalentes. A pergunta relevante não é qual algoritmo escolher, mas como representar os dados.

### 3.4 Sem vazamento, os modelos detectam apenas a ausência de nota fiscal

Na configuração `no_leak`, o desempenho cai substancialmente e — resultado notável — **a ordem entre os
algoritmos se inverte** no cenário `exact`: o SVM (0,7808) supera a Random Forest (0,7568), que por sua
vez supera a Regressão Logística (0,6799). A conclusão de que "a Random Forest é o melhor algoritmo para
conciliação" não sobrevive à remoção do vazamento.

Mais importante é a **natureza** do que resta ser detectado. Decompondo os acertos na classe negativa do
conjunto de teste (203 registros sem correspondência e 251 registros pareados mas divergentes):

| Modelo (`no_leak`, `exact`) | Sem correspondência detectados | Divergentes detectados |
| --- | --- | --- |
| Random Forest | 203 / 203 (100,0%) | 1 / 251 (0,4%) |
| SVM (RBF) | 203 / 203 (100,0%) | 22 / 251 (8,8%) |

Ou seja: **sem as variáveis da regra, os modelos identificam com perfeição os pagamentos que não possuem
nota fiscal alguma, e são praticamente cegos às divergências de data e de valor.** Detectar a ausência de
correspondência, porém, não é aprendizado de máquina — é o resultado de uma junção relacional, obtenível
por uma consulta SQL de uma linha. O sinal aparentemente residual do modelo `no_leak` é, ele próprio,
vazamento indireto: `nfse_valor_iss` e `nfse_aliquota` valem zero exatamente quando não há nota.

A configuração `no_leak_strict` confirma o argumento ao remover também esses proxies. O desempenho colapsa
para a faixa de 0,41 a 0,50 de F1-macro no cenário `exact` e de 0,25 a 0,35 no `fuzzy`. O SVM degenera
para o classificador trivial: prediz a classe majoritária em 100% dos casos, obtendo acurácia de 0,6981 —
exatamente a proporção de registros conciliados na base (1.050/1.504) — com recall nulo na classe de
exceção.

**Conclusão do experimento de ablação:** descontado o vazamento direto e indireto, o conjunto de dados
sintéticos **não contém sinal aprendível para a tarefa de conciliação**. Isso não invalida o trabalho —
delimita com precisão o que ele demonstra, e transforma o que seria uma fragilidade oculta em resultado
metodológico explícito e mensurado.

### 3.5 Limitações do próprio experimento de ablação

- A mediana usada na imputação de `sentinel_fixed` foi calculada sobre a base completa e não apenas sobre
  a partição de treino, configurando vazamento transdutivo de estatística agregada. Como a mediana é
  exatamente 0,0 — valor que seria escolhido por qualquer critério neutro — o efeito é imaterial, mas a
  ressalva deve constar.
- A configuração `sentinel_fixed` **não corrige o vazamento de rótulo**: ela mantém `delta_days` e
  `delta_valor_pct` entre as variáveis. Seu propósito é isolar o efeito da escala, não produzir um modelo
  válido. O 1,0000 obtido ali continua sendo consequência da circularidade descrita em 2.1.
- Não foram aplicados testes de significância estatística às diferenças entre configurações
  `[VERIFICAR: Demšar, J. "Statistical Comparisons of Classifiers over Multiple Data Sets". JMLR, v. 7,
  2006]`.
- O experimento roda sob semente única (42), herdada de `config.yaml`.

---

## 4. Experimento de comparação válida: qual algoritmo, afinal

As seções 2 e 3 estabeleceram que a comparação original não tinha poder discriminante. Esta seção
apresenta o experimento corretivo que responde à primeira pergunta de pesquisa — e, na §4.6, um terceiro
achado metodológico que só apareceu quando a primeira versão deste próprio experimento foi auditada.

O desenho está em `docs/superpowers/specs/2026-09-10-comparacao-algoritmos-design.md`; reproduz-se com
`python run_comparison.py`.

### 4.1 O que mudou no desenho

Três alterações, todas na raiz do problema diagnosticado:

1. **O rótulo passa a vir do gerador, não de uma regra sobre as features.** O simulador sabe qual
   pagamento quita qual nota e persiste esse fato. Cada pagamento referencia explicitamente uma nota
   candidata, e o rótulo diz se aquele pareamento é genuíno. O desacoplamento é por construção, não por
   remoção de variáveis.
2. **Pares verdadeiros deixam de ser cópias.** O pagamento é derivado da nota por transformação legal —
   retenções tributárias brasileiras (IRRF, CSRF, ISS retido, INSS) e prazo característico do fornecedor.
   As divergências são sorteadas de faixa contínua sobreposta às retenções legítimas, de modo que uma
   razão de valor de 0,9535 deixe de significar "não conciliado" e passe a significar "compatível com
   retenção de PIS/COFINS/CSLL".
3. **A seleção de modelo passa a derivar do objetivo de controle.** A métrica é o recall da classe de
   exceção sob precisão ≥ 0,90, com o limiar escolhido na partição de validação e aplicado à de teste,
   reportado ao lado da taxa de encaminhamento à revisão manual.

Cada fornecedor emite **cinco notas**, de modo que seu prazo característico seja estimável a partir do
histórico — condição sem a qual a variável `desvio_prazo_fornecedor` não teria referente, como a §4.6
detalha. A base tem 7.500 pagamentos, 1.500 fornecedores e 30% de exceções.

Como consequência do item 1, **não há mais pagamentos sem nota candidata** — as deltas são sempre
computáveis e os valores-sentinela desaparecem. A patologia de escala descrita em §2.2, responsável
integral pela diferença entre algoritmos no experimento original, deixa de existir por construção.

### 4.2 O guarda anti-vazamento aprovou a base

Antes de treinar, um guarda ajusta uma árvore de profundidade 1 sobre **cada variável isoladamente** e
falha o pipeline se alguma ultrapassar 0,95 de acurácia. Nas dez execuções, a maior acurácia de variável
isolada foi **0,8173**.

| Variável | Acurácia média do toco |
| --- | --- |
| `razao_valor` | 0,8131 |
| `retencao_implicita_pct` | 0,8131 |
| `similaridade_descricao` | 0,7852 |
| `mesmo_municipio` | 0,7750 |
| `desvio_prazo_fornecedor` | 0,7711 |
| `delta_dias` | 0,7429 |

As seis mais informativas de catorze. As oito restantes ficam em 0,7302 ou abaixo, e sete delas — os
indicadores `compat_*` e `acima_limite_csrf` — marcam exatamente 0,70, a taxa da classe majoritária: um
exemplo vivo do que o guarda não consegue enxergar.

Nenhuma variável separa as classes sozinha, e as mais informativas são evidências de nível de par, não a
própria regra de rotulagem. A tarefa submetida aos algoritmos é genuína.

Registre-se, porém, o limite deste instrumento: **o guarda não detecta o defeito descrito em §4.6**. Uma
variável constante marca exatamente a taxa da classe majoritária, que está muito abaixo do teto — ela
passa no teste justamente por não conter informação nenhuma.

### 4.3 Resultados

Dez bases independentes, 7.500 registros cada, três algoritmos, mesma divisão treino/validação/teste
(60/20/20 estratificado) e mesma grade de hiperparâmetros do experimento original.

| Algoritmo | Recall da exceção | dp | Precisão da exceção | Taxa de encaminhamento | PR-AUC | Sementes aplicáveis |
| --- | --- | --- | --- | --- | --- | --- |
| **Random Forest** | **0,8216** | 0,0169 | 0,8931 | 27,62% | 0,9356 | 10/10 |
| SVM (RBF) | 0,8002 | 0,0191 | 0,8996 | 26,71% | 0,9258 | 10/10 |
| Regressão Logística | 0,7698 | 0,0251 | 0,9030 | 25,59% | 0,9153 | 10/10 |

Os três atingiram o piso de precisão na validação em todas as dez execuções. A precisão **de teste** fica
ligeiramente abaixo de 0,90 para a Random Forest (0,893) e para o SVM (0,900), e acima para a Regressão
Logística (0,903) — o que é esperado e desejável: o limiar foi
calibrado na validação e aplicado ao teste sem reajuste. Uma precisão de teste exatamente no piso seria
indício de que o teste havia sido usado na calibração.

**Wilcoxon pareado, correção de Holm para três comparações:**

| Comparação | Diferença mediana | p | p (Holm) | Significativo |
| --- | --- | --- | --- | --- |
| Random Forest × SVM | +1,44 p.p. | 0,0020 | 0,0059 | sim |
| Random Forest × Regressão Logística | +4,56 p.p. | 0,0020 | 0,0059 | sim |
| SVM × Regressão Logística | +3,00 p.p. | 0,0039 | 0,0059 | sim |

### 4.4 A resposta à primeira pergunta de pesquisa

> Sob rotulagem independente das features e critério derivado do objetivo de controle, a **Random Forest**
> detecta **1,44 pontos percentuais** a mais de divergências que o SVM e **4,56** a mais que a Regressão
> Logística, mantendo o piso de precisão de 0,90 na validação (Wilcoxon pareado, *p* = 0,0059 após
> correção de Holm, 10 execuções), ao custo de encaminhar **27,6%** do lote para revisão manual.

A significância estatística não deve ser confundida com relevância prática. A diferença entre a Random
Forest e o SVM é de **1,44 pontos percentuais**: em uma carteira de 7.500 pagamentos com 30% de exceções,
cerca de 32 divergências a mais. O teste a detecta porque a ordem se repete nas dez execuções, não porque
a magnitude seja grande — que é exatamente a razão de o tamanho de efeito ser reportado ao lado do
p-valor `[VERIFICAR: Demšar, J. JMLR, v. 7, 2006]`.

A leitura defensável é, portanto, mais matizada do que "a Random Forest vence": **os três algoritmos são
operacionalmente próximos**, com a Regressão Logística cerca de 4,6 pontos atrás, e a escolha entre eles
pode legitimamente recair sobre interpretabilidade e custo em vez de desempenho. Para um sistema que
precisa justificar cada exceção a um auditor, os coeficientes da regressão logística podem valer mais que
4,6 pontos de recall `[VERIFICAR: Rudin, C. Nature Machine Intelligence, v. 1, 2019]`.

### 4.5 Estabilidade

O desvio-padrão entre sementes é pequeno e semelhante nos três: 0,0169 para a Random Forest, 0,0191 para
o SVM e 0,0251 para a Regressão Logística. Nenhum algoritmo apresenta comportamento bimodal ou regimes
distintos.

| Semente | Random Forest | SVM | Regressão Logística |
| --- | --- | --- | --- |
| 0 | 0,829 | 0,793 | 0,738 |
| 1 | 0,820 | 0,791 | 0,742 |
| 2 | 0,829 | 0,824 | 0,789 |
| 3 | 0,840 | 0,827 | 0,776 |
| 4 | 0,811 | 0,764 | 0,738 |
| 5 | 0,851 | 0,802 | 0,809 |
| 6 | 0,802 | 0,793 | 0,780 |
| 7 | 0,796 | 0,789 | 0,764 |
| 8 | 0,813 | 0,798 | 0,764 |
| 9 | 0,824 | 0,820 | 0,798 |

A Random Forest é a melhor em todas as dez execuções, o que sustenta a significância apesar da margem
estreita.

Esta tabela merece atenção especial porque **a primeira versão deste experimento produzia um SVM
bimodal** — seis execuções perto de 0,31 e quatro perto de 0,08 — e este documento chegou a atribuir esse
padrão à calibração por `CalibratedClassifierCV(ensemble=False)`. A explicação estava errada, e a §4.6
mostra por quê.

### 4.6 Terceiro achado: o experimento corretivo reproduziu um artefato de representação

O trabalho já havia documentado dois modos de falha silenciosa: o vazamento de rótulo (§2.1) e a métrica
de seleção apontada para a classe errada (§4.7). A auditoria final deste experimento revelou um terceiro,
**dentro do próprio experimento construído para evitá-los**.

**O defeito.** A primeira versão do gerador emitia um pagamento por CNPJ. A variável
`desvio_prazo_fornecedor` é definida como o desvio entre o prazo observado e a **mediana histórica do
fornecedor**, ajustada apenas na partição de treino. Com um único pagamento por fornecedor, essa mediana
é o próprio valor da linha, e a variável era identicamente **zero em todas as linhas de treino**. Nenhum
fornecedor do teste aparecia no treino, de modo que todos recorriam à mediana global e a coluna tinha
dispersão real fora do treino.

**A consequência.** Não foi uma variável inerte, foi um artefato de escala. Medido sobre 2.000 registros:

| | Treino | Teste |
| --- | --- | --- |
| Valores distintos | 1 | — |
| Desvio-padrão | 0,00 | 25,48 |
| Fornecedores vistos no treino | — | 0 de 400 |

O `StandardScaler` encontra variância zero, aplica sua guarda (`scale_ = 1`), e as linhas de teste passam
sem normalização para uma dimensão que o modelo nunca viu variar: **83,75% delas com |z| > 5, máximo 90**.
O núcleo RBF colapsa. A Random Forest, invariante a transformações monotônicas, e a Regressão Logística,
cujo coeficiente para uma coluna constante é zero, não são afetadas — **só o SVM**.

**O efeito sobre a conclusão.** Corrigido o gerador para que cada fornecedor emita cinco notas, tornando a
mediana uma estatística real:

| | Antes | Depois |
| --- | --- | --- |
| Recall de exceção do SVM | 0,2211 ± 0,1248 | 0,8002 ± 0,0191 |
| Posição do SVM | último, 51,9 p.p. atrás | segundo, 1,4 p.p. atrás |
| Bimodalidade | quatro sementes em ~0,08 | ausente |

A conclusão anterior — "ambas superam o SVM por margens acima de 48 pontos percentuais" — era
**inteiramente artefato**. E a explicação por calibração oferecida para a bimodalidade era uma
racionalização plausível de um sintoma cuja causa era outra.

**Por que o guarda anti-vazamento não pegou.** Ele testa se alguma variável prediz o rótulo *bem demais*.
Uma variável constante prediz o rótulo *mal* — marca exatamente a taxa da classe majoritária. O
instrumento estava apontado para o excesso de informação, e o defeito era ausência de informação com
distribuições de treino e teste incompatíveis. **A verificação que teria pego é outra: exigir que nenhuma
coluna seja constante na partição de treino**, e que treino e teste tenham escalas comparáveis. Essa
asserção passou a existir na suíte de testes.

**A lição, e por que ela pertence à monografia.** Os três achados são a mesma família — configuração
silenciosa que produz conclusão substantiva e errada, sem se manifestar como erro:

| Achado | Sintoma | Conclusão falsa que produziria |
| --- | --- | --- |
| Vazamento de rótulo (§2.1) | acurácia 1,0000 | "a Random Forest resolve conciliação" |
| Scorer na classe errada (§4.7) | SVM em 0,088 | "o SVM não serve para conciliação" |
| Variável constante no treino (§4.6) | SVM em 0,221, bimodal | "o SVM é instável em conciliação" |

Os três produziriam tabelas plausíveis. Dois deles foram encontrados por revisão adversarial do código, e
não pelas métricas — que em nenhum dos casos exibiram sinal de anomalia. A contribuição metodológica não é
"verifique vazamento": é que **a verificação precisa ser sistemática e adversarial, porque cada instrumento
só enxerga o modo de falha para o qual foi construído**. O guarda anti-vazamento é útil e não teria
detectado dois dos três.

**Ressalva de honestidade.** O artefato foi reduzido, não eliminado. Com cinco pagamentos por fornecedor,
uma linha de treino contribui para a mediana do próprio fornecedor, de modo que seu desvio é encolhido em
relação ao de uma linha de teste: na configuração de 7.500 registros, desvio-padrão de cerca de 13,8 no
treino contra 18,5 no teste. É o mesmo mecanismo em amplitude muito menor — as linhas extremas ficam
perto de |z| ≈ 5, uma cauda fina, não os 83,75% que colapsaram o núcleo. Uma mediana *leave-one-out*, ou mais pagamentos por fornecedor, removeria o
resíduo. Fica registrado como limitação 27.

### 4.7 Achado anterior: a métrica de seleção apontava para a classe errada

Durante a implementação, descobriu-se que o `GridSearchCV` estava selecionando hiperparâmetros para a
classe **errada**. O scorer `average_precision` do scikit-learn usa `pos_label=1` — otimiza precisão média
para a classe *par legítimo*, enquanto todo o critério de decisão e o objetivo de controle são sobre a
classe *exceção* (rótulo 0).

Random Forest e Regressão Logística quase não se alteraram, porque suas precisões médias para as duas
classes se movem juntas ao longo de uma grade grosseira. O SVM, cujo único hiperparâmetro ajustado é `C`,
foi severamente afetado. Corrigido o scorer para `pos_label=0`, o recall de exceção do SVM saltou de
**0,088 para 0,312** — e só depois da correção descrita em §4.6 chegou a 0,800.

Vale notar que **os dois defeitos se acumulavam sobre o mesmo algoritmo**, cada um punindo o SVM por uma
razão diferente. Isso é instrutivo: encontrar e corrigir um deles não revelou o outro, porque o resultado
continuou plausível — um SVM em 0,312 ainda parecia "simplesmente pior", e teria sido reportado como tal.

### 4.8 O que este experimento permite afirmar, e o que não

| Pergunta | Experimento original | Experimento de comparação |
| --- | --- | --- |
| Qual algoritmo é melhor? | Não responde — o rótulo é função das features | Random Forest, mas por margem estreita sobre o SVM |
| A diferença é do algoritmo ou da representação? | Da representação (artefato da sentinela) | Do algoritmo — depois de dois artefatos de representação terem sido removidos deste experimento também |
| A tarefa é aprendível? | Não, removido o vazamento | Sim: 0,82 de recall sob piso de precisão 0,90 |
| Os algoritmos diferem muito entre si? | Pergunta inrespondível | Não: 5,2 p.p. separam o melhor do pior |
| Vale generalizar para dados reais? | Não | Não — a base continua sintética (§5.2, §5.4) |

---

## 5. Limitações do estudo

Recomenda-se apresentar as limitações em quatro blocos, na ordem abaixo. A coluna "gravidade" é sugestão
de priorização para a arguição: limitações **altas** devem ser assumidas espontaneamente no texto e na
apresentação oral, antes que a banca as levante.

### 5.1 Limitações de validade interna

| # | Limitação | Gravidade | Evidência no repositório |
| --- | --- | --- | --- |
| 1 | **Vazamento de rótulo.** `delta_days` e `delta_valor_pct` são simultaneamente a regra de rotulagem e variáveis preditoras, tornando o alvo função determinística das entradas. Quantificado em §3.4. **Não herdada pelo experimento de comparação** (§4.1), onde o rótulo vem da verdade de origem. | Alta | `labeler.py:56-70` vs. `features.py:22-23` |
| 2 | **Vazamento secundário.** `nfse_valor_iss` e `nfse_aliquota` codificam indiretamente a existência de correspondência. Isolado em `no_leak_strict`. **Não herdada** — no experimento de comparação toda nota candidata existe, e o guarda anti-vazamento verifica cada variável (§4.2). | Média | `features.py:25-26`; §3.4 |
| 3 | **Métricas saturadas.** Com acurácia unitária, os intervalos de confiança colapsam e a comparação entre algoritmos perde poder estatístico. | Alta | `metrics_summary.csv`, dp de CV = 0,0000 |
| 4 | ~~**Ausência de teste de significância.**~~ **Resolvida em §4.3:** Wilcoxon pareado sobre 10 sementes com correção de Holm, tamanho de efeito reportado ao lado do p. Permanece válida para o experimento original e para a ablação. | Resolvida | `comparison_stats.py`; `wilcoxon.csv` |
| 5 | **Validação cruzada não temporal.** Os dados possuem dimensão temporal (datas de pagamento e emissão), mas o `StratifiedKFold` embaralha os registros, permitindo treino com informação posterior ao teste. | Média | `trainer.py:80` |
| 6 | **Métrica de seleção inadequada ao domínio.** `f1_macro` trata falsos positivos e falsos negativos como equivalentes, contrariando a assimetria de custo da conciliação. **Corrigida no experimento de comparação** (§4.1), que seleciona por recall de exceção sob piso de precisão — mas ver §4.7: a orientação do scorer precisou ser verificada empiricamente. | Média | `config.yaml`, `scoring: f1_macro` |
| 7 | ~~**Semente única.**~~ **Resolvida em §4.3:** o experimento de comparação roda 10 bases independentes e reporta desvio-padrão entre sementes — que expôs, na primeira versão, a instabilidade do SVM depois rastreada até o artefato de escala de §4.6. Permanece válida para o experimento original e para a ablação. | Resolvida | `run_comparison.py`; `per_seed_metrics.csv` |
| 8 | **Importância por impureza.** Método enviesado para variáveis contínuas; não foram calculadas importâncias por permutação. | Baixa | `evaluator.py` |

### 5.2 Limitações de validade externa

| # | Limitação | Gravidade |
| --- | --- | --- |
| 9 | **Dados inteiramente sintéticos.** Nenhum registro contábil real foi utilizado; a distribuição conjunta das variáveis reflete as decisões do gerador, não a realidade empresarial. | Alta |
| 10 | **Correspondência 1:1 por construção.** CNPJs únicos por pagamento e uma NFS-e por pagamento eliminam o cenário N:M (um pagamento para várias notas, pagamentos parcelados), que é dominante na prática. | Alta |
| 11 | **Janelas de tolerância vazias.** O gerador produz divergências de 6-30 dias e 3-20%, fora das tolerâncias de ±5 dias e ±2%; o cenário `fuzzy` não exercitou a tolerância que se propunha a testar. | Alta |
| 12 | ~~**Ausência de ruído textual realista.**~~ **Resolvida no experimento de comparação** (§4.1), onde a descrição do pagamento é derivada da discriminação da nota por abreviação, transposição de caracteres e perda de acentuação, e `similaridade_descricao` passa a carregar sinal genuíno (0,785 de acurácia isolada, §4.2). Permanece válida para o experimento original. Texto original: Razões sociais, descrições e discriminações são geradas independentemente; não há abreviações, erros de digitação, grafias alternativas ou variação de nomenclatura — exatamente o ruído que motiva o uso de similaridade textual. | Resolvida |
| 13 | **Ausência de sinal aprendível residual.** Removido o vazamento, a base não contém informação suficiente para a tarefa (§3.4), o que impede generalizar qualquer conclusão sobre viabilidade de aprendizado supervisionado em conciliação. | Alta |
| 14 | **Taxa de conciliação fixa em 70%.** Parâmetro arbitrário, não calibrado por evidência empírica sobre taxas reais de conciliação em contas a pagar. | Média |
| 15 | **Escopo restrito a NFS-e.** Apenas notas de serviço no padrão ABRASF; NF-e de mercadorias, notas de importação e documentos não fiscais ficaram fora. | Média |
| 16 | **Ausência de deriva temporal.** A base cobre um ano sem mudanças de regime (alteração de fornecedores, política de pagamento, sazonalidade), impedindo avaliar degradação do modelo ao longo do tempo. | Média |
| 17 | **Volume único.** Um só tamanho de base (7.500 registros); não há curva de aprendizado que informe o volume mínimo necessário. | Baixa |

### 5.3 Limitações de validade de construto

| # | Limitação | Gravidade |
| --- | --- | --- |
| 18 | **A conciliação foi operacionalizada como regra de limiar.** Na prática contábil, conciliar envolve julgamento profissional, conhecimento contratual e contexto de negócio que não se reduzem a tolerâncias sobre data e valor. O construto medido é mais estreito do que o construto nomeado. | Alta |
| 19 | **Ausência de padrão-ouro independente.** Não houve rotulagem por especialista humano; o rótulo é produto do próprio sistema, o que impede medir concordância com o julgamento contábil (kappa de Cohen, por exemplo). | Alta |
| 20 | ~~**A classe "parcialmente conciliado" é convenção arbitrária.**~~ **Resolvida:** a classe foi removida no experimento de comparação, que é binário — par verdadeiro contra exceção — justamente por não haver base normativa para os multiplicadores 4× e 5×. Permanece válida para o cenário `fuzzy` original. | Resolvida |

### 5.4 Limitações específicas do experimento de comparação

| # | Limitação | Gravidade |
| --- | --- | --- |
| 24 | **A base continua sintética.** Todas as limitações de validade externa 9, 10, 11, 14, 15, 16 e 17 permanecem integralmente. O experimento demonstra que a tarefa *construída* é aprendível e que os algoritmos diferem *nela* — não que conciliação real seja aprendível. | Alta |
| 25 | **As alíquotas de retenção são parâmetros, não afirmações normativas.** IRRF 1,5%, CSRF 4,65%, INSS 11% e o limite de R$ 5.000 estão em `config.yaml` e precisam de conferência contra a legislação vigente antes de qualquer afirmação jurídica no texto. | Alta |
| 26 | **A dificuldade da tarefa é um parâmetro escolhido.** `hard_negative_rate: 0.30` determina quanto da base é resolvível apenas pela razão de valor. Um valor diferente moveria as três médias. O valor usado foi fixado antes de observar os resultados e não foi ajustado depois — mas isso é uma afirmação sobre o processo, não uma propriedade verificável do artefato. | Alta |
| 27 | **Resíduo do artefato de escala em `desvio_prazo_fornecedor`.** Com cinco pagamentos por fornecedor, uma linha de treino contribui para a mediana do próprio fornecedor e tem o desvio encolhido em relação a uma linha de teste: na configuração de 7.500 registros, desvio-padrão de cerca de 13,8 no treino contra 18,5 no teste, com as linhas extremas em |z| ≈ 5. É o mesmo mecanismo de §4.6 em amplitude muito menor, mas não é zero. Uma mediana *leave-one-out* o eliminaria. | Média |
| 28 | **A similaridade textual é acoplada à partição.** O `TfidfVectorizer` é reajustado a cada chamada de `build_features_v2`, de modo que o IDF de um registro depende de com quais outros ele foi processado — a mesma acoplagem transdutiva que o estudo critica, em escala menor. Como consequência, `similaridade_descricao` não é computável para um registro novo isolado em inferência. | Média |
| 29 | **O guarda anti-vazamento só enxerga um modo de falha.** Ele detecta variáveis que predizem o rótulo *bem demais*. Não detecta variáveis constantes no treino, nem incompatibilidade de escala entre partições — que foram exatamente os defeitos de §4.6. A suíte ganhou asserções para esses casos, mas a lista de verificações continua sendo enumerada por descoberta, não por construção. | Alta |

### 5.5 Limitações de implementação e documentação

| # | Limitação | Gravidade |
| --- | --- | --- |
| 21 | ~~**Divergência entre documentação e código.**~~ **Resolvida:** o `README.md` descrevia `is_mesmo_municipio`, que não existe em `features.py`; a tabela agora documenta `cnpj_match`, que é o que o código usa. | Resolvida |
| 22 | **Variável constante em produção.** `cnpj_match` assume valor 1 em 100% dos registros, sem contribuição informacional. | Baixa |
| 23 | **Codificação por sentinela.** `delta_days = 9999` e `delta_valor_pct = 100` para ausência de correspondência distorcem a padronização e prejudicam modelos sensíveis à escala — responsável integral pela diferença entre algoritmos (§3.3). | Alta |

> **Nota.** As limitações 4, 7, 12, 20 e 21 foram resolvidas ao longo deste trabalho, e cada uma indica
> o escopo de sua resolução. As demais permanecem, e as de gravidade **Alta** devem ser assumidas
> espontaneamente no texto e na apresentação oral.

---

## 6. Contribuições teóricas

**6.1 Evidência empírica quantificada de circularidade em rotulagem por regra.**
O estudo documenta, com pipeline reprodutível, artefatos versionados e experimento de ablação controlado,
um mecanismo de vazamento estruturalmente inevitável quando rótulos de conciliação são derivados de regras
determinísticas sobre variáveis que também compõem o vetor de atributos. A contribuição não está em
relatar o vazamento — o fenômeno é conhecido — mas em **medir sua magnitude e demonstrar que ele é o modo
de falha esperado, e não excepcional, da aplicação de aprendizado supervisionado à conciliação contábil**,
dado que a esmagadora maioria das organizações só dispõe de rótulos gerados por suas próprias regras de
negócio. A queda de 1,0000 para 0,7568 e, sem os proxies, para 0,4966 de F1-macro estabelece um
referencial numérico para o custo de ignorar essa circularidade.

**6.2 Demonstração experimental de que o ranking entre algoritmos é artefato da representação.**
A configuração `sentinel_fixed` mostra que os três classificadores atingem desempenho idêntico e perfeito
mediante uma única alteração de pré-processamento, sem mudança de algoritmo. Combinada à explicação
quantitativa da seção 2.2 — a faixa informativa de `delta_days` comprimida em 0,0085 desvios-padrão —, a
evidência sustenta o deslocamento da pergunta de pesquisa: **de "qual algoritmo é melhor para conciliação"
para "qual representação dos dados torna a conciliação aprendível"**. A inversão de ordem entre SVM e
Random Forest sob `no_leak` reforça que rankings obtidos sob representação inadequada não são estáveis.

**6.3 Caracterização da assimetria de custo como propriedade estrutural do domínio.**
O trabalho evidencia que classificadores otimizados para métricas simétricas erram sistematicamente na
direção mais nociva ao controle interno (130 falsos positivos contra zero falsos negativos na linha de
base; 250 contra zero sob `no_leak`; 50,4% dos parciais promovidos a conciliados). Isso fundamenta a
proposição de que **conciliação financeira pertence à classe de problemas em que a métrica de avaliação
deve ser derivada do objetivo de controle, não da convenção estatística** — articulando aprendizado
sensível a custo com estruturas normativas de controle interno, ponte pouco explorada na literatura de
contabilidade e sistemas de informação.

**6.4 Explicitação da lacuna entre conciliação e *record linkage*.**
Ao formalizar a conciliação como problema de pareamento de registros e evidenciar que a simplificação
1:1 elimina a geração de candidatos — núcleo do modelo de Fellegi-Sunter —, o trabalho delimita
precisamente o que separa o experimento acadêmico da aplicação organizacional, oferecendo agenda de
pesquisa em vez de conclusão prematura.

**6.5 Contribuição metodológica sobre validação em pipelines contábeis.**
O caso sustenta a proposição de que, em domínios de rótulo derivado de regra, **acurácia elevada deve ser
tratada como sinal de alarme e disparar auditoria da procedência do rótulo**, e não como critério de
aceitação. O experimento de ablação é apresentado como protocolo replicável para essa auditoria:
comparar o desempenho sob remoção progressiva das variáveis suspeitas de circularidade estabelece um
limite superior para o quanto o modelo realmente aprendeu.

---

## 7. Contribuições práticas

**7.1 Artefato de software reprodutível e auditável.**
O repositório entrega pipeline completo e determinístico — simulação, ETL, engenharia de atributos,
treinamento com busca em grade, avaliação e ablação — com parametrização integralmente externalizada em
`config.yaml`, semente fixa, empacotamento Python e suíte de testes automatizados cobrindo os quatro
módulos. Qualquer terceiro reproduz os resultados com dois comandos, e a reprodução dos valores originais
sob bibliotecas mais recentes (§3.2) constitui evidência da robustez do pipeline. Isso atende aos
critérios de pesquisa computacional reprodutível `[VERIFICAR: Peng, R. D. "Reproducible Research in
Computational Science". Science, v. 334, n. 6060, 2011]` e é, por si, contribuição de engenharia de
software — coerente com a natureza do MBA.

**7.2 Gerador de dados sintéticos de conciliação fiscal brasileira.**
Os módulos de simulação produzem planilhas de pagamento e XMLs de NFS-e aderentes ao padrão ABRASF, com
CNPJs válidos (dígitos verificadores calculados), códigos IBGE de municípios, itens da lista de serviços
da LC 116/2003 e quatro modos de divergência parametrizáveis. Trata-se de recurso reutilizável para
ensino, prototipagem e *benchmarking* em contexto fiscal brasileiro, onde a escassez de bases públicas é
obstáculo reconhecido à pesquisa — o que dialoga com a literatura de dados sintéticos
`[VERIFICAR: Patki, N.; Wedge, R.; Veeramachaneni, K. "The Synthetic Data Vault". In: IEEE International
Conference on Data Science and Advanced Analytics (DSAA), 2016]`.

**7.3 Checklist de diagnóstico para projetos de conciliação automatizada.**
Do estudo derivam verificações diretamente aplicáveis por equipes de engenharia e auditoria interna:

1. Confirmar que nenhuma variável preditora participa da regra que gerou o rótulo.
2. Tratar acurácia acima de ~0,98 em tarefa de conciliação como hipótese de vazamento até prova em
   contrário.
3. Executar ablação por remoção progressiva das variáveis suspeitas antes de aceitar qualquer resultado.
4. Substituir valores-sentinela por variável indicadora explícita de ausência (`has_match`) combinada com
   imputação neutra — alteração que, neste estudo, elevou a Regressão Logística de 0,8873 para 1,0000 de
   F1-macro e igualou os três algoritmos.
5. Selecionar modelos por recall da classe de exceção sob restrição de precisão mínima, não por acurácia
   ou F1-macro.
6. Reportar a matriz de confusão completa, e não apenas métricas agregadas — foi a matriz que revelou a
   assimetria 130/0, e foi a decomposição por tipo de divergência que revelou a cegueira dos modelos às
   exceções reais.
7. Validar temporalmente quando os dados possuem dimensão temporal.

**7.4 Evidência para decisão de investimento em automação.**
Os resultados sustentam recomendação concreta para organizações: **quando a regra de conciliação é
conhecida e determinística, aprendizado de máquina não agrega valor sobre a implementação direta da
regra** — a Random Forest apenas reproduziu, com custo computacional e opacidade adicionais, aquilo que
uma consulta SQL executa exatamente. A ablação torna o argumento ainda mais forte: removida a regra, o
que os modelos conseguem detectar é apenas a ausência de nota fiscal, que é o resultado de uma junção
relacional. O aprendizado supervisionado só se justifica onde a regra é desconhecida, instável ou
intratável: ruído textual em razões sociais, pareamento N:M, priorização de exceções por risco. Essa
delimitação evita investimento mal direcionado e reforça o argumento de que modelos interpretáveis devem
ser preferidos em decisões de alto risco `[VERIFICAR: Rudin, C. "Stop Explaining Black Box Machine
Learning Models for High Stakes Decisions and Use Interpretable Models Instead". Nature Machine
Intelligence, v. 1, 2019]`.

**7.5 O guarda anti-vazamento como artefato reutilizável.**
`models/leak_guard.py` ajusta uma árvore de profundidade 1 sobre cada variável isoladamente e **falha o
pipeline** — não emite aviso — se alguma ultrapassar um teto configurável. Converte a recomendação
"acurácia elevada deve ser tratada como sinal de alarme" em verificação executável, e roda antes de cada
treino. É diretamente transplantável para qualquer projeto que treine sobre rótulos gerados por regras
internas, que é a situação da maioria das organizações. O experimento de comparação o exercita em 10
execuções (§4.2).

A delimitação do artefato é parte da contribuição, e é honesta: **o guarda não capturou dois dos três
modos de falha que este trabalho documenta**. Ele detecta excesso de informação em uma variável, e foi
cego tanto à métrica de seleção apontada para a classe errada (§4.7) quanto à variável constante no
treino com escalas incompatíveis entre partições (§4.6). A suíte de testes ganhou asserções para o
segundo caso. O aprendizado que se transfere não é o instrumento, e sim que **cada instrumento só enxerga
o modo de falha para o qual foi construído** — o que torna a revisão adversarial do código insubstituível
por verificação automatizada.

**7.6 Arquitetura de referência em camadas.**
A separação entre simulação, ETL, modelagem e avaliação, com configuração externalizada e artefatos
persistidos por cenário, constitui modelo transponível para implantações reais, mitigando o débito
técnico característico de sistemas de aprendizado de máquina
`[VERIFICAR: Sculley, D. et al. "Hidden Technical Debt in Machine Learning Systems". In: Advances in
Neural Information Processing Systems (NeurIPS), 2015]`.

---

## 8. Agenda de pesquisa futura

Os itens 1, 2, 4, 6, 8 e 9 da agenda original **já foram executados** e integram as seções 3 e 4. Os
demais permanecem em aberto, em ordem de retorno esperado:

1. ~~Replicar sem as variáveis vazadas.~~ **Executado** (§3.4): F1-macro cai de 1,0000 para 0,7568 (RF,
   `exact`) e a ordem entre algoritmos se inverte.
2. ~~Substituir a sentinela por indicador explícito.~~ **Executado** (§3.3): a diferença entre Random
   Forest e modelos de fronteira suave desaparece integralmente.
3. **Calibrar o gerador para exercitar as tolerâncias.** Alterar as faixas de divergência para incluir o
   intervalo de 0 a 5 dias e de 0% a 2%, tornando o cenário `fuzzy` efetivamente informativo. Aplica-se ao
   experimento original; o de comparação abandonou as tolerâncias em favor da verdade de origem.
4. ~~Introduzir sinal genuíno na simulação.~~ **Executado** (§4.1): retenções tributárias legítimas,
   prazo característico por fornecedor e município por centro de custo. O guarda anti-vazamento confirma
   que nenhuma variável isolada separa as classes (§4.2).
5. **Estender ao pareamento N:M.** Modelar pares candidatos em vez de registros, incorporando geração de
   candidatos (*blocking*) e adotando o arcabouço de Fellegi-Sunter como linha de base.
6. ~~Introduzir ruído textual realista.~~ **Executado** (§4.1): abreviação, transposição de caracteres
   e perda de acentuação. `similaridade_descricao` passou a ser a terceira variável mais informativa.
7. **Validar com base real.** Convênio com organização para rotulagem por especialista e medição de
   concordância entre modelo e julgamento contábil.
8. ~~Adotar seleção sensível a custo.~~ **Parcialmente executado** (§4.1): o limiar é otimizado para
   recall de exceção sob piso de precisão, e as curvas precisão-recall são reportadas. Falta a matriz de
   custo monetário explícita, que exigiria justificar valores sem base empírica.
9. ~~Repetir sob múltiplas sementes e aplicar testes de significância.~~ **Executado** (§4.3): 10
   sementes, Wilcoxon pareado com correção de Holm e tamanho de efeito.
10. **Comparar com linha de base não supervisionada.** Detecção de anomalias (*Isolation Forest*,
    *autoencoders*) dispensa rótulos e, por isso, é imune ao vazamento aqui documentado.
11. **Varrer `hard_negative_rate`.** Mapear como a diferença entre algoritmos responde à dificuldade da
    tarefa, transformando a limitação 26 em resultado. Com apenas 5,2 pontos percentuais separando o
    melhor do pior algoritmo (§4.3), é plausível que a ordem se altere em outros regimes de dificuldade —
    o que seria um resultado mais informativo que o ranking atual.
12. **Adotar mediana *leave-one-out* para o prazo do fornecedor.** Elimina o resíduo de escala da
    limitação 27, e é o fecho natural do achado de §4.6.
13. **Ampliar o conjunto de verificações automatizadas de representação.** O guarda anti-vazamento cobre
    um modo de falha; a suíte ganhou asserções para variável constante e escala entre partições. Um
    catálogo sistemático — deriva de distribuição entre partições, colunas de variância quase nula,
    features com correlação perfeita entre si — transformaria a lição de §4.6 em ferramenta.

---

## 9. Como responder à banca

Antecipe as três perguntas prováveis. Assumir a fragilidade antes da arguição, com experimento que a
quantifica, converte vulnerabilidade em demonstração de rigor.

**"Por que a Random Forest acertou 100%?"**
> Porque o rótulo é uma função determinística de duas variáveis que também são entradas do modelo. Isso
> configura vazamento de rótulo. Identifiquei o problema durante a análise e o submeti a teste: removendo
> essas duas variáveis, o F1-macro cai de 1,0000 para 0,7568; removendo também as variáveis de ISS, que
> funcionam como proxy da existência da nota, cai para 0,4966 — praticamente o acaso. Optei por manter o
> desenho original e documentar o achado, porque ele representa exatamente a armadilha em que
> organizações incorrem ao treinar modelos sobre rótulos gerados por suas próprias regras de negócio.

**"Então o trabalho não demonstra qual algoritmo é melhor?"**
> Não demonstra, e essa é a conclusão. Demonstrei experimentalmente que a diferença de 9 pontos
> percentuais entre a Random Forest e os demais era artefato de pré-processamento: o valor-sentinela 9999,
> usado para marcar pagamentos sem nota, domina a padronização e comprime toda a faixa de 0 a 30 dias de
> defasagem em 0,0085 desvios-padrão. Substituindo a sentinela por uma variável indicadora, os três
> algoritmos passam a acertar 100% — inclusive a Regressão Logística. E quando removo o vazamento, a
> ordem se inverte: o SVM supera a Random Forest. Rankings obtidos sob representação inadequada não são
> estáveis.

**"E afinal, qual algoritmo é melhor?"**
> A Random Forest — mas a margem é estreita e a história de como cheguei a ela é mais importante que o
> ranking. Em 10 execuções independentes, com o rótulo vindo da verdade de origem do gerador e seleção
> por recall de exceção sob piso de precisão de 0,90, a Random Forest detecta 1,44 pontos percentuais a
> mais que o SVM e 4,56 a mais que a Regressão Logística, com p = 0,0059 após correção de Holm. A
> significância vem da ordem se repetir nas dez execuções, não da magnitude: 5,2 pontos separam o melhor
> do pior, e os três são operacionalmente próximos. Minha leitura é que a escolha pode legitimamente
> recair sobre interpretabilidade em vez de desempenho.

**"Como você sabe que esse resultado está certo, se os anteriores estavam errados?"**
> Não sei que está certo — sei que sobreviveu a três auditorias que derrubaram versões anteriores. A
> primeira resposta deste mesmo experimento dizia que o SVM ficava 51,9 pontos atrás; era artefato de uma
> variável constante na partição de treino, que fazia o `StandardScaler` passar valores de teste sem
> normalizar e colapsar o núcleo RBF. Corrigido, o SVM foi de 0,221 para 0,800. Antes disso, o scorer do
> `GridSearchCV` otimizava a classe errada, o que já havia punido o mesmo algoritmo. Documentei os três
> defeitos na §4.6 porque eles são o achado mais transferível do trabalho: nenhum se manifestou como
> erro, os três produziram tabelas plausíveis, e dois deles passaram pelo guarda anti-vazamento que eu
> mesmo construí para detectar esse tipo de problema. A conclusão metodológica é que cada instrumento de
> verificação só enxerga o modo de falha para o qual foi construído.

**"Qual é a utilidade prática, então?"**
> Três resultados acionáveis. Primeiro: quando a regra de conciliação é conhecida, aprendizado de máquina
> não agrega sobre a implementação direta da regra — sem o vazamento, o que os modelos detectam é apenas
> a ausência de nota fiscal, que é uma junção SQL. Segundo: a codificação de ausência por
> valores-sentinela inviabiliza modelos sensíveis à escala, e a correção é trivial e mensurável. Terceiro,
> e mais importante para auditoria: os modelos erraram 130 vezes por falso positivo e nenhuma por falso
> negativo — ou seja, na direção que silencia divergências. Isso indica que a métrica de seleção em
> conciliação deve ser o recall da classe de exceção, não a acurácia.

---

## 10. Referências sugeridas — conferir antes de citar

Todas as obras abaixo são reais e reconhecidas em suas áreas. **Confirme os dados bibliográficos completos
em base indexada e formate conforme ABNT NBR 6023 antes de incluí-las.** Não cite o que não tiver
consultado.

**Vazamento, validação e metodologia de aprendizado de máquina**
- Kaufman, S.; Rosset, S.; Perlich, C.; Stitelman, O. *Leakage in Data Mining: Formulation, Detection, and Avoidance*. ACM TKDD, 2012.
- Domingos, P. *A Few Useful Things to Know about Machine Learning*. Communications of the ACM, 2012.
- Wolpert, D. H. *The Lack of A Priori Distinctions Between Learning Algorithms*. Neural Computation, 1996.
- Fernández-Delgado, M. et al. *Do we Need Hundreds of Classifiers to Solve Real World Classification Problems?* JMLR, 2014.
- Demšar, J. *Statistical Comparisons of Classifiers over Multiple Data Sets*. JMLR, 2006.
- Bergmeir, C.; Benítez, J. M. *On the Use of Cross-Validation for Time Series Predictor Evaluation*. Information Sciences, 2012.
- Saito, T.; Rehmsmeier, M. *The Precision-Recall Plot Is More Informative than the ROC Plot...* PLoS ONE, 2015.
- Elkan, C. *The Foundations of Cost-Sensitive Learning*. IJCAI, 2001.

**Algoritmos**
- Breiman, L. *Random Forests*. Machine Learning, 2001.
- Cortes, C.; Vapnik, V. *Support-Vector Networks*. Machine Learning, 1995.
- Strobl, C. et al. *Bias in Random Forest Variable Importance Measures*. BMC Bioinformatics, 2007.
- Chawla, N. V. et al. *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR, 2002.

**Pareamento de registros**
- Fellegi, I. P.; Sunter, A. B. *A Theory for Record Linkage*. JASA, 1969.
- Christen, P. *Data Matching*. Springer, 2012.
- Winkler, W. E. *Overview of Record Linkage and Current Research Directions*. US Census Bureau, 2006.

**Aprendizado de máquina em contabilidade e auditoria**
- Vasarhelyi, M. A.; Kogan, A.; Tuttle, B. M. *Big Data in Accounting: An Overview*. Accounting Horizons, 2015.
- Appelbaum, D.; Kogan, A.; Vasarhelyi, M. A. *Big Data and Analytics in the Modern Audit Engagement*. Auditing: A Journal of Practice & Theory, 2017.
- Issa, H.; Sun, T.; Vasarhelyi, M. A. *Research Ideas for Artificial Intelligence in Auditing*. Journal of Emerging Technologies in Accounting, 2016.
- Sutton, S. G.; Holt, M.; Arnold, V. *"The Reports of My Death Are Greatly Exaggerated" — Artificial Intelligence Research in Accounting*. International Journal of Accounting Information Systems, 2016.
- Bao, Y. et al. *Detecting Accounting Fraud in Publicly Traded U.S. Firms Using a Machine Learning Approach*. Journal of Accounting Research, 2020.

**Engenharia de software, reprodutibilidade e interpretabilidade**
- Sculley, D. et al. *Hidden Technical Debt in Machine Learning Systems*. NeurIPS, 2015.
- Peng, R. D. *Reproducible Research in Computational Science*. Science, 2011.
- Rudin, C. *Stop Explaining Black Box Machine Learning Models for High Stakes Decisions...* Nature Machine Intelligence, 2019.
- Patki, N.; Wedge, R.; Veeramachaneni, K. *The Synthetic Data Vault*. IEEE DSAA, 2016.

**Normas e estruturas de controle**
- COSO. *Internal Control — Integrated Framework*, 2013.
- Conselho Federal de Contabilidade. *NBC TA 530 — Amostragem em Auditoria*.
- Conselho Federal de Contabilidade. *NBC TA 500 — Evidência de Auditoria*.
- ABRASF. *Modelo Conceitual e de Especificação de Requisitos — Nota Fiscal de Serviços Eletrônica*.
- Brasil. *Lei Complementar nº 116, de 31 de julho de 2003*.

---

## Anexo A — Reprodução do experimento de ablação

```bash
python run_ablation.py
```

Saídas em `data/results/ablation/`:

```text
ablation_summary.csv                  # 24 execuções (4 configurações × 2 cenários × 3 algoritmos)
<config>/<scenario>/cm_<algoritmo>.txt   # matrizes de confusão
<config>/<scenario>/models/*.joblib      # modelos treinados
```

Os artefatos do experimento principal em `data/results/exact/` e `data/results/fuzzy/` não são
modificados.

---

## Anexo B — Reprodução do experimento de comparação

```bash
python run_comparison.py --seeds 2    # ensaio rápido
python run_comparison.py              # completo: 10 sementes
```

Saídas em `data/results/comparison/`:

```text
per_seed_metrics.csv     # 30 linhas: 10 sementes × 3 algoritmos
wilcoxon.csv             # 3 comparações: p bruto, p corrigido (Holm), tamanho de efeito
decision_summary.csv     # tabela final, com n_seeds_applicable
leak_guard_report.csv    # acurácia do toco por variável, por semente
pr_curves.png            # curvas precisão-recall da classe de exceção (primeira semente)
recall_boxplot.png       # dispersão do recall de exceção entre as 10 sementes
```

O guarda anti-vazamento roda antes de cada treino e **interrompe a execução** se qualquer variável
isolada ultrapassar `leak_guard_max_stump_accuracy` (0,95). Se isso ocorrer, a correção é aumentar a
sobreposição entre classes no gerador — nunca remover a variável acusada, que foi exatamente o que a
configuração `no_leak` da ablação mostrou ser destrutivo.

Os artefatos do experimento principal e da ablação não são modificados.
