# Roteiro da apresentação

## 1 - Definição do problema

Na maior parte da minha carreira como desenvolvedor de software, trabalhei em consultorias. Por isso, tive oportunidade de trabalhar em três das maiores instituições financeiras do país. Em duas delas, encontrei um problema semelhante: a conciliação contábil.

## 2 - O que é conciliação contábil?

A conciliação contábil é o processo de comparar e verificar se os saldos registrados na contabilidade de uma empresa estão de acordo com os extratos, documentos e registros externos (como bancos, fornecedores e clientes).  
Em termos simples: é conferir se o que está no livro contábil bate com a realidade financeira.  
Ela serve para:

- Identificar erros ou inconsistências.
- Garantir que os registros estejam corretos e confiáveis.
- Apoiar a tomada de decisão com informações precisas.
- Permitir que sejam realizadas auditorias.

## 3 - Como esse problema está sendo resolvido hoje?

### 3.1 - Caso 1

No primeiro caso que encontrei, era necessário conciliar mensagens do sistema Swift com operações registradas no sistema da instituição. Porém a chave usada para conciliar não era padronizada, pois vinha de diversas instituições diferentes ao redor do mundo, e cada uma enviava à sua maneira. O desafio, então, consistia em identificar essas chaves e compará-las com os registros internos. As chaves consistiam de textos representando o nome do destinatário da remessa, e foi desenvolvido um sistema que comparava o texto com os registros internos usando um algoritmo de similaridade, que gerava um percentual. Se o percentual atingisse um valor considerado suficiente, o registro era conciliado.

### 3.2 - Caso 2

No segundo caso, diversos registros eram periodicamente enviados de instituições que atuam no mercado financeiro, como B3 e CETIP. Esses registros vinham em diversos formatos: tabelas de Excel, arquivos de texto, arquivos CSV, PDFs, etc. Foi criado então um sistema ETL, que padronizava todos esses arquivos em CSV, processava os dados, executava a conciliação e devolvia um resultado para cada arquivo. Esse sistema dependia que o layout dos arquivos de entrada fosse sempre no mesmo padrão, o que nem sempre acontecia, e falhas eram frequentes. Trabalhei num projeto que visava modernizar esse sistema, usando metadados para padronizar os dados e executar as conciliações. No fim, o sistema atual era sujeito a falhas e a solução nova tinha um custo para funcionar em cloud e alta complexidade.

Nos dois casos, a lógica de conciliação era uma regra escrita à mão: um limiar de similaridade no primeiro, um layout fixo no segundo. Quando a realidade fugia da regra, o sistema falhava.

## 4 - O que este projeto visa?

A intenção deste projeto é trazer uma nova abordagem para resolução deste problema. Esta abordagem é através da aplicação de uma pipeline ETL aliada a um algoritmo de Machine Learning. O projeto visa não só demonstrar que a resolução deste problema é possível com essa abordagem, como também definir, dentre 3 diferentes algoritmos (Random Forest, SVM ou Regressão Logística), qual deles é mais eficiente para chegar a esta resolução.

A pergunta de pesquisa ficou assim: qual algoritmo melhor automatiza a conciliação entre pagamentos e notas fiscais de serviço, e quanto do desempenho observado vem do algoritmo e quanto vem da forma como os dados são representados?

E a hipótese era a de que o Machine Learning, junto com um processo de ETL, não apenas automatizaria a conciliação, como também melhoraria a sua qualidade, encontrando inconsistências que escapariam de uma abordagem baseada em regras.

## 5 - Como o projeto foi construído

Como não existe uma base contábil real e pública com rótulos auditados, os dados foram simulados. Simulei duas fontes diferentes, como acontece na prática:

- Uma planilha Excel com 7.500 pagamentos a fornecedores.
- As notas fiscais de serviço eletrônicas correspondentes (NFS-e), em XML no padrão ABRASF, com CNPJs válidos, códigos de município do IBGE e cálculo de ISS.

Cerca de 70% dos pagamentos tinham uma nota que batia, e 30% tinham algum problema: CNPJ errado, nota inexistente, divergência de data ou divergência de valor.

Esses dados passaram por uma pipeline ETL: carga das duas fontes, limpeza e padronização de CNPJs, datas e valores, e rotulagem de cada registro como conciliado ou não conciliado. A partir daí, foram criados os atributos que o modelo usa, como a diferença de dias entre pagamento e nota e a diferença percentual de valor.

Os três algoritmos foram treinados com busca de hiperparâmetros e validação cruzada de 5 partições, usando a biblioteca scikit-learn. Todo o projeto é reproduzível: os parâmetros ficam num arquivo de configuração e a semente aleatória é fixa.

## 6 - Primeiro experimento: um resultado bom demais

O primeiro resultado parecia excelente. A Random Forest acertou 100% dos casos, em todas as métricas, nos dois cenários testados. SVM e Regressão Logística ficaram em torno de 89% de F1-macro.

A leitura óbvia seria: "a Random Forest é o melhor algoritmo para conciliação". Mas um acerto de 100%, com desvio-padrão zero na validação cruzada, não é normal. Isso é um sinal de alarme, não um sucesso.

Investigando o código, encontrei a causa. O rótulo era definido por uma regra: o registro é conciliado se a diferença de dias e a diferença de valor estão dentro da tolerância. E essas mesmas duas grandezas eram entregues ao modelo como atributos. Ou seja, o modelo não estava aprendendo a conciliar, estava apenas redescobrindo a regra que gerou o rótulo. Isso tem nome na literatura: vazamento de rótulo (*label leakage*). Não por acaso, essas duas variáveis concentravam cerca de 87% da importância do modelo.

## 7 - Segundo experimento: o que a ablação revelou

Para medir esse problema, fiz um experimento de ablação: mantive os dados, a divisão e os hiperparâmetros, e mudei apenas a representação das variáveis.

Os resultados foram dois:

- **Removendo as variáveis da regra**, o F1 da Random Forest caiu de 1,00 para 0,76, e a ordem dos algoritmos se inverteu: o SVM passou à frente. Removendo também as variáveis de ISS, que só indicavam se a nota existia, o desempenho caiu para perto do acaso. Sem o vazamento, os modelos só conseguiam detectar pagamentos sem nota fiscal, o que uma consulta SQL de uma linha já faz.
- **A diferença entre os algoritmos era um artefato.** Pagamentos sem nota recebiam um valor-sentinela de 9999 dias. Na padronização de escala, esse valor esmagava toda a faixa útil de 0 a 30 dias em menos de 0,01 desvio-padrão. Para o SVM e a Regressão Logística, pagar no dia certo ou 30 dias depois ficou indistinguível. Troquei a sentinela por um indicador explícito de "tem nota ou não", e os três algoritmos chegaram a 100%. A vantagem de 11 pontos da Random Forest desapareceu com uma única mudança de pré-processamento.

## 8 - O erro que importa

Esse experimento revelou também algo sobre a natureza dos erros. SVM e Regressão Logística cometeram 128 falsos positivos e nenhum falso negativo, e todos os 128 eram divergências de data.

Em conciliação, esses dois erros não têm o mesmo peso:

- **Falso negativo**: um pagamento correto vai para revisão manual. Gera trabalho, mas o controle continua íntegro.
- **Falso positivo**: uma divergência é aprovada automaticamente e ninguém nunca mais olha para ela. A divergência é silenciada.

A própria norma brasileira de auditoria, a NBC TA 530, faz essa distinção: um erro afeta a eficácia da auditoria, o outro só a eficiência. Por isso, a métrica certa para escolher um modelo de conciliação não é acurácia nem F1, mas o recall da classe de exceção, respeitando uma precisão mínima.

## 9 - Terceiro experimento: uma comparação justa

Com esses aprendizados, reconstruí o experimento para responder de fato à pergunta de pesquisa. Três mudanças:

1. **O rótulo vem do gerador, não de uma regra sobre os atributos.** O simulador sabe qual pagamento quita qual nota, e o rótulo diz apenas se aquele par é verdadeiro.
2. **Pares verdadeiros deixam de ser cópias.** O valor pago é derivado da nota aplicando retenções tributárias reais (IRRF, CSRF, ISS, INSS), e cada fornecedor tem um prazo de pagamento característico. Assim, uma diferença de valor pode ser legítima, e o modelo precisa aprender a distinguir.
3. **O modelo é escolhido pelo recall de exceções, com precisão mínima de 90%.**

Além disso, criei um guarda anti-vazamento: antes de treinar, ele testa cada variável sozinha e interrompe a execução se alguma separar as classes com mais de 95% de acurácia. Nesse experimento, a melhor variável isolada chegou a 82%, então a tarefa era genuína. O experimento foi repetido em 10 bases independentes, com testes estatísticos.

## 10 - Resultados da comparação

Agora sim, com uma comparação válida, a Random Forest venceu: detectou 82,2% das exceções, contra 80,0% do SVM e 77,0% da Regressão Logística, encaminhando cerca de 27% dos registros para revisão manual.

A diferença é estatisticamente significativa: o teste de Friedman rejeitou a equivalência entre os três, e as comparações par a par confirmaram a ordem nas 10 execuções. E a vantagem não foi obtida às custas de precisão: sobre a precisão, os três são estatisticamente equivalentes.

Traduzindo em unidades de controle interno: em 15.000 pagamentos, trocar a Random Forest pela Regressão Logística deixa passar 233 divergências a mais. Mas a margem é estreita: são pouco mais de 5 pontos entre o melhor e o pior. Isso significa que critérios como interpretabilidade e custo também podem legitimamente pesar na escolha.

## 11 - Respondendo à pergunta e à hipótese

**Qual algoritmo?** Com rótulo independente e o critério certo, a Random Forest, por margem estreita.

**Algoritmo ou representação?** A representação dos dados e a origem do rótulo pesaram mais do que o algoritmo. No primeiro experimento, toda a diferença entre eles vinha da forma como a ausência de nota era codificada.

**A hipótese se confirmou?** Em parte. Onde a regra de conciliação é conhecida, o Machine Learning não melhorou nada: apenas reproduziu a regra, com mais custo e menos transparência. Onde a evidência é ambígua, os modelos automatizaram a triagem de exceções com precisão controlada. Mas, como não comparei diretamente com uma solução baseada em regras nesse cenário, a superioridade sobre ela ainda não está demonstrada.

## 12 - O que se leva para a prática

Voltando às instituições financeiras do início, estas são as lições que eu levaria para um projeto real:

- Se a regra de conciliação é conhecida e determinística, implemente a regra. ML só vale a pena onde a regra é desconhecida, instável ou difícil de escrever, como textos livres (o caso do Swift), pareamentos de muitos-para-muitos e priorização de exceções por risco.
- Acurácia muito alta em conciliação deve disparar uma auditoria do rótulo, não uma comemoração.
- Nenhuma variável usada para gerar o rótulo pode ser usada como atributo.
- Use um indicador explícito para "não encontrado", e não valores-sentinela.
- Escolha o modelo pelo recall de exceções sob uma precisão mínima.
- Olhe a matriz de confusão, não só as métricas agregadas.

O repositório também entrega um gerador de dados sintéticos de conciliação fiscal brasileira e o guarda anti-vazamento, que podem ser reutilizados em outros projetos.

## 13 - Limitações e próximos passos

É importante reconhecer os limites do trabalho:

- Os dados são simulados. O próximo passo natural é validar com uma base real, rotulada por especialistas.
- O pareamento é um-para-um. Na prática, um pagamento pode quitar várias notas e uma nota pode ser paga em parcelas.
- Faltou comparar com uma linha de base por regras e com métodos não supervisionados, como detecção de anomalias, que não dependem de rótulo.

Também aprendi que cada ferramenta de verificação só enxerga o tipo de falha para o qual foi construída. O guarda anti-vazamento, por exemplo, não detectou outros dois defeitos que encontrei ao longo do trabalho. A revisão crítica do código continua insubstituível.

## 14 - Conclusão

Comecei este trabalho querendo descobrir qual algoritmo resolve a conciliação contábil. Termino com uma resposta, a Random Forest, mas com uma lição mais importante: na conciliação automatizada, a forma como representamos os dados e a origem do rótulo pesam mais do que a escolha do algoritmo.

Agradeço a atenção.
