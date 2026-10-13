# Data representation determines algorithm performance in automated accounting reconciliation

> English translation of the final paper for the MBA in Software Engineering (USP/ESALQ), written in
> Brazilian Portuguese. The original is
> [`TCC - Artur Rios da Silva Oliveira.docx`](TCC%20-%20Artur%20Rios%20da%20Silva%20Oliveira.docx)
> ([PDF](TCC%20-%20Artur%20Rios%20da%20Silva%20Oliveira.pdf)), and the Markdown version of the
> original is [`monografia.md`](monografia.md); where they differ, the original prevails. The abstract,
> title and keywords are the authors' own English text; the figures keep their Portuguese labels.
> Technical terms are defined in the [glossary](../glossary.md).

Artur Rios da Silva Oliveira<sup>1\*</sup>; Emerson Aparecido Mouco Júnior<sup>2</sup>

<sup>1\*</sup> Bachelor of Information Systems. UMC. Corresponding author e-mail: arturdev@duck.com

<sup>2</sup> Master of Business Administration. USP. E-mail: mouco@alumni.usp.br

## Abstract

Accounting reconciliation is time consuming and remains exposed to human error in organisations that still perform it manually, a condition that motivated interest in automation supported by supervised learning. This work investigated which algorithm best automates the reconciliation between payment records and electronic service invoices, and to what extent the observed performance follows from the algorithm or from the representation assigned to the data. Two heterogeneous sources were simulated, a supplier payment spreadsheet and a set of tax documents in XML format, processed through extraction, transformation and loading and used to train a random forest, a support vector machine and a logistic regression. The first experiment produced unit accuracy for the random forest, a performance that the code audit attributed to circularity between the labelling rule and the predictive attributes. An ablation experiment quantified that circularity and showed that the eleven percentage point macro-F1 advantage of the forest over the remaining algorithms followed entirely from sentinel value encoding. A second experiment, with the label taken from generator ground truth, an automated leakage guard and model selection by exception recall under a minimum precision of 0.90, repositioned the comparison: across ten independent runs the random forest detected 1.44 percentage points more divergences than the support vector machine and 4.56 percentage points more than the logistic regression, at the cost of referring 27.6% of the batch to manual review. Data representation and label provenance weighed more on automated accounting reconciliation than the choice of algorithm.

**Keywords:** supervised learning; label leakage; random forest; internal control; electronic invoice.

## Introduction

Accounting reconciliation consists of checking and matching financial transaction data produced by different systems or subsystems, from standardised source documents, in order to identify and correct errors, detect fraud and ensure that closing balances reflect the organisation's economic reality (Mahmud and Fitriani, 2025). In a substantial share of companies the activity remains manual or semi-automated, a condition that lengthens execution time, increases exposure to human error and limits how far the control can scale as the volume of processed documents grows.

Automation services have become a trend across many sectors, driven by recent advances in artificial intelligence and machine learning (Soni et al., 2020). Machine learning brings together algorithms that exploit correlations within a dataset, discover patterns and produce models able to map inputs to outputs and to generalise that behaviour to new examples (França et al., 2021). Applied to reconciliation, the supervised paradigm turns the decision to reconcile into a classification problem over pairs of records, in which the classifier receives observable evidence about the pairing and returns the corresponding category.

The heterogeneity of the sources is an obstacle that precedes modelling. Extraction, transformation and loading [ETL] processes collect data from multiple origins with different schemas, transform them into a usable representation and perform the cleaning needed to ensure the validity of the information used in training (Cormier et al., 2025). Reconciliation keys such as the supplier's registration number, the issue and payment dates and the nominal amounts require prior normalisation for the comparison between sources to be possible.

Formally, reconciling payment records against tax documents is a record linkage problem, whose probabilistic foundation goes back to the model of Fellegi and Sunter (1969) and was systematised by Christen (2012). In that framework the object of decision is the candidate pair, and the computational effort concentrates on candidate generation, since the space of pairs grows quadratically with the number of records in each source.

The use of supervised learning in this domain faces a specific difficulty. Most organisations only have labels produced by their own business rules, and when the quantities that make up the labelling rule are also part of the attribute vector, the target becomes a deterministic function of the inputs: the model recovers the rule instead of learning the task. Kaufman et al. (2012) described this introduction, among the predictors, of information derived from the target itself, and identified implausibly high and stable performance as its characteristic symptom. The literature on machine learning applied to accounting and auditing records a recurring gap between reported accuracy and the operational usefulness of the models (Appelbaum et al., 2017; Bao et al., 2020).

In this work the algorithms were used for supervised classification, on simulated data of supplier payments and electronic service invoices generated by explicit statistical procedures. Three classifiers from different families were compared: the random forest, which recursively partitions the attribute space (Breiman, 2001); the support vector machine, which seeks the maximum-margin boundary in a kernel-transformed space (Cortes and Vapnik, 1995); and logistic regression, a linear boundary whose coefficients are directly interpretable. The method carries a limitation known in advance, which the design itself sought to expose rather than hide: labels derived from an internal rule restrict what can be learned and what can be concluded about the algorithms.

This work aimed to determine which of these three algorithms best automates the reconciliation between payment records and electronic service invoices, and to establish to what extent the observed performance follows from the chosen algorithm or from the representation assigned to the data. The hypothesis guiding the research was that applying machine learning algorithms, together with an extraction, transformation and loading process, would not only automate reconciliation but also improve its quality, identifying inconsistencies that rule-based approaches would miss.

## Implementation of Machine Learning Algorithm(s)

### Research design

The research was an applied study with a quantitative approach, conducted under the design of implementing machine learning algorithms. Development took place in three chained stages, each motivated by the result of the previous one. The first implemented the complete reconciliation pipeline, from source simulation to model evaluation, in two label-strictness scenarios. The second tested, through controlled ablation, two hypotheses raised in the audit of the first stage's results. The third rebuilt the experiment under labelling independent of the attributes, in order to answer the research question with discriminating power.

All code was written in Python and organised in the reconciliacao package, with simulation, ETL, modelling and evaluation kept separate. Every experimental parameter was externalised to a configuration file, with no hard-coded values in the code modules, and the seed of the pseudorandom generator was held constant, so that each run reproduces exactly the same artefacts. The models and metrics were obtained with the scikit-learn library (Pedregosa et al., 2011).

### Generation of the simulated data

In the absence of a real accounting dataset with audited labels, the data were generated by simulation, a procedure accepted for papers on algorithm implementation provided the generation criteria are described in detail. The generator produced two sources that are heterogeneous and independent in format, replicating the situation in which reconciliation presents itself in practice: a spreadsheet recording supplier payments and a set of electronic service invoices in XML format, structured according to the interchange standard adopted by Brazilian municipalities.

Supplier registration numbers were generated with computed, not random, check digits, so that the registration validation applied in the cleaning stage had a real referent. Each invoice received a municipality code from the official code table, an item from the service list, a service amount, a rate and the amount of the service tax [ISS] computed from those two fields. The description of each payment and the service description of each invoice were produced by independent calls to the synthetic text generator. The simulation parameters are listed in Table 1.

**Table 1.** Simulation parameters of the main experiment

| Parameter | Value |
| :--- | ---: |
| Payment records generated | 7,500 |
| Reconciliation rate | 70% |
| Pseudorandom generator seed | 42 |
| Date divergence introduced | 6 to 30 days |
| Amount divergence introduced | 3% to 20% |
| Test partition, stratified | 20% |
| Folds of the stratified cross-validation | 5 |

Source: Original research data

The non-reconciled fraction of the dataset, 30% of the records, was spread across three divergence modes. The first removed the match, either by assigning a non-existent registration number to the payment or by issuing invoices with no associated payment. The second shifted the payment date by an interval drawn between six and thirty days. The third altered the amount paid by a factor drawn between 3% and 20% of the service amount.

### Extraction, transformation and loading

The extraction module read the payment spreadsheet and walked through the XML documents, converting both into tabular structures. The cleaning module standardised registration numbers by removing the mask and validating the check digits, converted dates to a single temporal type and normalised monetary amounts. The sources were joined on the supplier's registration number, producing 7,518 records from the 7,500 payments requested. The difference did not come from duplicates resolved in the join: eighteen payments had their registration number repeated in two invoices, and the join multiplied them into two rows, twelve of them with mutually contradictory labels, for a reason discussed later among the silent failure modes.

The labelling module assigned each record's class by a deterministic threshold rule over two quantities computed in the join: the absolute difference in days between the payment date and the issue date, and the relative difference between the amount paid and the service amount. Two strictness scenarios were defined, described in Table 2. In the exact scenario, reconciliation required a perfect match of date and amount, with two classes. In the fuzzy scenario, tolerances were allowed and an intermediate class was added, assigned to records within four times the date tolerance or five times the amount tolerance. Figure 1 reproduces the part of the labeller that implements the rule.

**Table 2.** Label strictness scenarios

| Scenario | Date tolerance | Amount tolerance | Classes |
| :--- | ---: | ---: | ---: |
| exact | 0 days | 0% | 2 |
| fuzzy | 5 days | 2% | 3 |

Source: Original research data

```python
# Absolute date and amount divergence between the payment and the invoice
merged["delta_days"] = (merged["data_pagamento"]
                        - merged["nfse_data_emissao"]).dt.days.abs()
                        .fillna(9999)  # sentinel value
merged["delta_valor_pct"] = ((merged["valor_pago"]
                              - merged["nfse_valor_servicos"]).abs()
                             / safe_valor * 100).fillna(100.0)

# Deterministic threshold rule that defines the label
mask_date_ok = merged["delta_days"] <= date_tolerance_days
mask_valor_ok = merged["delta_valor_pct"] <= value_tolerance_pct
if n_classes == 2:  # exact scenario
    merged.loc[mask_date_ok & mask_valor_ok, "label"] = 1
else:  # fuzzy scenario
    mask_date_close = merged["delta_days"] <= date_tolerance_days * 4
    mask_valor_close = merged["delta_valor_pct"] <= value_tolerance_pct * 5
    mask_parcial = (~(mask_date_ok & mask_valor_ok)) & (mask_date_close
                                                        | mask_valor_close)
    merged.loc[mask_parcial, "label"] = 1
    merged.loc[mask_date_ok & mask_valor_ok, "label"] = 2
```

**Figure 1.** Labelling rule of the main experiment, applied to the date difference and the relative amount difference

Source: Original research data

### Feature engineering and algorithms evaluated

The attribute matrix of the main experiment comprised seven variables, described in Table 3. Payment records without a matching invoice have no computable date or amount difference, and received sentinel values: 9,999 for the difference in days and 100 for the percentage amount difference. This apparently harmless encoding decision turned out to be decisive for the result, as discussed later.

**Table 3.** Attributes of the main experiment

| Attribute | Derivation |
| :--- | :--- |
| delta_days | absolute difference, in days, between the payment date and the issue date |
| delta_valor_pct | absolute relative difference between the amount paid and the service amount |
| valor_pago | nominal amount recorded in the payment |
| nfse_valor_iss | ISS amount declared on the invoice |
| nfse_aliquota | ISS rate declared on the invoice |
| descricao_similarity | cosine between the TF-IDF representations of the payment description and the invoice service description |
| cnpj_match | indicator that the supplier's registration number is present in the payment record |

Source: Original research data

The three algorithms were wrapped in processing pipelines that included, where needed, z-score standardisation, and were tuned by exhaustive grid search with five-fold stratified cross-validation, maximising macro-F1. Table 4 details the preprocessing and search space of each algorithm, and Figure 2 reproduces the pipeline definitions. The test partition was 20% of the records, stratified by class and held out before any fitting.

**Table 4.** Algorithms, preprocessing and hyperparameter grids

| Algorithm | Preprocessing | Hyperparameters searched |
| :--- | :--- | :--- |
| Random forest | none | number of trees (100; 200); maximum depth (unlimited; 10) |
| Support vector machine | z-score standardisation; probability calibration | cost C (0.1; 1.0; 10.0) |
| Logistic regression | z-score standardisation | inverse regularisation C (0.01; 0.1; 1.0; 10.0) |

Source: Original research data

Note: the three algorithms were fitted with class weights inversely proportional to frequency

```python
"random_forest": (
    Pipeline([("clf", RandomForestClassifier(
        n_estimators=mcfg["random_forest"]["n_estimators"],
        class_weight=mcfg["random_forest"]["class_weight"],
        random_state=seed))]),
    {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10]}),
"svm": (
    Pipeline([("scaler", StandardScaler()),
              ("clf", CalibratedClassifierCV(
                  SVC(kernel=mcfg["svm"]["kernel"],
                      class_weight=mcfg["svm"]["class_weight"],
                      random_state=seed), ensemble=False))]),
    {"clf__estimator__C": [0.1, 1.0, 10.0]}),
"logistic_regression": (
    Pipeline([("scaler", StandardScaler()),
              ("clf", LogisticRegression(
                  class_weight=mcfg["logistic_regression"]["class_weight"],
                  max_iter=1000, random_state=seed))]),
    {"clf__C": [0.01, 0.1, 1.0, 10.0]}),
```

**Figure 2.** Definition of the three processing pipelines and their hyperparameter grids, reused unchanged in all three experiments

Source: Original research data

### Ablation experiment

The audit of the main experiment's results raised two explanatory hypotheses, stated in the next section, that required empirical verification. To test them, an ablation experiment was run that held constant the data, the train/test split, the seed, the hyperparameter grid and the cross-validation protocol, varying only the attribute representation. The original experiment's artefacts were preserved; the ablation wrote to its own directory. The four configurations evaluated are listed in Table 5.

**Table 5.** Configurations of the ablation experiment

| Configuration | Attributes | Hypothesis tested |
| :--- | :--- | :--- |
| full | the seven originals | baseline, replicating the main experiment |
| sentinel_fixed | the seven originals plus a match indicator, with neutral imputation in place of the sentinels | the difference between algorithms is a scale artefact |
| no_leak | without delta_days and delta_valor_pct | share of performance attributable to direct leakage |
| no_leak_strict | also without nfse_valor_iss and nfse_aliquota | share attributable to indirect leakage through the ISS fields |

Source: Original research data

Note: in the sentinel_fixed configuration, records without a match received a zero indicator and had their differences imputed by the median of the matched records

### Comparison experiment with independent labelling

The comparison experiment was built on a single principle: the label comes from the generator and no attribute takes part in defining it. The decoupling was obtained by construction, not by removing variables. The generator knows which payment settles which invoice, persists that match in a truth table and keeps it out of reach of feature engineering, which reconstructs only what a reconciliation analyst would observe.

The second departure from the original design concerns the nature of legitimate pairs. In the main experiment, a true pair was a copy: same date and same amount. In the comparison experiment, the payment was derived from the invoice by a legitimate transformation. The payment date was the issue date plus a term drawn once per supplier, so that each supplier came to have a characteristic term estimable from its history, with a declared fraction of legitimate exceptions. The amount paid resulted from applying tax withholdings to the service amount, drawn among the combinations in Table 6, which involve income tax withheld at source [IRRF], social contributions withheld at source [CSRF], the social security contribution [INSS] and ISS withheld by the service taker.

**Table 6.** Withholding combinations allowed when generating true pairs

| Combination | Total withholding | Condition for application |
| :--- | :--- | :--- |
| None | 0.00% | — |
| IRRF | 1.50% | — |
| CSRF | 4.65% | service amount above R$ 5,000.00 |
| IRRF and CSRF | 6.15% | service amount above R$ 5,000.00 |
| Withheld ISS | declared rate, from 2.00% to 5.00% | — |
| IRRF and withheld ISS | 1.50% plus the declared rate | — |
| IRRF, CSRF and withheld ISS | 6.15% plus the declared rate | service amount above R$ 5,000.00 |
| INSS | 11.00% | — |

Source: Original research data

Note: the rates and the application threshold are generator configuration parameters and do not constitute a normative statement; the rates of 1.50%, 4.65% and 11.00% match the withholdings in force for IRRF on professional services, for CSRF and for the social security contribution, whereas the R$ 5,000.00 threshold reproduces the wording of art. 31 of Lei nº 10.833/2003 prior to Lei nº 13.137/2015, which revoked it — it was kept because it creates the conditional interaction the experiment explores, not because it describes the rule in force

The consequence of this transformation is that an amount ratio of 0.9535 no longer meant a divergence and came to mean compatibility with withholding of the social contributions. Because that withholding applies only above the configured threshold, a genuine conditional interaction arose between the service amount and the expected legitimate difference, that is, a non-linear boundary over discrete combinations of rates conditioned on a threshold in another variable. This is precisely the kind of structure on which axis-aligned partitioning, the radial-kernel margin boundary and the linear boundary have different capabilities.

Non-pairs were built so as not to be trivially separable. At least 30% of them received an amount ratio whose implied withholding falls within the tolerance of some legitimate combination, so they cannot be resolved by amount and require other evidence. Symmetrically, a fraction of the true pairs departs from the supplier's typical term, shows a few cents of rounding divergence in the declared ISS or carries an unrelated description, and the non-pairs corrupt a randomly drawn subset of the evidence dimensions rather than a fixed one. Without this deliberate overlap, any single dimension would turn into a new deterministic rule and the original problem would be reproduced. The payment description came to be derived from the invoice's service description by abbreviating words, transposing adjacent characters and dropping accents, which gave the text similarity genuine, if imperfect, informational content.

The evidence matrix came to comprise fourteen attributes, described in Table 7, all computable by someone who does not know whether the pairing is genuine. The compatibility indicators encode domain knowledge from observable fields, and a non-pair can be coincidentally compatible, which is exactly what makes a case hard. The median term per supplier was estimated on the training partition only and applied to the others, falling back on the global training median for unseen suppliers, to avoid transductive leakage of an aggregate statistic.

**Table 7.** Attributes of the comparison experiment

| Attribute | Derivation |
| :--- | :--- |
| delta_dias | signed difference, in days, between the payment date and the issue date |
| razao_valor | ratio of the amount paid to the service amount |
| retencao_implicita_pct | percentage complement of the amount ratio |
| valor_retido_brl | difference, in reais, between the service amount and the amount paid |
| compat_irrf | proximity of the implied withholding to the IRRF rate |
| compat_csrf | proximity of the implied withholding to the CSRF rate |
| compat_irrf_csrf | proximity of the implied withholding to the sum of IRRF and CSRF |
| compat_iss | proximity of the implied withholding to the declared ISS rate |
| compat_combo_iss | proximity to the combinations that include withheld ISS |
| compat_inss | proximity of the implied withholding to the INSS rate |
| acima_limite_csrf | indicator that the service amount is above the CSRF application threshold |
| desvio_prazo_fornecedor | difference between the observed term and the supplier's historical median, estimated on the training partition |
| similaridade_descricao | cosine between the TF-IDF representations of the payment description and the invoice service description |
| mesmo_municipio | indicator that the municipality of the payment's cost centre matches the municipality declared on the invoice |

Source: Original research data

Note: the compatibility indicators use a tolerance of 0.05 percentage point on the implied withholding

### Label leakage guard

The requirement that no evidence dimension separate the classes on its own was turned into an executable check. Before each training run, the guard fitted a depth-one decision tree on each attribute in isolation, measured its accuracy by cross-validation and stopped the pipeline with an error, not a warning, whenever any attribute exceeded the configured ceiling of 0.95. The guard implemented as a regression test the recommendation derived from the main experiment, and its implementation appears in Figure 3.

```python
def stump_accuracies(X, y, cv=3, seed=42):
    """Cross-validated accuracy of a depth-one tree
    fitted on each attribute in isolation."""
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    scores = {}
    for column in X.columns:
        stump = DecisionTreeClassifier(max_depth=1, random_state=seed)
        scores[column] = float(cross_val_score(
            stump, X[[column]], y, cv=folds, scoring="accuracy").mean())
    return pd.Series(scores).sort_values(ascending=False)

def assert_no_leak(X, y, max_accuracy, cv=3, seed=42):
    """Fail the pipeline when an isolated attribute predicts too well."""
    report = stump_accuracies(X, y, cv=cv, seed=seed)
    offenders = report[report > max_accuracy]
    if not offenders.empty:
        raise LeakDetected(
            f"single-attribute accuracy above {max_accuracy}: "
            f"{dict(offenders)}. The label is recoverable from one attribute.")
    return report
```

**Figure 3.** Label leakage guard, run before each training run of the comparison experiment

Source: Original research data

### Decision criterion and statistical comparison

Model selection stopped relying on symmetric metrics and came to derive from the control objective. In reconciliation, letting a divergence through and sending a legitimate payment to manual review do not have the same cost: the first silences an exception and compromises internal control, whereas the second only creates redundant work. The primary metric adopted was therefore the recall of the exception class under the constraint of a minimum precision of 0.90 on that same class. The decision threshold that satisfied the constraint was chosen on the validation partition and applied without readjustment to the test partition, as in Figure 4, and the rate of referral to manual review was reported next to the primary metric, as the operational cost that accompanies the gain in detection. The area under the precision–recall curve, macro-F1 and the full confusion matrix were recorded as secondary metrics.

```python
def choose_threshold(y_val, proba_excecao, min_precision):
    """Highest exception recall whose precision meets the required floor.
    Returns None when no threshold reaches the floor: in that case the
    algorithm is reported as unable to satisfy the constraint."""
    is_exception = (np.asarray(y_val) == 0).astype(int)
    precision, recall, thresholds = precision_recall_curve(
        is_exception, proba_excecao)
    best_threshold, best_recall = None, -1.0
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= min_precision and r > best_recall:
            best_threshold, best_recall = float(t), float(r)
    return best_threshold
```

**Figure 4.** Choice of the decision threshold on the validation partition, under a precision floor for the exception class

Source: Original research data

Hyperparameter tuning required a related correction. The library's default average-precision criterion scores the positive class, which in this design is the legitimate pair, whereas the whole decision criterion concerns the exception class. The scoring function was redefined for the exception class, so that hyperparameter selection, threshold choice and the reported metric all pointed at the same side of the decision.

The split per seed was 60% for training, 20% for validation and 20% for testing, stratified. Ten independent seeds generated ten complete datasets, yielding ten paired values per algorithm and per metric. The algorithms were compared with the Wilcoxon signed-rank test (Wilcoxon, 1945) applied to the pairs, with Holm correction (Holm, 1979) for the three multiple comparisons, as in Figure 5. The effect size, expressed as the median difference in percentage points, was reported next to the p-value, since statistical significance without magnitude does not support a practical recommendation (Demšar, 2006).

```python
def paired_comparisons(per_seed, metric):
    """Paired Wilcoxon between the three algorithms, with Holm correction."""
    wide = per_seed.pivot(index="seed", columns="algorithm", values=metric)
    algorithms = sorted(wide.columns)
    rows, raw_p_values = [], []
    for a, b in combinations(algorithms, 2):
        diff = wide[a] - wide[b]
        result = wilcoxon(wide[a], wide[b], zero_method="wilcox")
        raw_p_values.append(float(result.pvalue))
        rows.append({"algorithm_a": a, "algorithm_b": b,
                     "median_diff": float(diff.median()),
                     "p_value": float(result.pvalue)})
    for row, p_holm in zip(rows, _holm(raw_p_values)):
        row["p_holm"] = p_holm
        row["significant"] = p_holm < ALPHA
    return pd.DataFrame(rows)
```

**Figure 5.** Paired comparison between algorithms across the ten seeds, with Holm correction and effect size

Source: Original research data

### Reproducibility and automated verification

Reproducibility was treated as a design requirement, not a consequence. All simulation, labelling, modelling and evaluation parameters live in a single configuration file; the seed is explicit in every layer; and each scenario's artefacts are persisted in separate directories. The window of simulated dates was anchored to a reference date declared in the configuration rather than to the wall clock, and the list of registration numbers came to be built in draw order rather than by iterating over a set — two corrections whose absence made the main experiment non-reproducible, as the discussion details. A suite of 140 automated tests covers the package's four modules and includes regression assertions for the failure modes discovered during the work, among them leak detection on a deliberately circular dataset, the requirement that no column be constant in the training partition, the requirement that training and test have comparable scales, reproducibility of generation across separate processes, and a check that the decision threshold is never calibrated on the test set.

## Results and Discussion

### Performance in the main experiment

The three algorithms were trained and evaluated in both label-strictness scenarios, on the test partition of 1,504 records. Tables 8 and 9 gather the performance obtained. The random forest scored one on every metric in both scenarios, with a cross-validation standard deviation of exactly zero in the exact scenario. The support vector machine and logistic regression plateaued at a macro-F1 of 0.8892 in the exact scenario, eleven percentage points of macro-F1 below the forest.

**Table 8.** Performance of the algorithms in the exact scenario

| Metric | Random forest | Support vector machine | Logistic regression |
| :--- | ---: | ---: | ---: |
| Accuracy | 1.0000 | 0.9149 | 0.9149 |
| Macro-F1 | 1.0000 | 0.8892 | 0.8892 |
| Macro precision | 1.0000 | 0.9457 | 0.9457 |
| Macro recall | 1.0000 | 0.8590 | 0.8590 |
| Cross-validation | 1.0000 ± 0.0000 | 0.9061 ± 0.0072 | 0.9081 ± 0.0089 |
| ROC-AUC | 1.0000 | 0.9824 | 0.9877 |

Source: Original research results

Note: the cross-validation row shows the mean and standard deviation of macro-F1 over five folds of the training set

**Table 9.** Performance of the algorithms in the fuzzy scenario

| Metric | Random forest | Support vector machine | Logistic regression |
| :--- | ---: | ---: | ---: |
| Accuracy | 1.0000 | 0.9182 | 0.9282 |
| Macro-F1 | 1.0000 | 0.8672 | 0.8791 |
| Macro precision | 1.0000 | 0.9264 | 0.9664 |
| Macro recall | 1.0000 | 0.8371 | 0.8407 |
| Cross-validation | 0.9978 ± 0.0021 | 0.8611 ± 0.0137 | 0.8674 ± 0.0155 |

Source: Original research results

Note: the area under the ROC curve does not apply to this three-class scenario

### Circularity between the labelling rule and the attributes

The most relevant finding of this work is not that the random forest outperformed the other classifiers, but the reason why it scored one with zero dispersion. Inspecting the pipeline exposes a circularity between labelling and representation: the labelling module defines the label as the conjunction of two thresholds on the date difference and the relative amount difference, and the feature engineering module hands those same two quantities to the classifier. The target is therefore an exact and fully known function of two of the seven input variables. The problem given to the algorithms was not to learn to reconcile, but to recover, from examples, a conjunction of two axis-aligned thresholds, a task for which recursive tree partitioning is the perfectly suited inductive hypothesis.

The empirical evidence is unequivocal. In the exact scenario, the relative amount difference and the difference in days accounted for about 0.44 and 0.43, respectively, of the total impurity-reduction importance, roughly 87% of the total; the remaining five variables added up to about 13%, and the supplier registration indicator had an importance of exactly zero. The measure is biased in favour of high-cardinality continuous variables (Strobl et al., 2007), but the observed hierarchy proved robust enough to support the argument.

The pattern matches the leakage described by Kaufman et al. (2012), whose classic symptom is precisely implausibly high and stable performance. Unit accuracy with zero cross-validation standard deviation is thus a diagnosis, not a result. The warning transfers directly to teams that intend to apply supervised learning to reconciliation using labels generated by internal rules, an extremely common situation in practice, since few organisations have independently audited labels.

### The scale artefact in sentinel encoding

The contrast between the random forest and the smooth-boundary models did not come from any conceptual superiority of forests for the reconciliation domain, but from an interaction between the encoding of a missing match and scale standardisation. The 1,119 records without a matching invoice, 14.9% of the dataset, received the sentinel values described in the method. The random forest is invariant to monotonic transformations, since only order matters for choosing split points. The other two models operate on standardised variables, and the sentinel severely distorts the mean and standard deviation, as Table 10 shows.

**Table 10.** Effect of standardisation on the difference in days in the exact scenario

| Original value | Standardised score |
| :--- | ---: |
| 0 days | −0.4188 |
| 5 days | −0.4174 |
| 6 days | −0.4171 |
| 30 days | −0.4103 |
| 9,999 days, sentinel value | 2.3913 |

Source: Original research results

Note: the variable has a mean of 1,490.05 and a standard deviation of 3,558.25 in the exact-scenario dataset

The entire informative range of the difference in days, from zero to thirty days, which is where the reconciliation decision lies, was compressed into an interval of 0.0084 standard deviation, whereas the sentinel alone occupied 2.8 standard deviations. For a margin classifier, the distance between a payment on the exact date and one thirty days late became numerically indistinguishable from noise. The relative amount difference suffered much less distortion, with 0.167 standard deviation separating a perfect match from a 6% divergence, which explains the error pattern precisely: the smooth models kept separating amount divergences and almost completely lost the ability to separate date divergences. The breakdown of the errors confirms this exactly. The test set contains 454 negatives, 208 with no invoice at all, 115 with an amount divergence and 131 with a date divergence; of the 128 false positives made by the two models, all 128 are date divergences. Both detect 208 of 208 missing invoices and 115 of 115 amount divergences, and let 128 of the 131 date divergences through. The dimension the sentinel crushed is exactly, and only, the one the models lost.

A further sign of this degeneration is that the support vector machine and logistic regression produced strictly identical confusion matrices in the exact scenario. Two models from different families converging on the same partition of the test space suggests that both collapsed to essentially the same univariate decision on the relative amount difference.

### Results of the ablation experiment

Both hypotheses were tested in the ablation experiment, whose macro-F1 results appear in Table 11 and Figure 6. The baseline configuration reproduced the main experiment's values to four decimal places, despite running under newer library versions, which constitutes an independent check of the pipeline's reproducibility.

**Table 11.** Macro-F1 by attribute configuration, scenario and algorithm

| Configuration | Scenario | Random forest | Support vector machine | Logistic regression |
| :--- | :--- | ---: | ---: | ---: |
| full | exact | 1.0000 | 0.8892 | 0.8892 |
| sentinel_fixed | exact | 1.0000 | 1.0000 | 1.0000 |
| no_leak | exact | 0.7630 | 0.7837 | 0.6815 |
| no_leak_strict | exact | 0.4976 | 0.4111 | 0.4608 |
| full | fuzzy | 1.0000 | 0.8672 | 0.8791 |
| sentinel_fixed | fuzzy | 1.0000 | 1.0000 | 1.0000 |
| no_leak | fuzzy | 0.6526 | 0.6334 | 0.6540 |
| no_leak_strict | fuzzy | 0.3313 | 0.2741 | 0.2618 |

Source: Original research results

![Bar chart of macro-F1 by attribute configuration (full, sentinel_fixed, no_leak, no_leak_strict) for the random forest, support vector machine and logistic regression in the exact scenario](../../data/results/figuras/figura6_ablacao_exact.png)

**Figure 6.** Macro-F1 by attribute configuration and algorithm in the exact scenario

Source: Original research results

Note: the full configuration replicates the main experiment; sentinel_fixed replaces the sentinel values with an explicit match indicator

In the configuration that replaced the sentinel with an explicit match indicator, the three algorithms reached identical and perfect confusion matrices in the exact scenario, with 454 true negatives, 1,050 true positives and no error in either direction. The eleven-percentage-point macro-F1 gap between the random forest and the smooth-boundary models disappeared completely through a single preprocessing change, without any change of algorithm, hyperparameter or data. The explanation is geometrically transparent: with the sentinel removed, reconciled records sit exactly at the origin of the plane formed by the two differences, and the class becomes linearly separable with the help of the indicator. What kept logistic regression from finding that boundary was not its linear nature, but the scale crushing caused by the sentinel value.

The first research question must therefore be reformulated. Under an adequate representation the three algorithms are equivalent in this design, and the relevant question is no longer which algorithm to choose but how to represent the data. The result is in line with the classic literature on empirical comparison of classifiers: the no-free-lunch theorem (Wolpert, 1996) and the survey by Fernández-Delgado et al. (2014) over 121 datasets support that the relative performance of algorithms is conditioned on the representation and is not intrinsic to the model family.

Removing the rule's variables confirmed the circularity hypothesis and produced a further noteworthy result: the order between algorithms flipped in the exact scenario, with the support vector machine outperforming the random forest. The conclusion that the forest would be the best algorithm for reconciliation did not survive the removal of the leakage. More important than the size of the drop is the nature of what remained detectable, broken down in Table 12.

**Table 12.** Breakdown of correct detections in the negative class without the rule's variables, exact scenario

| Algorithm | Records without a match detected | Date or amount divergences detected |
| :--- | ---: | ---: |
| Random forest | 208 of 208 (100.0%) | 1 of 246 (0.4%) |
| Support vector machine | 208 of 208 (100.0%) | 22 of 246 (8.9%) |
| Logistic regression | 208 of 208 (100.0%) | 65 of 246 (26.4%) |

Source: Original research results

Without the rule's variables, the models perfectly identified the payments with no invoice at all and remained practically blind to date and amount divergences. Detecting a missing match, however, is not machine learning: it is the result of a relational join, obtainable with a one-line query in a structured language. The apparently residual signal is itself indirect leakage, since the ISS amount and rate are zero exactly when no invoice exists. The configuration that also removed those fields confirmed the argument: performance collapsed to the 0.41–0.50 macro-F1 range in the exact scenario and 0.25–0.35 in the fuzzy scenario, and the support vector machine degenerated into the trivial classifier, predicting the majority class in every case and reaching an accuracy of 0.6981, exactly the proportion of reconciled records in the dataset.

Once direct and indirect leakage are discounted, the synthetic dataset of the main experiment contains no learnable signal for the reconciliation task. This result does not invalidate the work: it delimits precisely what the work shows and turns what would have been a hidden weakness into an explicit, measured methodological finding.

### Error asymmetry and cost in internal control

The finding of greatest practical relevance lies in the structure of the errors, not in their number. Table 13 shows the distribution of errors in the exact-scenario test set.

**Table 13.** Distribution of errors in the exact-scenario test set

| Algorithm | True negatives | False positives | False negatives | True positives |
| :--- | ---: | ---: | ---: | ---: |
| Random forest | 454 | 0 | 0 | 1,050 |
| Support vector machine | 326 | 128 | 0 | 1,050 |
| Logistic regression | 326 | 128 | 0 | 1,050 |

Source: Original research results

In financial reconciliation, the two kinds of error have asymmetric consequences. A false negative, that is, a legitimate payment classified as not reconciled, creates unnecessary manual work: there is an operational cost, but the control remains intact. A false positive silences the divergence: a payment 3% above the invoice amount, or issued thirty days outside the accrual period, is approved automatically and never reaches human review. The smooth-boundary models made 128 false positives and no false negatives, that is, they erred systematically in the direction that costs internal control the most. In the fuzzy scenario the effect is even more pronounced, as 107 of the 226 partially reconciled records in the test set, or 47.3%, were certified as fully compliant. The ablation shows that the bias worsens when the leakage is removed: without the rule's variables, the random forest came to make 245 false positives against no false negatives.

The tendency to silence exceptions is not accidental. It follows from the combination of class imbalance, with 69.8% reconciled records, and a symmetric selection metric that treats a redundant manual review and an undetected divergence as equivalent. The point links the work to the theory of cost-sensitive learning (Elkan, 2001) and supports the recommendation that, in reconciliation, the model selection metric be the recall of the exception class under a minimum acceptable precision constraint, and not accuracy or macro-F1. The ordering is not a preference of this work: the Brazilian standard on audit sampling defines sampling risk as the possibility of two wrong conclusions and ranks them explicitly, attributing to the conclusion that there is no material misstatement when one exists the impairment of audit effectiveness, and to the opposite conclusion only that of efficiency, because it creates additional work (Conselho Federal de Contabilidade [CFC], 2009). The silenced divergence is the effectiveness error; the redundant review, the efficiency error. A system that refers 15% of records to manual review without letting any divergence through is operationally superior to one that automates 91% while silencing a substantial share of the exceptions.

The area under the ROC curve and macro-F1 diverged: the support vector machine reached an area of 0.9824 in the exact scenario with a macro-F1 of only 0.8892, and logistic regression 0.9877 with the same 0.8892. With imbalanced classes, the ROC curve gives an optimistic impression of performance, a phenomenon documented by Saito and Rehmsmeier (2015). The ablation offers an even more didactic illustration, since in the fuzzy scenario without leakage the support vector machine reached an accuracy of 0.8491 without ever predicting the intermediate class, with zero recall on that class.

### The effect of label strictness

The second research question, on the effect of label strictness on the ranking of the algorithms, calls for a more careful answer than the metrics table suggests. The generator produces date divergences by a shift drawn between six and thirty days, and amount divergences by a factor drawn between 3% and 20%. The tolerances of the fuzzy scenario are five days and 2%. By construction, therefore, no record falls inside the tolerance window, which the empirical check confirms: in the fuzzy dataset, all 5,250 records of the fully reconciled class have date and amount differences of exactly zero, with coinciding maxima and minima.

The fuzzy scenario thus did not evaluate robustness to tolerances. What it actually did was split the negative class in two, applying a proximity rule over the same leaked variables. The legitimate conclusion is narrower and more interesting: increasing the granularity of the label, from binary to ternary, degraded the performance of the smooth-boundary models, with macro-F1 dropping from 0.8892 to 0.8672 for the support vector machine, without affecting the random forest, because the intermediate class is defined by thresholds on the same quantities. The added difficulty falls entirely on the boundary between the partial and the reconciled class, which is precisely the region compressed by standardisation.

### Comparison between algorithms under independent labelling

The previous sections established that the original comparison had no discriminating power. The corrective experiment, with the label derived from ground truth, answers the first research question. Before any training, the leakage guard evaluated each attribute in isolation across the ten runs. The highest single-attribute accuracy was 0.8173, well below the 0.95 ceiling, and the six most informative attributes appear in Table 14.

**Table 14.** Depth-one tree accuracy per attribute, mean of the ten runs

| Attribute | Accuracy |
| :--- | ---: |
| razao_valor | 0.8131 |
| retencao_implicita_pct | 0.8131 |
| similaridade_descricao | 0.7852 |
| mesmo_municipio | 0.7750 |
| desvio_prazo_fornecedor | 0.7711 |
| delta_dias | 0.7429 |

Source: Original research results

Note: the remaining eight attributes scored 0.7302 or below; seven of them scored exactly 0.70, the majority-class rate

No attribute separated the classes on its own, and the most informative ones are pair-level evidence rather than the labelling rule itself. The task given to the algorithms was therefore genuine. On it, ten independent datasets of 7,500 records each produced the results in Table 15.

**Table 15.** Performance in the comparison experiment, mean of ten runs

| Indicator | Random forest | Support vector machine | Logistic regression |
| :--- | ---: | ---: | ---: |
| Exception recall | 0.8216 | 0.8002 | 0.7698 |
| Standard deviation across seeds | 0.0169 | 0.0191 | 0.0251 |
| Exception precision | 0.8931 | 0.8996 | 0.9030 |
| Referral rate | 27.62% | 26.71% | 25.59% |
| PR-AUC | 0.9356 | 0.9258 | 0.9153 |

Source: Original research results

Note: the three algorithms reached the 0.90 precision floor on the validation partition in all ten runs; the precision shown is the test precision, measured at the threshold calibrated on validation

Test precision sat slightly below the floor for the random forest and the support vector machine, and above it for logistic regression, which is expected and desirable, since the threshold was calibrated on validation and applied to the test set without readjustment. Test precision exactly at the floor would suggest that the test set had taken part in the calibration. The paired comparisons between algorithms appear in Table 16.

**Table 16.** Paired Wilcoxon comparison on exception recall, with Holm correction

| Comparison | Median difference | p | Corrected p | Significant |
| :--- | ---: | ---: | ---: | ---: |
| Random forest vs. support vector machine | 1.44 p.p. | 0.0020 | 0.0059 | yes |
| Random forest vs. logistic regression | 4.56 p.p. | 0.0020 | 0.0059 | yes |
| Support vector machine vs. logistic regression | 3.00 p.p. | 0.0039 | 0.0059 | yes |

Source: Original research results

Note: the median difference always favours the first algorithm in the comparison; p.p. stands for percentage points

The pairwise comparisons should only be read once the global hypothesis of equivalence has been rejected, which is the protocol recommended by Demšar (2006) and which the first version of this analysis did not follow. The Friedman omnibus test over the ten runs appears in Table 17, and was applied to three metrics, not only the primary one, for a reason that Table 18 makes explicit.

**Table 17.** Friedman omnibus test over the ten runs

| Metric | χ² | p | Order of mean ranks | Rejects equivalence |
| :--- | ---: | ---: | ---: | ---: |
| Exception recall | 18.20 | 0.00011 | forest 1.0; SVM 2.1; logistic 2.9 | yes |
| Exception PR-AUC | 18.20 | 0.00011 | forest 1.0; SVM 2.1; logistic 2.9 | yes |
| Exception precision | 4.20 | 0.1225 | logistic 1.6; SVM 1.9; forest 2.5 | no |

Source: Original research results

Reading the three rows of Table 17 together answers the strongest objection that can be raised against this comparison. Recall is measured at each algorithm's operating point, and those points do not fall on the same realised precision: 0.893 for the forest against 0.903 for logistic regression. Since recall and precision trade off against each other, the forest's recall advantage could, in principle, be merely the effect of a looser threshold. Two pieces of evidence from the same ten runs rule out that reading. The first is that the ranking holds under PR-AUC, which summarises the whole curve and depends on no threshold. The second is that the precision differences do not survive testing: no pairwise comparison is significant after correction, and the omnibus does not reject equivalence among the three, with a p of 0.1225. Table 18 gathers the pairwise comparisons on the two secondary metrics.

**Table 18.** Paired Wilcoxon comparison on the secondary metrics, with Holm correction

| Metric and comparison | Median difference | p | Corrected p | Significant |
| :--- | ---: | ---: | ---: | ---: |
| PR-AUC: forest vs. SVM | +0.0088 | 0.0020 | 0.0059 | yes |
| PR-AUC: forest vs. logistic | +0.0210 | 0.0020 | 0.0059 | yes |
| PR-AUC: SVM vs. logistic | +0.0125 | 0.0059 | 0.0059 | yes |
| Precision: forest vs. SVM | −0.0033 | 0.2754 | 0.5508 | no |
| Precision: forest vs. logistic | −0.0076 | 0.0840 | 0.2520 | no |
| Precision: SVM vs. logistic | −0.0078 | 0.5566 | 0.5566 | no |

Source: Original research results

Note: the median difference always favours the first algorithm in the comparison

Taken together: the random forest ranks exceptions better and detects more of them, without operating at a statistically lower precision. Without any one of these pieces of evidence, the claim that the forest detects more divergences would be a claim about the chosen threshold, not about the algorithm. What remains is to report the full confusion matrix, which is check no. 6 of this work's own checklist and which Table 19 provides, summed over the ten runs, in internal-control units instead of percentage points.

**Table 19.** Confusion matrix of the exception class, summed over the ten runs

| Algorithm | Exceptions detected | Divergences silenced | Redundant reviews | Pairs correctly approved |
| :--- | ---: | ---: | ---: | ---: |
| Random forest | 3,697 | 803 | 446 | 10,054 |
| Support vector machine | 3,601 | 899 | 405 | 10,095 |
| Logistic regression | 3,464 | 1,036 | 374 | 10,126 |

Source: Original research results

Note: 15,000 test pairs in all, of which 4,500 are exceptions; replacing the forest with logistic regression silences 233 more divergences and saves 72 manual reviews

The answer to the first research question can be stated precisely. Under labelling independent of the attributes and a criterion derived from the control objective, the random forest detected 1.44 percentage points more divergences than the support vector machine and 4.56 percentage points more than logistic regression, while keeping the 0.90 precision floor on validation, with a p-value of 0.0059 after Holm correction across ten runs, at the cost of referring 27.6% of the batch to manual review. The ranking is rejected as equivalent by the Friedman omnibus test, holds under PR-AUC, which does not depend on a threshold, and is not explained by unequal precision, since on exception precision the omnibus does not reject equivalence.

Statistical significance is not, however, the same as practical relevance. The difference between the random forest and the support vector machine is 1.44 percentage points, which in a portfolio of 7,500 payments with 30% exceptions corresponds to about 32 additional divergences. The test detects it because the order repeats across the ten runs, not because the magnitude is large, which is exactly why the effect size is reported next to the p-value (Demšar, 2006). The defensible reading is more nuanced than a simple win for the forest: the three algorithms are operationally close, with logistic regression about 4.6 points behind, and the choice between them may legitimately rest on interpretability and cost instead of performance. For a system that must justify each exception to an auditor, the coefficients of a logistic regression may be worth more than 4.6 percentage points of recall (Rudin, 2019).

Stability across runs was similar for the three algorithms, with standard deviations of 0.0169, 0.0191 and 0.0251, and none showed bimodal behaviour or distinct regimes, as Table 20 and Figure 7 show. The random forest obtained the best result in all ten runs, which supports the significance despite the narrow margin.

**Table 20.** Exception-class recall per seed

| Seed | Random forest | Support vector machine | Logistic regression |
| :--- | ---: | ---: | ---: |
| 0 | 0.829 | 0.793 | 0.738 |
| 1 | 0.820 | 0.791 | 0.742 |
| 2 | 0.829 | 0.824 | 0.789 |
| 3 | 0.840 | 0.827 | 0.776 |
| 4 | 0.811 | 0.764 | 0.738 |
| 5 | 0.851 | 0.802 | 0.809 |
| 6 | 0.802 | 0.793 | 0.780 |
| 7 | 0.796 | 0.789 | 0.764 |
| 8 | 0.813 | 0.798 | 0.764 |
| 9 | 0.824 | 0.820 | 0.798 |

Source: Original research results

![Line chart of exception-class recall per random seed, from 0 to 9, for the random forest, support vector machine and logistic regression](../../data/results/figuras/figura7_recall_por_semente.png)

**Figure 7.** Exception-class recall per seed and per algorithm, under a 0.90 precision floor

Source: Original research results

### A third silent failure mode

The final audit of the corrective experiment revealed a third failure mode, inside the very experiment built to avoid the first two. The initial version of the generator issued one payment per supplier. The term-deviation attribute is defined as the difference between the observed term and the supplier's historical median, fitted on the training partition only; with a single payment per supplier, that median is the row's own value, and the attribute became identically zero on every training row. Since no test supplier appeared in training, all of them fell back on the global median and the column had real dispersion outside training.

The consequence was not an inert attribute but a scale artefact. Measured over 2,000 records, the column's standard deviation was zero in training and 25.48 in test. The standardiser finds zero variance, applies its internal guard and keeps unit scale, so test rows pass without normalisation into a dimension the model never saw vary: 83.75% of them with a standardised score above five in absolute value, with a maximum of 90. The radial kernel collapses. The random forest, invariant to monotonic transformations, and logistic regression, whose coefficient for a constant column is zero, were not affected; only the support vector machine was. Table 21 summarises the effect of the correction.

**Table 21.** Effect of the generator correction on the support vector machine

| Indicator | Before the correction | After the correction |
| :--- | ---: | ---: |
| Mean exception recall | 0.2211 | 0.8002 |
| Standard deviation across seeds | 0.1248 | 0.0191 |
| Position in the ranking | third | second |
| Distance to first place | 51.9 p.p. | 1.4 p.p. |
| Bimodal behaviour | four seeds near 0.08 | absent |

Source: Original research results

Note: the correction consisted of assigning five invoices to each supplier, so that the median term became a real statistic

The earlier conclusion, that both other algorithms outperformed the support vector machine by margins above 48 percentage points, was entirely an artefact. The probability-calibration explanation initially offered for the bimodality was a plausible rationalisation of a symptom whose cause was something else.

The leakage guard did not catch this defect, and the reason is instructive. The instrument tests whether some attribute predicts the label too well; a constant attribute predicts the label poorly, scoring exactly the majority-class rate, and passes the test precisely because it contains no information at all. The check that would have caught it is a different one: requiring that no column be constant in the training partition and that training and test have comparable scales. Both assertions became part of the test suite, the second standardising the test partition with the training mean and standard deviation and requiring that no column throw an appreciable fraction of rows beyond five standard deviations. Table 22 gathers the six failure modes documented over the course of the work.

### Two further modes, in the main experiment's generator

The audit of the generator revealed two other modes of the same family, this time in the main experiment. The first undoes a premise of the design: the two sources were presented as heterogeneous and independent, and independence of format does exist, but statistical independence did not. The functions that generate the payment spreadsheet and the set of invoices received the same seed, and each instantiated its own pseudorandom generator; two instances built with the same seed produce the same sequence, and both drew registration numbers from it. The first 7,500 registration numbers of an independent stream with the same seed are, one by one, the same as those in the supplier register.

The consequences are measurable in the published dataset. In 277 invoices the service taker's registration number matches that of some supplier, an impossible situation in the simulated scenario; in 14 of them the provider's legal name is identical to the taker's; and in 18 the provider's number collides with that of another payment. With 7,500 registration numbers drawn from a space of 10¹², the probability of a single chance coincidence is of the order of 10⁻⁵, so 277 is not chance. It is the 18 provider collisions that reach the model: because the join is made on the registration number, a payment whose number appears in two invoices produces two rows, and this is the origin of the 7,518 records. Because the label is a threshold on each row's differences, twelve of those payments appear simultaneously as reconciled and as not reconciled, and some fall into both partitions of the stratified split at once.

The second further mode contradicted the reproducibility claim. Registration numbers were accumulated in a set and then converted to a list, and the iteration order of a set of strings depends on the process's hash seed, which the interpreter randomises at every start. Each run therefore assigned a different registration number to each payment, while drawing identical amounts, dates and descriptions. On top of that, the date window was anchored to the wall clock, so the simulated dataset depended on the day it was generated. The combined effect was small — regenerating the dataset on another day produced 7,520 rows instead of 7,518 — and that is exactly why it went unnoticed.

The most instructive aspect is that an automated test had been written for this property, and it passed: it called the generator twice and compared the results, but both calls happened in the same process, where the hash seed is fixed for the interpreter's whole lifetime. The test measured internal consistency, not reproducibility, and would have passed under any ordering. The check that sees the defect requires two processes with different hash seeds, and it became part of the suite. Both causes were corrected — the list is built in draw order and the date window is anchored to an explicit configuration date — and the artefacts were regenerated under the corrected implementation. Neither mode affects the comparison experiment, whose generator uses a single pseudorandom stream for both sources.

A sixth mode, found in the final review, is the purest of the series. The main pipeline built the attribute matrix from the data frame still in memory, whereas the ablation and the notebooks build it from the CSV file the pipeline had just written: two code paths for the same computation, on the same data. The round trip through the file perturbs the relative amount difference by about 10⁻¹⁴ in 94 of the 7,518 rows, meaningless representation noise; that noise passes through standardisation, reaches the optimiser near its convergence tolerance, and moves logistic regression's cross-validated macro-F1 in the fuzzy scenario from 0.8678 to 0.8674. An input difference of 10⁻¹⁴ produces an output difference of 3×10⁻⁴, which falls exactly in the fourth decimal place at which this work reports its metrics. The pipeline came to read back the file it writes, so that there is a single path and a single set of numbers, and the choice of direction is not arbitrary: the CSV is the artefact a reader can inspect and, therefore, should be the one the model is fitted on.

**Table 22.** Silent failure modes identified

| Failure mode | Symptom | False conclusion it would support |
| :--- | :--- | :--- |
| Label leakage | accuracy of 1.0000 with zero dispersion | the random forest solves reconciliation |
| Selection metric pointed at the wrong class | exception recall of 0.088 | the support vector machine is unfit for reconciliation |
| Attribute constant in the training partition | recall of 0.221, with bimodality | the support vector machine is unstable in reconciliation |
| Pseudorandom stream shared between the two sources | 7,518 rows from 7,500 payments; 277 invoices whose service taker is a supplier | the two simulated sources are independent |
| Set order and date window dependent on the process and the clock | none; the existing reproducibility test passed | the main experiment's results are reproducible |
| Two code paths for the same computation | two tables disagreeing in the fourth decimal place | the ablation reproduces the main experiment exactly |

Source: Original research results

The six modes belong to the same family: a silent configuration that produces a substantive, wrong conclusion without manifesting as an error. All six would produce plausible tables, and five of them were found by adversarial code review, not by the metrics, which in no case showed any sign of anomaly. The fifth is the most instructive, because an automated test had been written precisely for it and that test passed, measuring a weaker property than its name announced. The methodological contribution is not just the recommendation to check for leakage: it is that the check must be systematic and adversarial, because each instrument only sees the failure mode it was built for, and that a test can attest to less than it appears to. The leakage guard is useful and would not have detected five of the six.

The artefact was reduced, not eliminated. With five payments per supplier, a training row contributes to its own supplier's median, so its deviation is shrunk relative to that of a test row: in the 7,500-record configuration, the standard deviation is about 13.8 in training against 18.5 in test. It is the same mechanism at a much smaller amplitude, with the extreme rows at around five standard deviations, a thin tail rather than the 83.75% that collapsed the kernel. A median computed excluding the observation itself would remove the residue.

### Limitations of the study

The limitations are presented in Tables 23, 24 and 25, organised by the type of validity affected and ranked by severity. The list is deliberately long, because precisely delimiting the reach of the results is part of the contribution of a work whose central finding is how easily plausible conclusions form on flawed designs.

**Table 23.** Internal validity limitations

| Limitation | Severity |
| :--- | :--- |
| The labelling rule and the predictive attributes share the same quantities in the main experiment, which makes the target a deterministic function of the inputs | High |
| The ISS amount and rate indirectly encode whether a match exists, which constitutes secondary leakage | Medium |
| Saturated metrics collapse the confidence intervals and drain statistical power from the comparison between algorithms | High |
| Cross-validation shuffles records that have a temporal dimension, which allows training on information later than the test | Medium |
| Sentinel value encoding distorts standardisation and fully accounts for the difference between algorithms in the main experiment | High |
| The two simulated sources shared the same pseudorandom stream, which produced 277 invoices whose service taker is a supplier and 18 spurious joins, with twelve payments receiving contradictory labels | High |
| The main experiment was not reproducible, because of process-dependent set order and a date window anchored to the clock; corrected, and the artefacts were regenerated | Resolved |

Source: Original research results

**Table 24.** External validity limitations

| Limitation | Severity |
| :--- | :--- |
| No real accounting record was used; the joint distribution of the variables reflects the generator's decisions | High |
| Matching is one-to-one by construction, which eliminates the many-payments-to-many-invoices scenario that dominates in practice | High |
| The tolerance windows of the fuzzy scenario are empty, so robustness to tolerances was not exercised | High |
| Once leakage is removed, the main experiment's dataset contains no residual learnable signal for the task | High |
| The scope is limited to service invoices; goods invoices, import invoices and non-tax documents were left out | Medium |
| The dataset covers one year with no regime changes, which prevents evaluating model degradation over time | Medium |
| A single dataset size was evaluated, with no learning curve to indicate the minimum volume needed | Low |

Source: Original research results

**Table 25.** Construct limitations and limitations specific to the comparison experiment

| Limitation | Severity |
| :--- | :--- |
| Reconciliation was operationalised as a threshold rule, whereas in accounting practice it involves professional judgement and business context | High |
| There was no labelling by a human expert, which prevents measuring agreement between the model and accounting judgement | High |
| The CSRF application threshold reproduces the wording prior to Lei nº 13.137/2015, which revoked it; the rates were checked and are in force, but the generator describes a regime that is no longer positive law | High |
| The difficulty of the task is a chosen parameter; another value would move the three means and could change the ranking | High |
| A residue of the scale artefact persists in the supplier term deviation, at a much smaller amplitude | Medium |
| The text similarity is refitted on every call, so it cannot be computed for a single new record at inference time | Medium |
| The leakage guard sees a single failure mode, and the list of checks is still being enumerated by discovery | High |
| Two attributes of the comparison experiment are exactly collinear, so the matrix has fourteen columns and thirteen directions; the effect penalises the two standardised algorithms, never the forest | Low |
| The threshold is chosen by maximising recall among thousands of cut points that reach the floor on validation, which selects points whose precision crossed the floor by luck and makes the floor hold only approximately on test | Medium |
| Non-pairs are the same payment with corrupted evidence, not a payment to another counterparty, so the algorithms discriminate corrupted evidence rather than the wrong counterparty | High |
| The ten datasets are ten samples of a single generating process, not ten distinct datasets, so the tests measure stability under resampling and not generalisation across regimes | Medium |
| The comparison experiment did not include a rule-based baseline, so the superiority of supervised learning over rule-based reconciliation was not tested under independent labelling | High |

Source: Original research results

### Contributions and practical implications

The main theoretical contribution is quantified empirical evidence of circularity in rule-based labelling. The phenomenon of leakage is known; what this work adds is the measurement of its magnitude and the demonstration that it is the expected, not exceptional, failure mode of applying supervised learning to accounting reconciliation, given that the vast majority of organisations only have labels generated by their own business rules. The drop in macro-F1 from 1.0000 to 0.7630 and, without the ISS proxies, to 0.4976 establishes a numerical reference for the cost of ignoring that circularity.

The second contribution is the experimental demonstration that the ranking between algorithms was an artefact of the representation, which shifts the research question from which algorithm is best to which representation makes the task learnable. The third characterises cost asymmetry as a structural property of the domain and grounds the proposition that, in reconciliation, the evaluation metric should derive from the control objective and not from statistical convention. The fourth makes explicit the gap between reconciliation and record linkage: the one-to-one simplification eliminates candidate generation, which is the computational and conceptual core of the Fellegi and Sunter (1969) model, and it is there that the academic experiment departs most from reconciliation as practised in organisations.

On the practical side, the repository delivers a complete, deterministic pipeline, with fully externalised parameters, a fixed seed and an automated test suite, whose reproduction of the original values under newer libraries is evidence of the artefact's robustness, in line with the criteria for reproducible computational research (Peng, 2011). The synthetic data generator for Brazilian tax reconciliation, with valid registration numbers, municipality codes, service-list items and parameterisable divergence modes, is a reusable resource for teaching, prototyping and comparing methods in a context where the scarcity of public datasets is a recognised obstacle (Patki et al., 2016). The leakage guard, in turn, turns into an executable check the recommendation to treat high accuracy as an alarm, and can be transplanted directly into any project that trains on labels generated by internal rules. The layered architecture, with externalised configuration and artefacts persisted per scenario, mitigates the technical debt characteristic of machine learning systems (Sculley et al., 2015). The checklist derived from the study appears in Appendix A.

The most direct implication for automation investment decisions is that, when the reconciliation rule is known and deterministic, machine learning adds no value over implementing the rule directly: the random forest merely reproduced, with added computational cost and opacity, what a structured query executes exactly. Supervised learning is only justified where the rule is unknown, unstable or intractable, as in the textual noise of company names, in matching many payments to many invoices and in prioritising exceptions by risk.

The research agenda that remains open is well delimited. It includes extending the design to matching many payments to many invoices, incorporating candidate generation and adopting the probabilistic framework as a baseline; validating on real data, through an agreement that allows expert labelling and measurement of agreement with accounting judgement; comparing against an unsupervised baseline, immune by construction to the leakage documented here; sweeping the difficulty of the task to map how the difference between algorithms responds to different regimes; and expanding the catalogue of automated representation checks, turning into a tool the lesson drawn from the third failure mode.

## Conclusion

The representation assigned to the data and the provenance of the label weighed more on automated accounting reconciliation than the choice of algorithm. With the label derived from a rule applied to the attributes themselves, the classifiers reproduced the rule instead of learning the task, and the random forest's apparent advantage disappeared with a single encoding change. With an independent label and a criterion derived from the control objective, the random forest was the algorithm that best automated reconciliation, detecting more divergences without a statistically significant loss of precision, but by a margin narrow enough for interpretability and operational cost to be legitimate selection criteria. The hypothesis that machine learning would improve the quality of reconciliation was not confirmed where the rule was known, since the models merely reproduced it with greater opacity; where the evidence was ambiguous, they automated the triage of exceptions with controlled precision, but without being confronted with a rule-based approach, so their superiority over it remained undemonstrated.

For financial management, it follows that investment in supervised learning is justified only where the reconciliation rule is unknown, unstable or intractable, since known and deterministic rules are executed exactly and transparently by a structured query, without the computational cost and opacity of a model. It also follows that model selection in reconciliation should rest on the recall of the exception class under a precision floor, and not on accuracy or macro-F1, because symmetric metrics lead to classifiers that err in the direction that silences divergences and compromises internal control. High accuracy, in this domain, should act as an alarm that triggers an audit of the label's provenance, not as an acceptance criterion; and automated verification, though indispensable, does not replace adversarial code review, because each instrument only sees the failure mode it was built for.

## References

Appelbaum, D.; Kogan, A.; Vasarhelyi, M.A. 2017. Big data and analytics in the modern audit engagement: research needs. Auditing: A Journal of Practice & Theory 36(4): 1-27.

Bao, Y.; Ke, B.; Li, B.; Yu, Y.J.; Zhang, J. 2020. Detecting accounting fraud in publicly traded U.S. firms using a machine learning approach. Journal of Accounting Research 58(1): 199-235.

Breiman, L. 2001. Random forests. Machine Learning 45(1): 5-32.

Christen, P. 2012. Data Matching: Concepts and techniques for record linkage, entity resolution and duplicate detection. Springer, Heidelberg, Germany.

Conselho Federal de Contabilidade [CFC]. 2009. Resolução CFC nº 1.222, de 27 de novembro de 2009. Aprova a NBC TA 530 – Amostragem em Auditoria. Diário Oficial da União, Brasília, 4 Dec. 2009.

Cormier, K.; Gagnier, K.; Padron-Uy, J.; Sareen, D.; Parihar, A.; Khmelevsky, Y.; Hains, G.; Wong, A. 2025. Data extraction, transformation, and loading (ETL) process automation and data warehouse implementation for algorithmic trading machine learning modelling. In: IEEE International Systems Conference (SysCon), 2025, Montreal, QC, Canada. Proceedings... p. 1-8. Available at: <https://doi.org/10.1109/SysCon64521.2025.11014812>. Accessed: 31 Mar. 2026.

Cortes, C.; Vapnik, V. 1995. Support-vector networks. Machine Learning 20(3): 273-297.

Demšar, J. 2006. Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research 7: 1-30.

Elkan, C. 2001. The foundations of cost-sensitive learning. In: International Joint Conference on Artificial Intelligence, 2001, Seattle, WA, USA. Proceedings... p. 973-978.

Fellegi, I.P.; Sunter, A.B. 1969. A theory for record linkage. Journal of the American Statistical Association 64(328): 1183-1210.

Fernández-Delgado, M.; Cernadas, E.; Barro, S.; Amorim, D. 2014. Do we need hundreds of classifiers to solve real world classification problems? Journal of Machine Learning Research 15: 3133-3181.

França, R.P.; Monteiro, A.C.B.; Arthur, R.; Iano, Y. 2021. An overview of deep learning in big data, image, and signal processing in the modern digital age. p. 63-87. In: Piuri, V.; Raj, S.; Genovese, A.; Srivastava, R. Trends in Deep Learning Methodologies. Academic Press. Available at: <https://doi.org/10.1016/B978-0-12-822226-3.00003-9>. Accessed: 31 Mar. 2026.

Holm, S. 1979. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics 6(2): 65-70.

Kaufman, S.; Rosset, S.; Perlich, C.; Stitelman, O. 2012. Leakage in data mining: formulation, detection, and avoidance. ACM Transactions on Knowledge Discovery from Data 6(4): 1-21.

Mahmud, A.; Fitriani, N. 2025. The role of financial reconciliation in improving the accuracy of monthly and annual financial reports at CV X. International Journal Administration, Business & Organization 6(2): 301-312. Available at: <https://doi.org/10.61242/ijabo.25.561>. Accessed: 31 Mar. 2026.

Patki, N.; Wedge, R.; Veeramachaneni, K. 2016. The synthetic data vault. In: IEEE International Conference on Data Science and Advanced Analytics, 2016, Montreal, Canada. Proceedings... p. 399-410.

Pedregosa, F.; Varoquaux, G.; Gramfort, A.; Michel, V.; Thirion, B.; Grisel, O.; Blondel, M.; Prettenhofer, P.; Weiss, R.; Dubourg, V.; Vanderplas, J.; Passos, A.; Cournapeau, D.; Brucher, M.; Perrot, M.; Duchesnay, E. 2011. Scikit-learn: machine learning in Python. Journal of Machine Learning Research 12: 2825-2830.

Peng, R.D. 2011. Reproducible research in computational science. Science 334(6060): 1226-1227.

Rudin, C. 2019. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence 1(5): 206-215.

Saito, T.; Rehmsmeier, M. 2015. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS ONE 10(3): e0118432.

Sculley, D.; Holt, G.; Golovin, D.; Davydov, E.; Phillips, T.; Ebner, D.; Chaudhary, V.; Young, M.; Crespo, J.-F.; Dennison, D. 2015. Hidden technical debt in machine learning systems. In: Conference on Neural Information Processing Systems, 2015, Montreal, Canada. Proceedings... p. 2503-2511.

Soni, N.; Sharma, E.K.; Singh, N.; Kapoor, A. 2020. Artificial intelligence in business: from research and innovation to market deployment. Procedia Computer Science 167: 2200-2210. Available at: <https://doi.org/10.1016/j.procs.2020.03.272>. Accessed: 31 Mar. 2026.

Strobl, C.; Boulesteix, A.-L.; Zeileis, A.; Hothorn, T. 2007. Bias in random forest variable importance measures: illustrations, sources and a solution. BMC Bioinformatics 8: 25.

Wilcoxon, F. 1945. Individual comparisons by ranking methods. Biometrics Bulletin 1(6): 80-83.

Wolpert, D.H. 1996. The lack of a priori distinctions between learning algorithms. Neural Computation 8(7): 1341-1390.

## Appendix A. Checklist for automated reconciliation projects

The checks below derive directly from the findings of this work and can be applied by engineering and internal audit teams before accepting any result from a reconciliation model.

1. Confirm that no predictive attribute takes part in the rule that generated the label, even indirectly, through fields that only exist when a match exists.
2. Treat accuracy above 0.98 on a reconciliation task as a leakage hypothesis until proven otherwise, and audit the label's provenance before celebrating the result.
3. Run an ablation by progressively removing the attributes suspected of circularity, which sets an upper bound on how much the model actually learned.
4. Replace sentinel values with an explicit missing-value indicator combined with neutral imputation; in this study the change raised logistic regression from 0.8892 to 1.0000 macro-F1 and made the three algorithms equal.
5. Require that no column be constant in the training partition and that training and test have comparable scales, a check that catches artefacts invisible to the leakage guard.
6. Select models by the recall of the exception class under a minimum precision constraint, with the threshold calibrated on a validation partition and applied without readjustment to the test set, and report the rate of referral to manual review next to the primary metric.
7. Report the full confusion matrix, not only aggregate metrics; in this study it was the matrix that revealed the asymmetry between false positives and false negatives, and the breakdown by divergence type that revealed the models' blindness to real exceptions.
8. Validate temporally whenever the data have a temporal dimension, instead of shuffling records in conventional cross-validation.
9. Subject the design to adversarial code review, not only to automated verification, because each instrument only sees the failure mode it was built for.
10. Check reproducibility across separate processes, not within a single one: a test that calls the generator twice in the same interpreter observes neither set iteration order nor the hash seed, and passes under any of them.
11. Anchor every time window to an explicit date, never to the clock, because a dataset generated against the current date changes on every run without showing up in any aggregate metric.
12. Seed distinct sources with distinct seeds: two sources declared independent but derived from the same seed share the sequence and produce impossible coincidences between them.
13. Also compare algorithms on a threshold-free metric, because operating points chosen per algorithm do not fall on the same realised precision and, without it, a gain in recall is indistinguishable from a looser threshold.
14. Require a margin above the floor when calibrating the decision threshold, since maximising recall over thousands of cut points selects the point that crossed the floor by luck, and the floor does not hold out of sample.
