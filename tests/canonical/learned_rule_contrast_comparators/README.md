# Learned-rule contrasts with stochastic regimes and the natural course

This directory will hold the gated repeated-sampling study `T58`. The declaration is committed
before any run; no declared run has been made, and the study is not yet in the evidence register.
The subject is `LearnedRuleContrast` against a known `Stochastic` regime or `NaturalCourse()`,
through its engine form `TMLE(learned_rule=LearnedRule(), interventions=[...], n_folds=10,
targeting_scheme="pooled", g_bounds="auto")` with RM30's interaction logistic outcome learner and
logistic treatment learner. Static comparators and known rules are covered by `T44`.

| part | declaration |
| --- | --- |
| laws | RM30's `non_exceptional` law (calibration) and `misspecified_limit` law (coverage), n = 2,000 |
| fits | two per draw on one fold seed, both reports. Fit C: beside `[Static(0), coin, offer, NaturalCourse()]`, where `coin` treats with probability 0.5 and `offer` with probability 0.5 where `W1 > 0`. Fit N: beside `[NaturalCourse()]` alone |
| parameters | fit C's nine (learned value, `Static(0)`, `coin`, `offer`, the natural course, four contrasts) and fit N's learned value and contrast; fit N's natural course is fit C's number, which the harness checks |
| truths | per replication for the learned value (each fold rule refit, rule L4); fixed for each comparator by the trapezoid rule on 4,001 points of `W1`. `non_exceptional`: `Static(0)` 0.51218, `coin` 0.52190, `offer` 0.54434, `E Y` 0.52657. `misspecified_limit`: `coin` 0.51151, `offer` 0.53340, `E Y` 0.51558 |
| primary | 1,600 replications of `non_exceptional`, fold-evaluated, all eleven parameters (bias leg, coverage floor, SE sanity) |
| `interval_calibration` | eight tags per report (`coin`, `offer`, the natural course, the four fit-C contrasts, fit N's contrast), each with the shrunken-SE control (0.80) and the noise control (one efficiency-bound unit). 3,600 draws, raised before the run if the pilot's arm-pooled rule asks for more |
| `conservative_coverage` | `misspecified_limit`, 4,000 draws, each report's five contrasts: bias leg and Clopper-Pearson lower end at or above 0.92; control SE x 0.60 below 0.92. The cells measure coverage only. The natural-course contrast is predicted at a reported over true SE of about 0.998 on this law |
| `contrast_covariance` | each contrast, both laws and reports, with the correlation-one SE; coverage below 0.80 |
| `independent_se_diagnostic` | each contrast with the independence SE; no verdict |
| `targeting_necessity` | fit C with `drop_w1_learner`, contrast against `offer`, 1,200 draws; the untargeted plug-in's limit bias is 0.0448, about 2.6 SEs |
| comparator | none: no external package fits these contrasts |

## The support rows

Declared before any run (`SUPPORT_ROWS` and `ROW_TAGS` in
`tests/studies/learned_rule_contrast_comparators.py`): per report, `T58.combined`
(`known_regimes`, `known_stochastic_regimes`, `natural_course`), `T58.stochastic`
(`known_regimes`, `known_stochastic_regimes`) and `T58.natural_course` (`known_regimes`,
`natural_course`). A row is added only if every verdict it lists passes as declared. A red cell
with a defect keeps its row out and its labels experimental. A red calibration cell whose SE-ratio
interval contains 1, whose coverage is consistent with 0.95 and whose diagnosis finds no defect is
a finite-sample red: the row goes to `reporting` with an owner row.

## Running

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m tests.studies.learned_rule_contrast_comparators_pilot --raw <scratch>/pilot --jobs <cores>
    python -m tests.studies.learned_rule_contrast_comparators_pilot --raw <scratch>/pilot --summarize
    python -m tests.canonical.learned_rule_contrast_comparators.regenerate --output <scratch dir>

A smoke run adds `--replicates <k> --allow-failures --property-harness-check <k>`.
`tests/unit/test_learned_rule_contrast_comparators_design.py` recomputes every declared number and
reads the pilot file once it exists.
