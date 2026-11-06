# Learned treatment rules with an intermediate held at each level

This study validates the interval of the cross-fitted learned-rule value when a binary
intermediate $Z$ is held at each of its levels. The estimator is
`TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=True, targeting_scheme="pooled",
g_bounds="auto")` fitted with `intermediate="Z"`, which reports one result per level $z$. At each
level every fold learns the plug-in rule $d^z_v(w) = \arg\max_a \bar Q_v(a, z, w)$ of its outcome
regression on its training rows, and one pooled fluctuation targets every fold with the clever
covariate $\mathbb 1\{A = d^z_v(W)\}\,\mathbb 1\{Z = z\} / (g_v(A \mid W)\, q_v(z \mid A, W))$.
[An intermediate held at each level](../point-treatment-tmle.md#learned-rules) gives the
construction and its conditions. The study also fits the known rule $\mathbb 1\{W_1 > 0\}$ through
the same fold-evaluated controlled-direct-effect path, the path that witness W-j1 reduces the
learned fit to bit for bit.

The target of a learned level is data-adaptive:
$\tilde\psi^z_{0n} = V^{-1}\sum_v E_0\,\bar Q_0(d^z_v(W), z, W)$, the value at that level of the
rules its folds learned. That is the best treatment given $Z = z$, not the best pair of treatment
and intermediate. Each replication carries its own truth, and every statistic reads the error, the
estimate minus that truth.
[A truth that varies by replication](../../development/method-benchmarking.md#a-truth-that-varies-by-replication)
gives the rule. The known rule's target is its fixed value at each level.

**No canonical implementation is compared.** No maintained package fits a learned rule with an
intermediate held fixed. A zero-row equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| law | $W_1 \sim U(-1, 1)$, $W_2 \sim \operatorname{Bernoulli}(0.5)$, $\operatorname{logit} g_0(1 \mid W) = 0.3 W_1 - 0.2 W_2$, $\operatorname{logit} P(Z = 1 \mid A, W) = 0.2 + 0.6 A - 1.2 W_1$ and $\operatorname{logit} \bar Q_0(a, z, W) = 0.2 + 0.5 W_1 - 0.3 W_2 + 0.3 z + a\,(0.1 + W_1 - 0.6 z)$ |
| optimal rules | $\mathbb 1\{W_1 > -0.1\}$ at $z = 0$ (55% treated) and $\mathbb 1\{W_1 > 0.5\}$ at $z = 1$ (25%); they differ on 30% of the population. The smallest $g_0 q_0$ on the support is 0.045 |
| fit | `TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto")` with `intermediate="Z"`, `repeats=1` and pointwise intervals; the known cell passes `interventions=[Rule(1{W1 > 0})]` in place of the learned rule |
| outcome learner | a logistic regression on every pairwise product of $(A, W_1, W_2, Z)$, a correct model of the law; the necessity learner leaves out $W_1$ |
| treatment and intermediate learners | `LogisticRegression(C=1e6, max_iter=1000)` on $(W_1, W_2)$ and on $(A, W_1, W_2)$, correct models, fitted in every cell |
| support keys | splitting `("fold_evaluated",)` and missing `("intermediate",)`, family `P11` for the learned rule and `P5` for the known rule, pinned by `tests/unit/test_learned_rule_intermediate_harness.py` |
| truth | per replication and level: refit each fold's outcome learner on the fit's training rows with the package's `fit_on_rows`, require its rule to equal the fit's on every validation row and the stored fold rule on every row, and integrate $\bar Q_0(d_v(W), z, W)$ by the trapezoid rule on 4,001 points of $W_1$ for each $W_2$ |
| stored variance | $V^{-2}\sum_v n_v^{-2}\sum_{i\in v} D_i^2$ of the raw fold curves, uncentred within each fold. The harness rebuilds each level's curve longhand and stops the run if its variance differs from the reported one by more than a relative $10^{-8}$ |
| primary budget | 1,200 replications at n = 2,000; one fit writes both levels' rows (`cde_z0` and `cde_z1`) |
| primary verdicts | the framework defaults at each level: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| seeds | `seed=20271311` and `resampling_seed=20271312` |
| policy | `gated` |

The law, the learners, the budgets, the seeds and the support-row conditions were fixed before any
run, in the X11 plan, its implementation review and the study design record with its review.
`tests/unit/test_learned_rule_intermediate_design.py` pins the law's values by quadrature, the
sizing and the seeds. `tests/unit/test_learned_rule_intermediate_harness.py` checks the fits, the
keys, the truth, the rule refit, the exactness gate and the property path.
`tests/unit/test_learned_rule_intermediate.py` holds the unit witnesses of the construction.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| learned-rule law with a binary intermediate Z after A, correct logistic outcome, treatment and intermediate learners, the level Z = 0, fold-evaluated report | `ey_learned_rule[learned rule]` | at each level z, the fold average of the values E Qbar0(d_v(W), z, W) of the rules learned at that level on each training fold | `cleverly` cross-validated TMLE of the learned-rule value with the intermediate held at each level: each fold's rule at level z the plug-in rule of its outcome model at (a, z), one pooled fluctuation and the fold-evaluated report | -0.0027 to 0.0011 | 0.9458 | 1.0022 | pass |
| learned-rule law with a binary intermediate Z after A, correct logistic outcome, treatment and intermediate learners, the level Z = 1, fold-evaluated report | `ey_learned_rule[learned rule]` | at each level z, the fold average of the values E Qbar0(d_v(W), z, W) of the rules learned at that level on each training fold | `cleverly` cross-validated TMLE of the learned-rule value with the intermediate held at each level: each fold's rule at level z the plug-in rule of its outcome model at (a, z), one pooled fluctuation and the fold-evaluated report | -0.0018 to 0.0013 | 0.9308 | 0.9769 | pass |
<!-- /generated -->

## Theory properties

Every property cell fits n = 2,000 on the one law.

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | at each level, does the SE ratio of the learned fit (20,000 replications) and of the known-rule fit (10,000 replications) lie inside 0.93 to 1.07, and its coverage inside 0.92 to 0.98 | a shrunken-SE control multiplies each SE by 0.80, a noise control adds noise of one asymptotic SD over $\sqrt{n}$ to each estimate, and a term-dropped control rebuilds the SE from the curve without the factor $\mathbb 1\{Z = z\}/q$. Each control's SE-ratio interval must fall below the band |
| `targeting_necessity` | at each level, does the fluctuation remove the plug-in's bias with an outcome learner that leaves out $W_1$, with 1,200 replications | the control is the same fit's untargeted fold average of the initial plug-ins. The positive's bias interval must lie inside the margin, the control's outside it, and the smaller displacement over the two levels must be at least 0.5 positive-arm SDs |

The noise scale is the SD of $\sqrt{n}$ times the estimate at each level's optimal rule, or at the
known rule, by quadrature, and no verdict reads it as an efficiency bound. The learned and known
counts follow the per-cell predictive rule and a probe declared before it ran; the design record
gives the sizing. The necessity learner's untargeted plug-in is biased at its limit by the
confounding through $W_1$: by quadrature its limit rule treats everyone at $z = 0$ and nobody at
$z = 1$, and the plug-in sits 0.098 above and 0.025 below that rule's value.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `known_z0__declared` | positive | the intermediate law with a correct interaction logistic outcome learner on (A, W1, W2, Z), the known rule 1{W1 > 0} at Z = 0, n = 2,000: the declared fit of the known rule 1{W1 > 0} with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9460 to 0.9572, SE ratio 0.9902 to 1.0251 | pass |
| `interval_calibration` | `known_z0__noise_control` | control | the intermediate law with a correct interaction logistic outcome learner on (A, W1, W2, Z), the known rule 1{W1 > 0} at Z = 0, n = 2,000: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8217 to 0.8410, SE ratio 0.6900 to 0.7161 | pass |
| `interval_calibration` | `known_z0__shrunken_se_control` | control | the intermediate law with a correct interaction logistic outcome learner on (A, W1, W2, Z), the known rule 1{W1 > 0} at Z = 0, n = 2,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8769 to 0.8934, SE ratio 0.7912 to 0.8201 | pass |
| `interval_calibration` | `known_z0__term_dropped_control` | control | the intermediate law with a correct interaction logistic outcome learner on (A, W1, W2, Z), the known rule 1{W1 > 0} at Z = 0, n = 2,000: the declared fit's estimate with the standard error of its curve rebuilt without the factor 1{Z = z} / q of the clever covariate | the SE-ratio interval must fall below the calibration band | coverage 0.7509 to 0.7729, SE ratio 0.5961 to 0.6173 | pass |
| `interval_calibration` | `known_z1__declared` | positive | the intermediate law with a correct interaction logistic outcome learner on (A, W1, W2, Z), the known rule 1{W1 > 0} at Z = 1, n = 2,000: the declared fit of the known rule 1{W1 > 0} with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9425 to 0.9540, SE ratio 0.9803 to 1.0154 | pass |
| `interval_calibration` | `known_z1__noise_control` | control | the intermediate law with a correct interaction logistic outcome learner on (A, W1, W2, Z), the known rule 1{W1 > 0} at Z = 1, n = 2,000: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8236 to 0.8429, SE ratio 0.6946 to 0.7205 | pass |
| `interval_calibration` | `known_z1__shrunken_se_control` | control | the intermediate law with a correct interaction logistic outcome learner on (A, W1, W2, Z), the known rule 1{W1 > 0} at Z = 1, n = 2,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8725 to 0.8893, SE ratio 0.7843 to 0.8131 | pass |
| `interval_calibration` | `known_z1__term_dropped_control` | control | the intermediate law with a correct interaction logistic outcome learner on (A, W1, W2, Z), the known rule 1{W1 > 0} at Z = 1, n = 2,000: the declared fit's estimate with the standard error of its curve rebuilt without the factor 1{Z = z} / q of the clever covariate | the SE-ratio interval must fall below the calibration band | coverage 0.8599 to 0.8774, SE ratio 0.7569 to 0.7845 | pass |
| `interval_calibration` | `learned_z0__declared` | positive | the intermediate law with a correct interaction logistic outcome learner on (A, W1, W2, Z), the learned rule at Z = 0, n = 2,000: the declared fit of the learned rule with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9445 to 0.9526, SE ratio 0.9849 to 1.0104 | pass |
| `interval_calibration` | `learned_z0__noise_control` | control | the intermediate law with a correct interaction logistic outcome learner on (A, W1, W2, Z), the learned rule at Z = 0, n = 2,000: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8328 to 0.8462, SE ratio 0.7090 to 0.7280 | pass |
| `interval_calibration` | `learned_z0__shrunken_se_control` | control | the intermediate law with a correct interaction logistic outcome learner on (A, W1, W2, Z), the learned rule at Z = 0, n = 2,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8748 to 0.8866, SE ratio 0.7872 to 0.8086 | pass |
| `interval_calibration` | `learned_z0__term_dropped_control` | control | the intermediate law with a correct interaction logistic outcome learner on (A, W1, W2, Z), the learned rule at Z = 0, n = 2,000: the declared fit's estimate with the standard error of its curve rebuilt without the factor 1{Z = z} / q of the clever covariate | the SE-ratio interval must fall below the calibration band | coverage 0.7436 to 0.7594, SE ratio 0.5832 to 0.5984 | pass |
| `interval_calibration` | `learned_z1__declared` | positive | the intermediate law with a correct interaction logistic outcome learner on (A, W1, W2, Z), the learned rule at Z = 1, n = 2,000: the declared fit of the learned rule with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9416 to 0.9499, SE ratio 0.9706 to 0.9962 | pass |
| `interval_calibration` | `learned_z1__noise_control` | control | the intermediate law with a correct interaction logistic outcome learner on (A, W1, W2, Z), the learned rule at Z = 1, n = 2,000: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8266 to 0.8402, SE ratio 0.6960 to 0.7144 | pass |
| `interval_calibration` | `learned_z1__shrunken_se_control` | control | the intermediate law with a correct interaction logistic outcome learner on (A, W1, W2, Z), the learned rule at Z = 1, n = 2,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8711 to 0.8831, SE ratio 0.7765 to 0.7969 | pass |
| `interval_calibration` | `learned_z1__term_dropped_control` | control | the intermediate law with a correct interaction logistic outcome learner on (A, W1, W2, Z), the learned rule at Z = 1, n = 2,000: the declared fit's estimate with the standard error of its curve rebuilt without the factor 1{Z = z} / q of the clever covariate | the SE-ratio interval must fall below the calibration band | coverage 0.8428 to 0.8559, SE ratio 0.7212 to 0.7399 | pass |
| `targeting_necessity` | `drop_w1_z0__targeted` | positive | the intermediate law with an outcome learner that has no W1 term, the learned rule at Z = 0, n = 2,000: the declared fit with that learner, so targeting does the adjusting | bias interval inside the equivalence margin | bias -0.0021 to 0.0027, margin 0.0080 | pass |
| `targeting_necessity` | `drop_w1_z0__untargeted` | control | the intermediate law with an outcome learner that has no W1 term, the learned rule at Z = 0, n = 2,000: the same fit's untargeted fold average of the initial plug-ins at its fold rules, with no fluctuation | bias interval must fall entirely outside the margin | bias 0.0973 to 0.1012, margin 0.0065 | pass |
| `targeting_necessity` | `drop_w1_z1__targeted` | positive | the intermediate law with an outcome learner that has no W1 term, the learned rule at Z = 1, n = 2,000: the declared fit with that learner, so targeting does the adjusting | bias interval inside the equivalence margin | bias -0.0012 to 0.0020, margin 0.0055 | pass |
| `targeting_necessity` | `drop_w1_z1__untargeted` | control | the intermediate law with an outcome learner that has no W1 term, the learned rule at Z = 1, n = 2,000: the same fit's untargeted fold average of the initial plug-ins at its fold rules, with no fluctuation | bias interval must fall entirely outside the margin | bias -0.0256 to -0.0226, margin 0.0051 | pass |
<!-- /generated -->

## Result

Every primary test and every gated property cell passed, and no fit failed. At $z = 0$ and
$z = 1$ the primary tests read coverage of 0.946 and 0.931 and SE ratios of 1.002 and 0.977 over
1,200 draws. The learned calibration positives read SE ratios of 0.998 and 0.983 over 20,000
draws, and the known-rule positives 1.007 and 0.997 over 10,000 draws, every interval inside its
band. Each control fails as declared.

| reading | value | source |
| --- | --- | --- |
| primary tests passed | 2 of 2 | `performance-tests.csv` |
| gated property cells passed | 20 of 20 | `properties.csv` |
| failed fits | 0 | `manifest.json`, `properties.csv` and the run log |
| learned SE-ratio interval, $z = 0$ | 0.9849 to 1.0104 | `properties.csv`, `learned_z0__declared` |
| learned SE-ratio interval, $z = 1$ | 0.9706 to 0.9962 | `properties.csv`, `learned_z1__declared` |
| known-rule SE-ratio interval, $z = 0$ | 0.9902 to 1.0251 | `properties.csv`, `known_z0__declared` |
| known-rule SE-ratio interval, $z = 1$ | 0.9802 to 1.0154 | `properties.csv`, `known_z1__declared` |

The support rows were declared before the run. The learned row needed the primary bias leg,
coverage floor and SE sanity at both levels, every learned `interval_calibration` verdict and the
`targeting_necessity` verdicts to pass. The known-rule row needed every known-rule
`interval_calibration` verdict to pass. They did, so the registered row `T47` covers the plug-in
rule with an intermediate held at each level, and `T48` a known regime through the same path,
both under the fold-evaluated report on rows. The stacked report, the in-sample fit, fold
targeting, several splits, weights, clusters, more than two arms, known regimes beside the rule
and a known $g$ stay experimental.

### The SE ratio at each level

The learned positive at $z = 1$ reads 0.983, with a 99% interval of 0.971 to 0.996 that sits
below 1 and inside the band. Plug-in learned rules read slightly below 1 at n = 2,000, the
between-fold covariance of order $1/n$ that the T39 limits describe. The known-rule cells, whose
rule does not move with the folds, read 1.007 and 0.997 with the same $g$, $q$ and clever covariate.

### The controls

The curve without $\mathbb 1\{Z = z\}/q$ reads SE ratios of 0.591 and 0.731 at the learned levels
and 0.607 and 0.770 at the known-rule levels, close to the population ratios of that curve, so the
instrument resolves a variance that misses the factor. With the outcome learner that leaves out
$W_1$, the untargeted plug-in is biased by +0.099 at $z = 0$ and -0.024 at $z = 1$, 3.82 and
-1.18 of its own SDs, and the targeted fit's bias interval sits inside the margin at both levels.
Its limit rules are constant, so the pair tests a gross misspecification of the outcome model,
not a subtle one.

No primary or property draw raised a solver warning, and the rule refit found the fit's rule on
every row of every draw.

## Measured values

Names beginning `margin:` are thresholds declared before the run. A name beginning `bound:` is the
SD of $\sqrt{n}$ times the estimate at a level's optimal rule, or at the known rule, over
$\sqrt{2000}$. It sizes the noise control only, and no verdict reads it as an efficiency bound.
Everything else is measured from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1200 | primary replications |
| `n` | 2000 | observations per replication |
| `independent_tests_total` | 2 | parameter tests against the truth |
| `independent_tests_passed` | 2 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 20 | gated repeated-sampling property cells |
| `property_cells_passed` | 20 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0300 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9308 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9098 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9263 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0595 | highest bootstrap primary SE-ratio endpoint |
| `performance[cleverly-learned-rule-intermediate/cde_z0/ey_learned_rule[learned rule]]:coverage` | 0.9458 | primary coverage at $z = 0$ |
| `performance[cleverly-learned-rule-intermediate/cde_z0/ey_learned_rule[learned rule]]:se_ratio` | 1.0022 | primary SE ratio at $z = 0$ |
| `performance[cleverly-learned-rule-intermediate/cde_z1/ey_learned_rule[learned rule]]:coverage` | 0.9308 | primary coverage at $z = 1$ |
| `performance[cleverly-learned-rule-intermediate/cde_z1/ey_learned_rule[learned rule]]:se_ratio` | 0.9769 | primary SE ratio at $z = 1$ |
| `mean_oracle_se_cde_z0` | 0.0253 | mean oracle SE at the primary fits' fold rules, $z = 0$ |
| `mean_oracle_se_cde_z1` | 0.0199 | mean oracle SE at the primary fits' fold rules, $z = 1$ |
| `properties[interval_calibration/learned_z0__declared]:se_ratio` | 0.9975 | learned calibration SE ratio, $z = 0$ |
| `properties[interval_calibration/learned_z0__declared]:se_ratio_ci_lower` | 0.9849 | its interval, lower end |
| `properties[interval_calibration/learned_z0__declared]:se_ratio_ci_upper` | 1.0104 | its interval, upper end |
| `properties[interval_calibration/learned_z0__declared]:coverage` | 0.9486 | learned calibration coverage, $z = 0$ |
| `properties[interval_calibration/learned_z1__declared]:se_ratio` | 0.9833 | learned calibration SE ratio, $z = 1$ |
| `properties[interval_calibration/learned_z1__declared]:se_ratio_ci_lower` | 0.9706 | its interval, lower end |
| `properties[interval_calibration/learned_z1__declared]:se_ratio_ci_upper` | 0.9962 | its interval, upper end |
| `properties[interval_calibration/learned_z1__declared]:coverage` | 0.9458 | learned calibration coverage, $z = 1$ |
| `properties[interval_calibration/known_z0__declared]:se_ratio` | 1.0070 | known-rule calibration SE ratio, $z = 0$ |
| `properties[interval_calibration/known_z0__declared]:se_ratio_ci_lower` | 0.9902 | its interval, lower end |
| `properties[interval_calibration/known_z0__declared]:se_ratio_ci_upper` | 1.0251 | its interval, upper end |
| `properties[interval_calibration/known_z0__declared]:coverage` | 0.9518 | known-rule calibration coverage, $z = 0$ |
| `properties[interval_calibration/known_z1__declared]:se_ratio` | 0.9975 | known-rule calibration SE ratio, $z = 1$ |
| `properties[interval_calibration/known_z1__declared]:se_ratio_ci_lower` | 0.9803 | its interval, lower end |
| `properties[interval_calibration/known_z1__declared]:se_ratio_ci_upper` | 1.0154 | its interval, upper end |
| `properties[interval_calibration/known_z1__declared]:coverage` | 0.9485 | known-rule calibration coverage, $z = 1$ |
| `properties[interval_calibration/learned_z0__term_dropped_control]:se_ratio` | 0.5907 | term-dropped control SE ratio, learned $z = 0$ |
| `properties[interval_calibration/learned_z1__term_dropped_control]:se_ratio` | 0.7306 | term-dropped control SE ratio, learned $z = 1$ |
| `properties[interval_calibration/known_z0__term_dropped_control]:se_ratio` | 0.6066 | term-dropped control SE ratio, known $z = 0$ |
| `properties[interval_calibration/known_z1__term_dropped_control]:se_ratio` | 0.7704 | term-dropped control SE ratio, known $z = 1$ |
| `properties[interval_calibration/learned_z1__term_dropped_control]:se_ratio_ci_upper` | 0.7399 | the largest term-dropped upper endpoint, against 0.93 |
| `properties[interval_calibration/known_z0__shrunken_se_control]:se_ratio_ci_upper` | 0.8201 | the largest shrunken-SE upper endpoint, against 0.93 |
| `properties[interval_calibration/known_z1__noise_control]:se_ratio_ci_upper` | 0.7205 | the largest noise-control upper endpoint, against 0.93 |
| `properties[targeting_necessity/drop_w1_z0__untargeted]:bias` | 0.0993 | untargeted plug-in bias, $z = 0$ |
| `properties[targeting_necessity/drop_w1_z0__untargeted]:standardized_bias` | 3.8221 | in its own SDs |
| `properties[targeting_necessity/drop_w1_z1__untargeted]:bias` | -0.0241 | untargeted plug-in bias, $z = 1$ |
| `properties[targeting_necessity/drop_w1_z1__untargeted]:standardized_bias` | -1.1836 | in its own SDs |
| `properties[targeting_necessity/drop_w1_z0__targeted]:standardized_bias` | 0.0093 | targeted standardized bias, $z = 0$ |
| `properties[targeting_necessity/drop_w1_z1__targeted]:standardized_bias` | 0.0193 | targeted standardized bias, $z = 1$ |
| `properties[targeting_necessity/drop_w1_z0__targeted]:targeting_displacement` | 1.1207 | the smaller displacement over the two levels, in positive-arm SDs |
| `bound:learned_z0_standard_error` | 0.0251 | the SD at the optimal rule at $z = 0$ over the root of n |
| `bound:learned_z1_standard_error` | 0.0197 | the SD at the optimal rule at $z = 1$ over the root of n |
| `bound:known_z0_standard_error` | 0.0247 | the SD at the known rule at $z = 0$ over the root of n |
| `bound:known_z1_standard_error` | 0.0191 | the SD at the known rule at $z = 1$ over the root of n |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in error SDs |
| `margin:coverage_floor` | 0.9000 | validity floor the primary coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:shrunken_se_factor` | 0.8000 | factor the shrunken-SE control applies to each SE |
| `margin:targeting_displacement` | 0.5000 | smallest displacement the necessity pair must show |
| `margin:type_i_ceiling` | 0.1000 | largest size a one-sided type-I bound may establish |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical SDs |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power control must clear |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:subject_failure_bound` | 0.0100 | the failed-fit bound of the committed run |
| `margin:subject_failures_allowed` | 12 | failed primary fits that bound allows |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | No maintained package fits a learned rule with an intermediate held fixed. The rows rest on accuracy against the truth and on the theory properties |
| One law, one size and one learner per role | The study fits n = 2,000 on one law with a binary treatment, outcome and intermediate, ten equal folds, pooled targeting, `repeats=1`, iid rows, and no weights, clusters, strata, fold targeting, stacked report or contrasts. The outcome, treatment and intermediate models are correctly specified parametric logistic models, fitted in every cell. The key holds neither the learner, the law nor n, so a flexible learner, another law or another n reads supported and rests on the theory alone |
| Each level targets its own learned rule | Each learned level's interval covers $V^{-1}\sum_v E_0\,\bar Q_0(d^z_v(W), z, W)$, the value of that level's own fold rules. That is neither the value of the optimal rule at that level nor the jointly optimal pair of treatment and intermediate |
| Positivity of the intermediate | The smallest $g_0 q_0$ on the support is 0.045, and along the learned rules it is 0.11 at $z = 0$ and 0.19 at $z = 1$. A law with $q_0(z \mid a, W)$ nearer 0 was not measured |
| The SE ratio of the learned levels | Each learned level's SE ratio is quoted with its interval: 0.998 (0.985 to 1.010) at $z = 0$ and 0.983 (0.971 to 0.996) at $z = 1$. Plug-in learned rules read slightly below 1 at n = 2,000 |
| One known rule, and its value only | Row `T48` was measured at the rule $\mathbb 1\{W_1 > 0\}$ and for its value. Its key also covers static regimes, contrasts between known regimes and any deterministic rule with a binary intermediate on the fold-evaluated path, and none of those was measured. T30, in sample with missing outcomes, is a different configuration |
| A gross necessity misspecification | The necessity learner's limit rules treat everyone at $z = 0$ and nobody at $z = 1$, so the pair shows that targeting removes a large confounding bias, not a subtle one |
| Compositions outside the rows | A rule learner with `intermediate=` refuses (X11 (j2)), and so do a non-binary intermediate (X63), strata under `cv_evaluation=True` (X28), the omitted-variable bound (X11 (n2)) and the placebo of a level in `refute`. Weights, clusters, fold targeting, several splits, the stacked and in-sample reports, three or more arms and a known $g$ read experimental |
| Pointwise intervals | The study reads the pointwise interval of each level's value |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_intermediate/README.md)
gives the run command and the run form. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_intermediate/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_intermediate/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_intermediate/replicates.parquet),
the [truth harness](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_intermediate/harness.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_intermediate/properties.csv)
carry every published row.
