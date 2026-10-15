# Vector modified treatment policies

This study validates modified treatment policies on a vector treatment. A vector holds a binary
flag and one or two doses. The point `TMLE` fits it with `policies=`, and `LTMLE` fits it as a
node. Both read the ratio from the joint density that a `JointPolicy` induces
(`ratio="density"`) or from a stacked classifier (`ratio="classifier"`). The
[vector treatments](../point-treatment-tmle.md#vector-treatments) and
[vector nodes](../longitudinal-tmle.md#vector-nodes-with-a-continuous-component) sections state
the estimators, their conditions and their exact-law evidence.

**Status of this page.** The study is registered and declared. It has not run, and the generated
tables below hold the word `PENDING` where the committed results will write a table or a number.
The evidence label is "Python property evidence, with comparator runs pending". The evidence has
two parts.

| part | what it is | status |
| --- | --- | --- |
| Python property evidence | exact laws with closed-form efficiency bounds, interval calibration with two controls, double robustness in four nuisance configurations, and three necessity pairs | declared. It decides whether the method ships. The study commit fills the tables |
| `lmtp` comparator runs | the pinned `lmtp` 1.5.4 fits the same samples with vector `trt` nodes, and the study pairs the results | pending. The study record sets `pending_reference="lmtp"`, so no paired row is published. The runs register in a later change, and a finding in them does not block the method. The fast-tier witnesses and the Python property cells carry the claim until then |

The publication policy is `reporting`. A red cell stays red on this page and in the
[red-cell ledger](red-cells.md), with the owner row that names it.

## What was compared

The four scenarios draw 1,000 samples of 2,000 rows each.

| scenario | law and plans | pairing |
| --- | --- | --- |
| `point_vector_mixed` | the point law, `TMLE`, treatment $(F, D)$ with a binary flag and a dose. The plans are `natural course`, `dose up` and `flag scaled`. The ratio is the oracle induced density, which `lmtp` receives with both shifted components | margin-read: `TMLE` and `lmtp` target by different constructions |
| `point_vector_categorical` | the categorical law, `TMLE`, treatment $(A_1, A_2)$ with two binary components. The plans are `natural course`, `a2 off` and `a1 on`. The ratio is the discrete formula on the exact joint probabilities | margin-read |
| `longitudinal_vector_mixed` | the two-node law, `LTMLE` at one fold, nodes $(F_1, D_1)$ and $(F_2, D_2)$. The plans are the point plans at both nodes. Each node's ratio (`node_ratio`) and shifted components go to `lmtp` | exact under a gap of $10^{-6}$ |
| `longitudinal_vector_ordered_classifier` | the ordered law as one `LTMLE` node $(F, D_1, D_2)$. The plans are `natural course`, `ordered` and `full`. Each side fits its own stacked classifier | the parity rule below |

| setting | `cleverly` | R `lmtp` |
| --- | --- | --- |
| datasets | 1,000 samples of 2,000 rows per scenario | the identical rows |
| density ratio | an oracle pooled-hazard density with `ceil(320 (n / 2000)^(2/3))` bins (320 at n = 2,000), and the exact joint probabilities on the categorical law | the same ratio array, read from the replicate data |
| outcome regressions | correctly specified logistic models: main terms plus the squared doses, or the saturated joint levels on the categorical law | `SL.glm` on the same columns |
| classifier of the ordered scenario | an unpenalized logistic regression of the stack label on one feature, the plan's true log ratio, which the learner computes from each stacked row | the R twin `SL.logratio.<plan>`, with `mtp = TRUE` |
| intervals | pointwise 95% Wald from the influence curve | the same, from `lmtp`'s influence curve |

The comparator adapter is `tests/canonical/lmtp_vector_adapter.R`. It builds an `LmtpTask` whose
treatment nodes are vectors, such as `trt = list(c("F", "D"))`. With `mtp = FALSE`, `lmtp` 1.5.4
multiplies each ratio by the indicator that no component moved (`R/density_ratios.R`, lines 58 to
61). That indicator is zero on every row that a plan moves, so the adapter sets `mtp = TRUE`.

The parity check requires that the share of positive `lmtp` ratios equal the package's share. Under
`mtp = TRUE` both shares are 1, because neither classifier predicts 0. Under `mtp = FALSE` the
`lmtp` share falls to the share of unmoved rows, and the check fails.

The ordered pair reads by three outcomes, declared before any run. Every ratio gap below $10^{-6}$
makes the pair an exactness witness. Every relative gap below $10^{-4}$, with a diagnosed and
named numerical cause, makes it a margin-read pair. Any other reading blocks the declaration until
the cause is found.

## Accuracy against known truth

<!-- generated: accuracy -->
| status |
| --- |
| PENDING: the study has not run. `python -m tests.studies.evidence.document --slug vector-mtp` writes this table from the committed results |
<!-- /generated -->

## Agreement with the canonical implementation

Python property evidence, with comparator runs pending. The `lmtp` runs have not run, so this
study publishes no paired row, and `equivalence.csv` is empty. The change that registers the runs
adds the paired table here.

## Theory properties

<!-- generated: properties -->
| status |
| --- |
| PENDING: the property phase has not run. The same command writes this table |
<!-- /generated -->

The truths and the efficiency bounds come from the exact laws and their closed forms
(`tests/studies/vector_mtp_common.py`). The quadrature is checked at $10^{-10}$.

| family | cells | replications and size |
| --- | --- | --- |
| `interval_calibration` | six positives: point `TMLE`, two-node `LTMLE` and the ordered law as one `LTMLE` node, each on the density route and on the classifier route. Each positive has a shrunken-SE control and a noise control | 24,000, 9,600 and 19,200 replications on the density routes, and 10,000 on each classifier route, at n = 2,000. The design test sizes them |
| `double_robustness` | point `TMLE` on the ordered law under `full`, two-node `LTMLE` under `flag scaled`, and the ordered law as one `LTMLE` node under `full`. Each runs `both_correct`, `outcome_correct`, `outcome_wrong` and the `both_wrong` control | 2,000 replications at n = 2,000 |
| `preimage_necessity` | two-node `LTMLE` under `flag scaled`, wrong outcome regression. The declared fit against the same draw with the preimage sum cut to the observed categorical cell | 1,000 pairs at n = 2,000 |
| `jacobian_necessity` | the ordered law under `full`, wrong outcome regression. The declared fit against the same draw with the Jacobian of the `D1` scale dropped. That scale is the only map whose Jacobian is not 1 | 1,000 pairs at n = 2,000 |
| `image_necessity` | the ordered law under `crossed`. The declared fit against the same draw with the flag map read at the observed second value of $D_1$ instead of the natural value of the preimage | 1,000 pairs at n = 2,000 |

Each necessity control removes one term that can vanish at the truth, so each needs a nonzero
witness. With a wrong outcome regression the targeting step carries the whole correction, and each
defect then moves the estimate. The `crossed` plan exists because the `full` plan is blind to the
flag map's read of $D_1$. The exact-law shift between the declared and the defective map is 0.000
under `full` and 1.243 SD under `crossed`
(`tests/unit/test_vector_mtp_design.py`). A control passes only when its pair moves by more than
the declared displacement.

The design test sizes every replication count before the run. It requires the product of the pass
probabilities of the six calibration positives and their controls to reach 0.90. The product holds
with every probed number moved one Monte Carlo standard error toward failing.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed. A value of `PENDING` stands for a
number that the study commit writes.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | paired replications |
| `n` | 2000 | observations per paired replication |
| `independent_tests_passed` | PENDING | truth tests passing |
| `independent_tests_total` | PENDING | truth tests reported |
| `paired_tests_passed` | PENDING | paired comparisons passing |
| `paired_tests_total` | PENDING | paired comparisons reported |
| `property_cells_passed` | PENDING | property cells passing |
| `property_cells_total` | PENDING | property cells reported |
| `max_standardized_bias` | PENDING | largest primary standardized bias |
| `min_coverage` | PENDING | lowest primary coverage |
| `max_margin_utilization` | PENDING | largest paired similarity-margin share |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound |
| `margin:minimum_power` | 0.8000 | power lower bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:necessity_displacement` | 0.1000 | minimum Jacobian and image necessity displacement |
| `margin:preimage_displacement` | 0.1000 | minimum preimage necessity displacement |

## Limits

| limit | what it means for use |
| --- | --- |
| three laws with a binary flag and at most two doses, at n = 2,000 | the evidence covers these laws and this size. The study has no size ladder, so it claims no rate |
| the density is an oracle | the density route reads the generating pooled-hazard probabilities at a bin count that grows as $n^{2/3}$. The default estimated density is not tested here |
| the classifier is a one-feature oracle | the classifier route reads the correctly specified logistic model of the plan's log ratio. The study says nothing about the rate or the interval of a learned classifier |
| the density of $D_2$ and the map of $D_2$ may read the shifted $D_1$ without a failing cell | those defects move the outcome-wrong positives by 0.42 and -0.19 SD only. The fast-tier mutation witnesses in `tests/unit/test_vector_density_mutations.py` carry them |
| the point pairs compare two constructions | `TMLE` targets with the covariate submodel over the observed and the policy values, and `lmtp` with a ratio-weighted intercept. The pairs read under the default margins and are not exactness checks |
| the `lmtp` comparison is pending | the agreement table is empty until the reference runs register. The Python cells carry the claim until then |
| a policy must be known and fixed | a policy learned from the sample, or one that depends on the law, is outside this evidence |
