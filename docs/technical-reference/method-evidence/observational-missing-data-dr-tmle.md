# Observational missing-data DR-TMLE

This study validates the composite-indicator construction for an observational missing outcome
and a missing treatment. The [contract](../dr-tmle/theorem.md#observational-missing-data-the-composite-indicator)
gives the construction and its conditions. For arm $a$ the fit targets
$C_a = \Delta_A \Delta 1\{A = a\}$ with the mechanism
$g_{c,a}(W) = P(\Delta_A = 1 \mid W)\,g(a \mid \Delta_A = 1, W)\,\pi(a, W)$.

The study has three scenarios. Each is a law of `tests/studies/mar_arm_indexed_laws.py`, with
every table indexed by the three levels of $W$ and the arms.

| scenario | arms | missing | smallest composite $g_{c,a}$ |
| --- | --- | --- | --- |
| `binary_observational_mar` | 2 | the outcome | 0.12 |
| `binary_mar_outcome_and_treatment` | 2 | the outcome and the treatment | 0.0825 |
| `three_arm_mar_outcome_and_treatment` | 3 | the outcome and the treatment | 0.07 |

The treatment is observational in each scenario. The fit estimates the treatment mechanism and
receives no known probabilities. In the two missing-treatment scenarios the recording of the
treatment depends on the treatment, so $P(A = a \mid W)$ is not identified.

The comparator is R `drtmle` 1.1.2 at pinned commit
[`538a3a2`](https://github.com/benkeser/drtmle/tree/538a3a264c1ca984b6d88978ca7f96165f43152c).
It implements the same construction. An `NA` treatment gives `DeltaA = 0`, and its fluctuation
indicator is `A == a & DeltaA == 1 & DeltaY == 1`. The runner passes the oracle outcome
regressions and the oracle composite mechanism, so `drtmle` skips `estimateG`. `out$drtmle` pairs
with the composite DR-TMLE. `out$tmle` is one logistic fluctuation along $C_a / g_{c,a}$, which on
a binary outcome is the fit the composite TMLE reaches. It pairs with the composite TMLE under
`tmle_ate` and `tmle_ate_mid`.

## Accuracy against known truth

<!-- generated: accuracy -->
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
<!-- /generated -->

## Repeated-sampling properties

<!-- generated: properties -->
<!-- /generated -->

The study declared every budget, seed, law, learner, size, drift and floor before any verdict. The
policy is `reporting`: a red cell stays red at its budget and margin, and the
[red-cell ledger](red-cells.md) records it.

| family | what it tests |
| --- | --- |
| `corrected_mar_inference` | each contrast with both nuisances correct, the outcome regression wrong, the composite mechanism wrong, and both wrong. The outcome drift is the constant regression $\bar Q = 0.5$. The mechanism drift replaces $g(a \mid \Delta_A = 1, W)$ and, where the treatment can be missing, $P(\Delta_A = 1 \mid W)$. The both-wrong control must clear a declared floor of half its large-sample bias |
| `ordinary_targeting` | the composite TMLE, `guard=()`, on the two missing-treatment scenarios |
| `root_n_and_efficiency`, `root_n_rate` | the `ate` of the binary missing-treatment scenario at n = 500, 2,000 and 8,000, and the efficiency ratio against the exact EIF SD |
| `interval_calibration` | the same `ate` against the calibration bands, with a shrunken-SE and a noise control |
| `simultaneous_coverage` | the default band over every reported name of each scenario, and its pointwise joint control |
| `type_i_error`, `power` | the `ate` at a sharp null, and at n = 1,000 |
| `correction_necessity` | the extra scores of every arm after and before the correction cycle, under the outcome drift |
| `treatment_complete_case` | a control that drops the rows with an unrecorded treatment. It must miss the truth |

The power cell runs at n = 1,000, not at the plan's n = 500. At n = 500 the planned power of the
two-sided test is 0.738, and at n = 1,000 it is 0.957. The design of the band cells is in
`tests/unit/test_simultaneous_cell_design.py`: pointwise joint coverage 0.8832, 0.8835 and 0.8220,
and control power 1.0 at 2,400 replications.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | pending | paired replications |
| `n` | pending | observations per primary replication |
| `independent_tests_passed` | pending | truth tests passing |
| `independent_tests_total` | pending | truth tests reported |
| `paired_tests_passed` | pending | paired comparisons passing |
| `paired_tests_total` | pending | paired comparisons reported |
| `property_cells_passed` | pending | property cells passing |
| `property_cells_total` | pending | property cells reported |
| `max_standardized_bias` | pending | largest primary standardized bias |
| `min_coverage` | pending | lowest primary coverage |
| `max_margin_utilization` | pending | largest paired similarity-margin share |
| `margin:confidence_level` | pending | Monte Carlo confidence level |
| `margin:alpha` | pending | nominal test size |
| `margin:nominal_coverage` | pending | nominal interval coverage |
| `margin:standardized_bias` | pending | standardized-bias margin |
| `margin:coverage_floor` | pending | primary coverage floor |
| `margin:se_ratio_sanity_lower` | pending | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | pending | primary SE-ratio upper screen |
| `margin:calibration_coverage_lower` | pending | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | pending | calibration coverage upper bound |
| `margin:type_i_ceiling` | pending | type-I upper bound |
| `margin:minimum_power` | pending | minimum power lower bound |

## Limits

- The study covers a binary outcome, one three-level baseline covariate, and finite-support
  oracle primaries. The reduced regressions are linear and logistic models fitted in sample.
- Partial guards, the bivariate reduction, weights and clusters have no coverage cell of their
  own. The exact laws of `tests/unit/test_composite_missing_data.py` cover the guards, the
  reductions and a weight. The [weighted](weighted-point-treatment-tmle.md) and
  [clustered](clustered-point-treatment-cv-tmle.md) studies cover the weight and cluster algebra,
  which the composite does not change.
- The composite is tilted inside $[10^{-6}, 0.99]$ at the defaults with both indicators, against
  $10^{-4}$ with a missing outcome alone. No scenario reaches that floor.
- No instrument here can detect a violation of the treatment condition, $Y(a)$ independent of
  $\Delta_A$ given $(A, W)$.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_composite/README.md),
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_composite/manifest.json),
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_composite/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_composite/equivalence.csv),
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_composite/properties.csv)
carry the protocol, provenance, and every published row.
