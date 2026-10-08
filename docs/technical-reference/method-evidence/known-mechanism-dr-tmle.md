# DR-TMLE on a declared known treatment mechanism

This study validates complete-data `DRTMLE` on data that declares its treatment mechanism. The
fit is Benkeser et al. (2017), Theorem 1, at the degenerate estimator $g_n = g_0$. The branch
$g = g_0$ of the theorem's hypothesis holds by declaration. The section
[a known treatment mechanism](../dr-tmle/theorem.md#a-known-treatment-mechanism) states the
construction for each guard.

This study measures the complete-data known-mechanism case only. It does not measure an estimated mechanism
with one wrong nuisance.

## What was compared

| setting | `cleverly` | R `drtmle` |
| --- | --- | --- |
| datasets | 1,000 samples of 2,000 rows. The four guard scenarios read the same samples | the identical rows |
| law | the two-arm law of the [TMLE study](known-treatment-mechanism.md). The three-arm scenario has the mechanism $(0.2, 0.3, 0.5)$ at $W_2 = 0$ and $(0.4, 0.4, 0.2)$ at $W_2 = 1$ | the same |
| scenarios | `guard_none`, `guard_q`, `guard_g` and `guard_qg`: each guard with the wrong outcome regression. `three_arm_guard_none`: the three-arm law at `guard=()` | the same guards. The three-arm fit passes the three levels in `a_0` |
| reported parameters | `ey0`, `ey1` and the ATE. At three arms, each arm mean and each contrast against arm 0 | the same |
| mechanism | the declared $g_0$ | `gn`, one vector for each level, with `tolg = 0.1` |
| reduced regressions | a linear regression on the mechanism, and a logistic regression on the outcome regression | `glm_Qr = "gn"` and `glm_gr = "Qn"` |
| fitting | in sample, with this package's initial outcome regression | `cvFolds = 1`, with the same regression as `Qn` |
| intervals | pointwise 95% Wald from the influence curve | the same |

No `tolg` floor binds, because every declared value lies above 0.1. At `guard=()` the fit is the
ordinary TMLE at the declared mechanism. The three-arm scenario is therefore the external
comparison for the multi-arm known path. Under the `"Q"` guards, both sides fluctuate the
mechanism along $Q_r / g$, so the targeted $g^*$ moves off the declaration.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| known mechanism, outcome regression without W2, guard g | `ate` | average treatment effect | `cleverly` DR-TMLE on a declared known mechanism | -0.000825 to 0.0029 | 0.9520 | 1.0249 | pass |
| known mechanism, outcome regression without W2, guard g | `ate` | average treatment effect | R `drtmle` with the known mechanism as `gn` | -0.000580 to 0.0031 | 0.9520 | 1.0256 | pass |
| known mechanism, outcome regression without W2, guard g | `ey0` | counterfactual mean under no treatment | `cleverly` DR-TMLE on a declared known mechanism | -0.0019 to 0.000723 | 0.9460 | 1.0079 | pass |
| known mechanism, outcome regression without W2, guard g | `ey0` | counterfactual mean under no treatment | R `drtmle` with the known mechanism as `gn` | -0.0020 to 0.000584 | 0.9420 | 1.0072 | pass |
| known mechanism, outcome regression without W2, guard g | `ey1` | counterfactual mean under treatment | `cleverly` DR-TMLE on a declared known mechanism | -0.000947 to 0.0018 | 0.9570 | 1.0295 | pass |
| known mechanism, outcome regression without W2, guard g | `ey1` | counterfactual mean under treatment | R `drtmle` with the known mechanism as `gn` | -0.000841 to 0.0020 | 0.9550 | 1.0305 | pass |
| known mechanism, outcome regression without W2, no DR-TMLE guard | `ate` | average treatment effect | `cleverly` DR-TMLE on a declared known mechanism | -0.000732 to 0.0030 | 0.9510 | 1.0249 | pass |
| known mechanism, outcome regression without W2, no DR-TMLE guard | `ate` | average treatment effect | R `drtmle` with the known mechanism as `gn` | -0.000732 to 0.0030 | 0.9590 | 1.0427 | pass |
| known mechanism, outcome regression without W2, no DR-TMLE guard | `ey0` | counterfactual mean under no treatment | `cleverly` DR-TMLE on a declared known mechanism | -0.0020 to 0.000666 | 0.9400 | 1.0065 | pass |
| known mechanism, outcome regression without W2, no DR-TMLE guard | `ey0` | counterfactual mean under no treatment | R `drtmle` with the known mechanism as `gn` | -0.0020 to 0.000666 | 0.9400 | 1.0114 | pass |
| known mechanism, outcome regression without W2, no DR-TMLE guard | `ey1` | counterfactual mean under treatment | `cleverly` DR-TMLE on a declared known mechanism | -0.000903 to 0.0019 | 0.9540 | 1.0327 | pass |
| known mechanism, outcome regression without W2, no DR-TMLE guard | `ey1` | counterfactual mean under treatment | R `drtmle` with the known mechanism as `gn` | -0.000903 to 0.0019 | 0.9650 | 1.0593 | pass |
| known mechanism, outcome regression without W2, guard Q | `ate` | average treatment effect | `cleverly` DR-TMLE on a declared known mechanism | -0.000866 to 0.0029 | 0.9600 | 1.0206 | pass |
| known mechanism, outcome regression without W2, guard Q | `ate` | average treatment effect | R `drtmle` with the known mechanism as `gn` | -0.000866 to 0.0029 | 0.9600 | 1.0209 | pass |
| known mechanism, outcome regression without W2, guard Q | `ey0` | counterfactual mean under no treatment | `cleverly` DR-TMLE on a declared known mechanism | -0.0019 to 0.000733 | 0.9450 | 1.0095 | pass |
| known mechanism, outcome regression without W2, guard Q | `ey0` | counterfactual mean under no treatment | R `drtmle` with the known mechanism as `gn` | -0.0019 to 0.000733 | 0.9430 | 1.0092 | pass |
| known mechanism, outcome regression without W2, guard Q | `ey1` | counterfactual mean under treatment | `cleverly` DR-TMLE on a declared known mechanism | -0.000971 to 0.0018 | 0.9530 | 1.0258 | pass |
| known mechanism, outcome regression without W2, guard Q | `ey1` | counterfactual mean under treatment | R `drtmle` with the known mechanism as `gn` | -0.000971 to 0.0018 | 0.9560 | 1.0269 | pass |
| known mechanism, outcome regression without W2, guards Q and g | `ate` | average treatment effect | `cleverly` DR-TMLE on a declared known mechanism | -0.000874 to 0.0028 | 0.9580 | 1.0216 | pass |
| known mechanism, outcome regression without W2, guards Q and g | `ate` | average treatment effect | R `drtmle` with the known mechanism as `gn` | -0.000944 to 0.0028 | 0.9600 | 1.0211 | pass |
| known mechanism, outcome regression without W2, guards Q and g | `ey0` | counterfactual mean under no treatment | `cleverly` DR-TMLE on a declared known mechanism | -0.0019 to 0.000746 | 0.9460 | 1.0104 | pass |
| known mechanism, outcome regression without W2, guards Q and g | `ey0` | counterfactual mean under no treatment | R `drtmle` with the known mechanism as `gn` | -0.0019 to 0.000764 | 0.9460 | 1.0104 | pass |
| known mechanism, outcome regression without W2, guards Q and g | `ey1` | counterfactual mean under treatment | `cleverly` DR-TMLE on a declared known mechanism | -0.000968 to 0.0018 | 0.9510 | 1.0270 | pass |
| known mechanism, outcome regression without W2, guards Q and g | `ey1` | counterfactual mean under treatment | R `drtmle` with the known mechanism as `gn` | -0.0010 to 0.0018 | 0.9540 | 1.0250 | pass |
| three-arm law with a known mechanism, outcome regression without W2, no DR-TMLE guard | `ate[1.0 vs 0.0]` | difference in counterfactual means, 1.0 versus 0.0 | `cleverly` DR-TMLE on a declared known mechanism | -0.0027 to 0.0017 | 0.9440 | 0.9741 | pass |
| three-arm law with a known mechanism, outcome regression without W2, no DR-TMLE guard | `ate[1.0 vs 0.0]` | difference in counterfactual means, 1.0 versus 0.0 | R `drtmle` with the known mechanism as `gn` | -0.0027 to 0.0017 | 0.9450 | 0.9771 | pass |
| three-arm law with a known mechanism, outcome regression without W2, no DR-TMLE guard | `ate[2.0 vs 0.0]` | difference in counterfactual means, 2.0 versus 0.0 | `cleverly` DR-TMLE on a declared known mechanism | -0.0015 to 0.0031 | 0.9440 | 0.9836 | pass |
| three-arm law with a known mechanism, outcome regression without W2, no DR-TMLE guard | `ate[2.0 vs 0.0]` | difference in counterfactual means, 2.0 versus 0.0 | R `drtmle` with the known mechanism as `gn` | -0.0015 to 0.0031 | 0.9440 | 0.9852 | pass |
| three-arm law with a known mechanism, outcome regression without W2, no DR-TMLE guard | `ey[0.0]` | counterfactual mean under treatment arm '0.0' | `cleverly` DR-TMLE on a declared known mechanism | -0.0018 to 0.0016 | 0.9450 | 0.9753 | pass |
| three-arm law with a known mechanism, outcome regression without W2, no DR-TMLE guard | `ey[0.0]` | counterfactual mean under treatment arm '0.0' | R `drtmle` with the known mechanism as `gn` | -0.0018 to 0.0016 | 0.9460 | 0.9779 | pass |
| three-arm law with a known mechanism, outcome regression without W2, no DR-TMLE guard | `ey[1.0]` | counterfactual mean under treatment arm '1.0' | `cleverly` DR-TMLE on a declared known mechanism | -0.0020 to 0.000898 | 0.9510 | 1.0058 | pass |
| three-arm law with a known mechanism, outcome regression without W2, no DR-TMLE guard | `ey[1.0]` | counterfactual mean under treatment arm '1.0' | R `drtmle` with the known mechanism as `gn` | -0.0020 to 0.000898 | 0.9510 | 1.0081 | pass |
| three-arm law with a known mechanism, outcome regression without W2, no DR-TMLE guard | `ey[2.0]` | counterfactual mean under treatment arm '2.0' | `cleverly` DR-TMLE on a declared known mechanism | -0.000868 to 0.0023 | 0.9510 | 1.0164 | pass |
| three-arm law with a known mechanism, outcome regression without W2, no DR-TMLE guard | `ey[2.0]` | counterfactual mean under treatment arm '2.0' | R `drtmle` with the known mechanism as `gn` | -0.000868 to 0.0023 | 0.9510 | 1.0171 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| known mechanism, outcome regression without W2, guard g | `ate` | average treatment effect | -0.000246 | 0.0719 | 1.0017 | 0 | 0.0033 vs 0.0500 | equivalent |
| known mechanism, outcome regression without W2, guard g | `ey0` | counterfactual mean under no treatment | 0.000140 | 0.0579 | 1.0017 | 0.0040 | 0.0026 vs 0.0500 | equivalent |
| known mechanism, outcome regression without W2, guard g | `ey1` | counterfactual mean under treatment | -0.000106 | 0.0415 | 1.0023 | 0.0020 | 0.0033 vs 0.0500 | equivalent |
| known mechanism, outcome regression without W2, no DR-TMLE guard | `ate` | average treatment effect | 1.390e-10 | 4.034e-08 | 1.0000 | -0.0080 | 0.0349 vs 0.0500 | equivalent |
| known mechanism, outcome regression without W2, no DR-TMLE guard | `ey0` | counterfactual mean under no treatment | -1.384e-10 | 5.725e-08 | 1.0000 | 0 | 0.0097 vs 0.0500 | equivalent |
| known mechanism, outcome regression without W2, no DR-TMLE guard | `ey1` | counterfactual mean under treatment | 6.802e-13 | 2.647e-10 | 1.0000 | -0.0110 | 0.0340 vs 0.0500 | equivalent |
| known mechanism, outcome regression without W2, guard Q | `ate` | average treatment effect | 4.718e-08 | 0.000014 | 1.0000 | 0 | 0.0011 vs 0.0500 | equivalent |
| known mechanism, outcome regression without W2, guard Q | `ey0` | counterfactual mean under no treatment | -1.976e-07 | 0.000082 | 1.0000 | 0.0020 | 0.0011 vs 0.0500 | equivalent |
| known mechanism, outcome regression without W2, guard Q | `ey1` | counterfactual mean under treatment | -1.504e-07 | 0.000059 | 1.0000 | -0.0030 | 0.0027 vs 0.0500 | equivalent |
| known mechanism, outcome regression without W2, guards Q and g | `ate` | average treatment effect | 0.000065 | 0.0191 | 1.0004 | -0.0020 | 0.0031 vs 0.0500 | equivalent |
| known mechanism, outcome regression without W2, guards Q and g | `ey0` | counterfactual mean under no treatment | -0.000017 | 0.0070 | 1.0010 | 0 | 0.0012 vs 0.0500 | equivalent |
| known mechanism, outcome regression without W2, guards Q and g | `ey1` | counterfactual mean under treatment | 0.000048 | 0.0189 | 0.9996 | -0.0030 | 0.0041 vs 0.0500 | equivalent |
| three-arm law with a known mechanism, outcome regression without W2, no DR-TMLE guard | `ate[1.0 vs 0.0]` | difference in counterfactual means, 1.0 versus 0.0 | 1.256e-09 | 3.053e-07 | 1.0000 | -0.0010 | 0.000135 vs 0.0500 | equivalent |
| three-arm law with a known mechanism, outcome regression without W2, no DR-TMLE guard | `ate[2.0 vs 0.0]` | difference in counterfactual means, 2.0 versus 0.0 | 1.575e-10 | 3.690e-08 | 1.0000 | 0 | 0.000090 vs 0.0500 | equivalent |
| three-arm law with a known mechanism, outcome regression without W2, no DR-TMLE guard | `ey[0.0]` | counterfactual mean under treatment arm '0.0' | -7.745e-10 | 2.505e-07 | 1.0000 | -0.0010 | 0.000162 vs 0.0500 | equivalent |
| three-arm law with a known mechanism, outcome regression without W2, no DR-TMLE guard | `ey[1.0]` | counterfactual mean under treatment arm '1.0' | 4.814e-10 | 1.779e-07 | 1.0000 | 0 | 0.0047 vs 0.0500 | equivalent |
| three-arm law with a known mechanism, outcome regression without W2, no DR-TMLE guard | `ey[2.0]` | counterfactual mean under treatment arm '2.0' | -6.170e-10 | 2.111e-07 | 1.0000 | 0 | 0.0014 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `known_mechanism_accuracy` | `ate__drtmle_known__Q` | positive | average treatment effect: DR-TMLE with guard Q on the declared mechanism, outcome regression without W2 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0012 to 0.0027, margin 0.0060, coverage 0.9342 to 0.9694, SE ratio 0.9786 | pass |
| `known_mechanism_accuracy` | `ate__drtmle_known__Qg` | positive | average treatment effect: DR-TMLE with guards Q and g on the declared mechanism, outcome regression without W2 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0012 to 0.0027, margin 0.0059, coverage 0.9307 to 0.9669, SE ratio 0.9762 | pass |
| `known_mechanism_accuracy` | `ate__drtmle_known__g` | positive | average treatment effect: DR-TMLE with guard g on the declared mechanism, outcome regression without W2 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000977 to 0.0030, margin 0.0060, coverage 0.9295 to 0.9661, SE ratio 0.9723 | pass |
| `known_mechanism_accuracy` | `ate__drtmle_known__none` | positive | average treatment effect: DR-TMLE with no guard on the declared mechanism, outcome regression without W2 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0010 to 0.0029, margin 0.0060, coverage 0.9272 to 0.9644, SE ratio 0.9754 | pass |
<!-- /generated -->

Each property cell fits the ATE at one guard with the wrong outcome regression. The four cells
read one draw for each replication, and these draws differ from the primary draws. At the
declared mechanism the remainder is zero for any outcome regression. Each cell must therefore be
unbiased, cover at the floor and report a standard error inside the SE-ratio screen.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | primary replications per scenario |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 34 | implementation-estimand tests against the truth |
| `independent_tests_passed` | 34 | of those, passing |
| `paired_tests_total` | 17 | paired comparisons with R `drtmle` |
| `paired_tests_passed` | 17 | of those, passing |
| `property_cells_total` | 4 | repeated-sampling property cells |
| `property_cells_passed` | 4 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0562 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9400 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.9228 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.1237 | highest bootstrap primary SE-ratio endpoint |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals, and the family level of the rule |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit. No cell of this study reads it |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit. No cell of this study reads it |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit. No cell of this study reads it |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear. No cell of this study reads it |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit. No cell of this study reads it |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:type_i_ceiling` | 0.1000 | the rate a positive cell must bound and a control must exceed. No cell of this study reads it |

## Limitations

| limit | what it means for use |
| --- | --- |
| a declared mechanism only | the study does not measure `DRTMLE` with an estimated mechanism. With one wrong nuisance and an estimated mechanism, the interval is not established here |
| complete data, in sample, with the univariate reduction | the cross-fitted, nested and bivariate paths have exact-law evidence only |
| no missing-outcome path | with `delta=`, the fit still estimates the observation mechanism. With a wrong outcome regression, the interval of that route is not established, because its remainder is linear in the observation mechanism's error. This is a gap in the theory, not in the evidence. See [a known treatment mechanism](../dr-tmle/theorem.md#a-known-treatment-mechanism) |
| one wrong outcome regression | a flexible outcome learner has fast-tier evidence only |

## Reproduction

The [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/known_mechanism_drtmle/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. Run `python -m tests.canonical.known_mechanism_drtmle.regenerate` to regenerate the
artifacts. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/known_mechanism_drtmle/replicates.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/known_mechanism_drtmle/properties.csv)
carry every published row.
