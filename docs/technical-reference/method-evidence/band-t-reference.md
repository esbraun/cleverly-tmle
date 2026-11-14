# Per-estimate t reference band

This study measures the simultaneous band built with `band_reference="t"`. The band keeps the
multiplier critical value $c$ of the default band and gives estimate $j$ its own critical value

$$
c_j = F^{-1}_{t,\nu_j}\{\Phi(c)\}, \qquad \nu_j = \max(K_j, 5),
$$

where $K_j = (\sum_i d_{ij}^2)^2 / \sum_i d_{ij}^4$ is the Kish count of estimate $j$'s centred
influence curve $d_{\cdot j}$. Each estimate keeps the two-sided tail $2\{1 - \Phi(c)\}$ of the
default band, read on its own Student t reference. The
[simultaneous bands](../inference.md#simultaneous-bands) section of the inference reference
defines the band, its scale and the `t` reference. The default band is unchanged.

Each cell refits the subject fit of a registered source study with the default band, then reads
the `t` band and the default band from the same estimates, multiplier draws and $c$. A second
study, [band-t-reference-held-out](band-t-reference-held-out.md), holds out one further cell.

**No canonical implementation is compared.** No pinned comparator publishes a multiplier band
with a per-estimate t reference. The zero-row equivalence artifact records the absence of a
comparator.

## What was tested

| setting | declaration |
| --- | --- |
| `t` band cell | `<cell>__t_band`. Each replication is covered when every truth lies inside the `t` band. The 99% exact interval of the joint coverage must lie inside $[0.92, 0.98]$ |
| pointwise control | `<cell>__pointwise_joint_control`. The pointwise 95% intervals of the same fits, read jointly. The 99% upper endpoint must lie below 0.95 |
| normal-reference control | `<cell>__normal_reference_control`. The default band of the same fits. Its row reports the paired gain of the `t` band over it, the share of replications the `t` band covers and the default band does not, minus the reverse, with its 99% interval. The gain gates nothing |
| diagnostic arms | the same fits read with the moment-matched df $2K/(1 - K/m)$ over the $m$ rows that carry the curve, the model-assisted df (the Kish count of the covariate-cell means of $d^2$), one common df $\max(\min_j K_j, 5)$ through the same mapping, and that common df through the exact multivariate t. No arm states a verdict |
| deterministic checks | on every replication the row module recomputes each $c_j$ and $\nu_j$ and requires the package's values to agree to a relative $10^{-12}$, checks the large-df expansion bound, and requires $\max_j (c_j/c - 1) > 0$ wherever a Kish count is finite. A replication that fails a check stops the run |
| constant arm cell | a replication whose band family holds a treatment arm with a single observed outcome in a stratum counts as not covered, whatever band the fit shows |
| budget | 5,600 replications for each cell: the larger of the arm-pooled predictive rule on coverage alone over the eleven cells (2,400) and over the three red shapes (5,600), sized on an 800-replication pilot on a salted stream |
| primary scenario | `t_reference_fit`: `TMLE()` with two main-terms logistic learners, `random_state=0` and `band_reference="t"`, every other argument at its default, on the binary law of the [ordinary point-treatment study](canonical-point-treatment-tmle.md). $n = 1000$, 1,000 replications. The `t` band changes no estimate, standard error or pointwise interval |
| Monte Carlo inference | exact 99% intervals around every coverage rate and paired gain, and 99% intervals around every primary endpoint |

The table gives each cell, the support rows its verdict licenses, and the gain value declared for
it before the run. The red group holds the three shapes whose default band is red in
[default simultaneous bands](default-simultaneous-bands.md) or in the
[stratified point-treatment study](stratified-point-treatment-tmle.md).

| cell | shape | n | group | support rows | declared gain value |
| --- | --- | ---: | --- | --- | --- |
| `categorical_ltmle` | in-sample categorical LTMLE, five regimen means and four contrasts | 2,000 | red | `G08.in_sample` | 0.0001 |
| `cde_z0` | controlled direct effect at $z = 0$, missing outcomes | 2,000 | red | `T30` | 0.0001 |
| `strata` | in-sample TMLE with three baseline strata, 24 parameters | 2,000 | red | `T13` | 0.0001 |
| `categorical_ltmle_n8000` | as `categorical_ltmle` | 8,000 | larger size | `G08.in_sample` | none |
| `cde_z0_n8000` | as `cde_z0` | 8,000 | larger size | `T30` | none |
| `strata_n8000` | as `strata` | 8,000 | larger size | `T13` | none |
| `categorical_ltmle_crossfit` | cross-fitted categorical LTMLE | 2,000 | cross-fitted | `G08.cross_fitted` | none |
| `crossfit_strata` | cross-fitted TMLE with three baseline strata, 12 parameters | 2,000 | cross-fitted | `T14` | none |
| `six_regimens` | cross-fitted sequentially doubly robust fit, six regimen means and five contrasts | 2,000 | cross-fitted | `G23.cross_fitted` | none |
| `ordinary_binary` | ordinary in-sample TMLE over ten binary-outcome estimands | 1,000 | high Kish | `T01.binary` | none |
| `point_msm` | in-sample point-treatment MSM, three projection terms | 2,000 | high Kish | `T18` | none |

A declared gain value is a formula of pilot quantities fixed before the run. At 5,600
replications each red cell's magnitude formula fell below the floor, so the three red cells
declare the nonzero witness $10^{-4}$, and no other cell's pilot gain selected a witness. The
module docstring of `tests/studies/band_t_reference.py` states the rule, the cells and the
reading declared before the pilot. `tests/unit/test_band_t_reference_design.py` recomputes the
declared values, the replication count and the pass probabilities from the committed pilot.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law, fitted with the shipped TMLE() defaults and band_reference='t' | `atc` | average effect on the untreated | `cleverly` shipped TMLE() default fit with band_reference='t' | -0.0031 to 0.0018 | 0.9570 | 1.0300 | pass |
| binary-outcome law, fitted with the shipped TMLE() defaults and band_reference='t' | `ate` | average treatment effect | `cleverly` shipped TMLE() default fit with band_reference='t' | -0.0032 to 0.0016 | 0.9520 | 1.0189 | pass |
| binary-outcome law, fitted with the shipped TMLE() defaults and band_reference='t' | `att` | average effect on the treated | `cleverly` shipped TMLE() default fit with band_reference='t' | -0.0035 to 0.0016 | 0.9520 | 1.0054 | pass |
| binary-outcome law, fitted with the shipped TMLE() defaults and band_reference='t' | `ey0` | counterfactual mean under no treatment | `cleverly` shipped TMLE() default fit with band_reference='t' | -0.0016 to 0.0020 | 0.9500 | 0.9978 | pass |
| binary-outcome law, fitted with the shipped TMLE() defaults and band_reference='t' | `ey1` | counterfactual mean under treatment | `cleverly` shipped TMLE() default fit with band_reference='t' | -0.0024 to 0.0012 | 0.9580 | 1.0282 | pass |
| binary-outcome law, fitted with the shipped TMLE() defaults and band_reference='t' | `or` | marginal odds ratio, reported on the log scale | `cleverly` shipped TMLE() default fit with band_reference='t' | -0.0117 to 0.0085 | 0.9550 | 1.0182 | pass |
| binary-outcome law, fitted with the shipped TMLE() defaults and band_reference='t' | `rr` | marginal risk ratio, reported on the log scale | `cleverly` shipped TMLE() default fit with band_reference='t' | -0.0061 to 0.0049 | 0.9550 | 1.0139 | pass |
<!-- /generated -->

## Theory properties

The cell name gives the shape, then the reading. A `reference_control` row prints the joint
coverage of the default band and the paired gain of the `t` band over it.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `simultaneous_coverage` | `categorical_ltmle__common_df_copula_diagnostic` | diagnostic | five categorical regimen means and four contrasts: the t band with one df, the floored smallest Kish count, for every estimate | none. The row is a diagnostic and states no verdict | joint coverage 0.9420 to 0.9572 | reported |
| `simultaneous_coverage` | `categorical_ltmle__common_df_mvt_diagnostic` | diagnostic | five categorical regimen means and four contrasts: the band from the multivariate t with one shared df, the floored smallest Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9399 to 0.9554 | reported |
| `simultaneous_coverage` | `categorical_ltmle__model_df_diagnostic` | diagnostic | five categorical regimen means and four contrasts: the t band with the Kish count of the covariate-cell means of the squared curve | none. The row is a diagnostic and states no verdict | joint coverage 0.9344 to 0.9506 | reported |
| `simultaneous_coverage` | `categorical_ltmle__moment_df_diagnostic` | diagnostic | five categorical regimen means and four contrasts: the t band with the moment-matched df 2K / (1 - K/m) in place of the Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9333 to 0.9496 | reported |
| `simultaneous_coverage` | `categorical_ltmle__normal_reference_control` | reference_control | five categorical regimen means and four contrasts: the default band, on the normal reference, from the same fits, draws and critical value | none. The paired gain of the t band over it is reported and gates nothing | joint coverage 0.9283 to 0.9452, paired gain of the t band 0.0059 to 0.0123 | pass |
| `simultaneous_coverage` | `categorical_ltmle__pointwise_joint_control` | control | five categorical regimen means and four contrasts: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.6657 to 0.6979 | pass |
| `simultaneous_coverage` | `categorical_ltmle__t_band` | positive | five categorical regimen means and four contrasts: the default multiplier band with band_reference='t', which reads each estimate's critical value on a Student t reference with max(Kish, 5) degrees of freedom | joint coverage interval inside the calibration coverage band | joint coverage 0.9378 to 0.9536 | pass |
| `simultaneous_coverage` | `categorical_ltmle_crossfit__common_df_copula_diagnostic` | diagnostic | five cross-fitted categorical regimen means and four contrasts: the t band with one df, the floored smallest Kish count, for every estimate | none. The row is a diagnostic and states no verdict | joint coverage 0.9476 to 0.9620 | reported |
| `simultaneous_coverage` | `categorical_ltmle_crossfit__common_df_mvt_diagnostic` | diagnostic | five cross-fitted categorical regimen means and four contrasts: the band from the multivariate t with one shared df, the floored smallest Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9451 to 0.9599 | reported |
| `simultaneous_coverage` | `categorical_ltmle_crossfit__model_df_diagnostic` | diagnostic | five cross-fitted categorical regimen means and four contrasts: the t band with the Kish count of the covariate-cell means of the squared curve | none. The row is a diagnostic and states no verdict | joint coverage 0.9380 to 0.9537 | reported |
| `simultaneous_coverage` | `categorical_ltmle_crossfit__moment_df_diagnostic` | diagnostic | five cross-fitted categorical regimen means and four contrasts: the t band with the moment-matched df 2K / (1 - K/m) in place of the Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9371 to 0.9529 | reported |
| `simultaneous_coverage` | `categorical_ltmle_crossfit__normal_reference_control` | reference_control | five cross-fitted categorical regimen means and four contrasts: the default band, on the normal reference, from the same fits, draws and critical value | none. The paired gain of the t band over it is reported and gates nothing | joint coverage 0.9331 to 0.9494, paired gain of the t band 0.0055 to 0.0120 | pass |
| `simultaneous_coverage` | `categorical_ltmle_crossfit__pointwise_joint_control` | control | five cross-fitted categorical regimen means and four contrasts: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.6881 to 0.7197 | pass |
| `simultaneous_coverage` | `categorical_ltmle_crossfit__t_band` | positive | five cross-fitted categorical regimen means and four contrasts: the default multiplier band with band_reference='t', which reads each estimate's critical value on a Student t reference with max(Kish, 5) degrees of freedom | joint coverage interval inside the calibration coverage band | joint coverage 0.9422 to 0.9574 | pass |
| `simultaneous_coverage` | `categorical_ltmle_n8000__common_df_copula_diagnostic` | diagnostic | five categorical regimen means and four contrasts, at n = 8,000: the t band with one df, the floored smallest Kish count, for every estimate | none. The row is a diagnostic and states no verdict | joint coverage 0.9428 to 0.9579 | reported |
| `simultaneous_coverage` | `categorical_ltmle_n8000__common_df_mvt_diagnostic` | diagnostic | five categorical regimen means and four contrasts, at n = 8,000: the band from the multivariate t with one shared df, the floored smallest Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9422 to 0.9574 | reported |
| `simultaneous_coverage` | `categorical_ltmle_n8000__model_df_diagnostic` | diagnostic | five categorical regimen means and four contrasts, at n = 8,000: the t band with the Kish count of the covariate-cell means of the squared curve | none. The row is a diagnostic and states no verdict | joint coverage 0.9411 to 0.9564 | reported |
| `simultaneous_coverage` | `categorical_ltmle_n8000__moment_df_diagnostic` | diagnostic | five categorical regimen means and four contrasts, at n = 8,000: the t band with the moment-matched df 2K / (1 - K/m) in place of the Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9409 to 0.9562 | reported |
| `simultaneous_coverage` | `categorical_ltmle_n8000__normal_reference_control` | reference_control | five categorical regimen means and four contrasts, at n = 8,000: the default band, on the normal reference, from the same fits, draws and critical value | none. The paired gain of the t band over it is reported and gates nothing | joint coverage 0.9397 to 0.9552, paired gain of the t band 0.000536 to 0.0034 | pass |
| `simultaneous_coverage` | `categorical_ltmle_n8000__pointwise_joint_control` | control | five categorical regimen means and four contrasts, at n = 8,000: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.6938 to 0.7252 | pass |
| `simultaneous_coverage` | `categorical_ltmle_n8000__t_band` | positive | five categorical regimen means and four contrasts, at n = 8,000: the default multiplier band with band_reference='t', which reads each estimate's critical value on a Student t reference with max(Kish, 5) degrees of freedom | joint coverage interval inside the calibration coverage band | joint coverage 0.9416 to 0.9569 | pass |
| `simultaneous_coverage` | `cde_z0__common_df_copula_diagnostic` | diagnostic | the controlled direct-effect fit at intermediate level zero: the t band with one df, the floored smallest Kish count, for every estimate | none. The row is a diagnostic and states no verdict | joint coverage 0.9605 to 0.9730 | reported |
| `simultaneous_coverage` | `cde_z0__common_df_mvt_diagnostic` | diagnostic | the controlled direct-effect fit at intermediate level zero: the band from the multivariate t with one shared df, the floored smallest Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9574 to 0.9704 | reported |
| `simultaneous_coverage` | `cde_z0__model_df_diagnostic` | diagnostic | the controlled direct-effect fit at intermediate level zero: the t band with the Kish count of the covariate-cell means of the squared curve | none. The row is a diagnostic and states no verdict | joint coverage 0.9369 to 0.9527 | reported |
| `simultaneous_coverage` | `cde_z0__moment_df_diagnostic` | diagnostic | the controlled direct-effect fit at intermediate level zero: the t band with the moment-matched df 2K / (1 - K/m) in place of the Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9413 to 0.9565 | reported |
| `simultaneous_coverage` | `cde_z0__normal_reference_control` | reference_control | the controlled direct-effect fit at intermediate level zero: the default band, on the normal reference, from the same fits, draws and critical value | none. The paired gain of the t band over it is reported and gates nothing | joint coverage 0.9295 to 0.9462, paired gain of the t band 0.0120 to 0.0204 | pass |
| `simultaneous_coverage` | `cde_z0__pointwise_joint_control` | control | the controlled direct-effect fit at intermediate level zero: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8552 to 0.8788 | pass |
| `simultaneous_coverage` | `cde_z0__t_band` | positive | the controlled direct-effect fit at intermediate level zero: the default multiplier band with band_reference='t', which reads each estimate's critical value on a Student t reference with max(Kish, 5) degrees of freedom | joint coverage interval inside the calibration coverage band | joint coverage 0.9466 to 0.9612 | pass |
| `simultaneous_coverage` | `cde_z0_n8000__common_df_copula_diagnostic` | diagnostic | the controlled direct-effect fit at intermediate level zero, at n = 8,000: the t band with one df, the floored smallest Kish count, for every estimate | none. The row is a diagnostic and states no verdict | joint coverage 0.9392 to 0.9547 | reported |
| `simultaneous_coverage` | `cde_z0_n8000__common_df_mvt_diagnostic` | diagnostic | the controlled direct-effect fit at intermediate level zero, at n = 8,000: the band from the multivariate t with one shared df, the floored smallest Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9384 to 0.9541 | reported |
| `simultaneous_coverage` | `cde_z0_n8000__model_df_diagnostic` | diagnostic | the controlled direct-effect fit at intermediate level zero, at n = 8,000: the t band with the Kish count of the covariate-cell means of the squared curve | none. The row is a diagnostic and states no verdict | joint coverage 0.9344 to 0.9506 | reported |
| `simultaneous_coverage` | `cde_z0_n8000__moment_df_diagnostic` | diagnostic | the controlled direct-effect fit at intermediate level zero, at n = 8,000: the t band with the moment-matched df 2K / (1 - K/m) in place of the Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9340 to 0.9502 | reported |
| `simultaneous_coverage` | `cde_z0_n8000__normal_reference_control` | reference_control | the controlled direct-effect fit at intermediate level zero, at n = 8,000: the default band, on the normal reference, from the same fits, draws and critical value | none. The paired gain of the t band over it is reported and gates nothing | joint coverage 0.9325 to 0.9489, paired gain of the t band 0.0013 to 0.0050 | pass |
| `simultaneous_coverage` | `cde_z0_n8000__pointwise_joint_control` | control | the controlled direct-effect fit at intermediate level zero, at n = 8,000: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8595 to 0.8827 | pass |
| `simultaneous_coverage` | `cde_z0_n8000__t_band` | positive | the controlled direct-effect fit at intermediate level zero, at n = 8,000: the default multiplier band with band_reference='t', which reads each estimate's critical value on a Student t reference with max(Kish, 5) degrees of freedom | joint coverage interval inside the calibration coverage band | joint coverage 0.9357 to 0.9517 | pass |
| `simultaneous_coverage` | `crossfit_strata__common_df_copula_diagnostic` | diagnostic | the three marginal and nine stratum parameters of the cross-fitted fit: the t band with one df, the floored smallest Kish count, for every estimate | none. The row is a diagnostic and states no verdict | joint coverage 0.9905 to 0.9961 | reported |
| `simultaneous_coverage` | `crossfit_strata__common_df_mvt_diagnostic` | diagnostic | the three marginal and nine stratum parameters of the cross-fitted fit: the band from the multivariate t with one shared df, the floored smallest Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9894 to 0.9954 | reported |
| `simultaneous_coverage` | `crossfit_strata__model_df_diagnostic` | diagnostic | the three marginal and nine stratum parameters of the cross-fitted fit: the t band with the Kish count of the covariate-cell means of the squared curve | none. The row is a diagnostic and states no verdict | joint coverage 0.9491 to 0.9633 | reported |
| `simultaneous_coverage` | `crossfit_strata__moment_df_diagnostic` | diagnostic | the three marginal and nine stratum parameters of the cross-fitted fit: the t band with the moment-matched df 2K / (1 - K/m) in place of the Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9501 to 0.9641 | reported |
| `simultaneous_coverage` | `crossfit_strata__normal_reference_control` | reference_control | the three marginal and nine stratum parameters of the cross-fitted fit: the default band, on the normal reference, from the same fits, draws and critical value | none. The paired gain of the t band over it is reported and gates nothing | joint coverage 0.9458 to 0.9605, paired gain of the t band 0.0054 to 0.0118 | pass |
| `simultaneous_coverage` | `crossfit_strata__pointwise_joint_control` | control | the three marginal and nine stratum parameters of the cross-fitted fit: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.6126 to 0.6461 | pass |
| `simultaneous_coverage` | `crossfit_strata__t_band` | positive | the three marginal and nine stratum parameters of the cross-fitted fit: the default multiplier band with band_reference='t', which reads each estimate's critical value on a Student t reference with max(Kish, 5) degrees of freedom | joint coverage interval inside the calibration coverage band | joint coverage 0.9549 to 0.9682 | pass |
| `simultaneous_coverage` | `ordinary_binary__common_df_copula_diagnostic` | diagnostic | ordinary in-sample TMLE over ten binary-outcome estimands: the t band with one df, the floored smallest Kish count, for every estimate | none. The row is a diagnostic and states no verdict | joint coverage 0.9426 to 0.9577 | reported |
| `simultaneous_coverage` | `ordinary_binary__common_df_mvt_diagnostic` | diagnostic | ordinary in-sample TMLE over ten binary-outcome estimands: the band from the multivariate t with one shared df, the floored smallest Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9424 to 0.9575 | reported |
| `simultaneous_coverage` | `ordinary_binary__model_df_diagnostic` | diagnostic | ordinary in-sample TMLE over ten binary-outcome estimands: the t band with the Kish count of the covariate-cell means of the squared curve | none. The row is a diagnostic and states no verdict | joint coverage 0.9416 to 0.9569 | reported |
| `simultaneous_coverage` | `ordinary_binary__moment_df_diagnostic` | diagnostic | ordinary in-sample TMLE over ten binary-outcome estimands: the t band with the moment-matched df 2K / (1 - K/m) in place of the Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9409 to 0.9562 | reported |
| `simultaneous_coverage` | `ordinary_binary__normal_reference_control` | reference_control | ordinary in-sample TMLE over ten binary-outcome estimands: the default band, on the normal reference, from the same fits, draws and critical value | none. The paired gain of the t band over it is reported and gates nothing | joint coverage 0.9405 to 0.9559, paired gain of the t band 0.000179 to 0.0023 | pass |
| `simultaneous_coverage` | `ordinary_binary__pointwise_joint_control` | control | ordinary in-sample TMLE over ten binary-outcome estimands: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8299 to 0.8552 | pass |
| `simultaneous_coverage` | `ordinary_binary__t_band` | positive | ordinary in-sample TMLE over ten binary-outcome estimands: the default multiplier band with band_reference='t', which reads each estimate's critical value on a Student t reference with max(Kish, 5) degrees of freedom | joint coverage interval inside the calibration coverage band | joint coverage 0.9416 to 0.9569 | pass |
| `simultaneous_coverage` | `point_msm__common_df_copula_diagnostic` | diagnostic | the three terms of the point-treatment MSM projection: the t band with one df, the floored smallest Kish count, for every estimate | none. The row is a diagnostic and states no verdict | joint coverage 0.9439 to 0.9589 | reported |
| `simultaneous_coverage` | `point_msm__common_df_mvt_diagnostic` | diagnostic | the three terms of the point-treatment MSM projection: the band from the multivariate t with one shared df, the floored smallest Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9443 to 0.9592 | reported |
| `simultaneous_coverage` | `point_msm__model_df_diagnostic` | diagnostic | the three terms of the point-treatment MSM projection: the t band with the Kish count of the covariate-cell means of the squared curve | none. The row is a diagnostic and states no verdict | joint coverage 0.9413 to 0.9565 | reported |
| `simultaneous_coverage` | `point_msm__moment_df_diagnostic` | diagnostic | the three terms of the point-treatment MSM projection: the t band with the moment-matched df 2K / (1 - K/m) in place of the Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9418 to 0.9570 | reported |
| `simultaneous_coverage` | `point_msm__normal_reference_control` | reference_control | the three terms of the point-treatment MSM projection: the default band, on the normal reference, from the same fits, draws and critical value | none. The paired gain of the t band over it is reported and gates nothing | joint coverage 0.9399 to 0.9554, paired gain of the t band 0.0013 to 0.0048 | pass |
| `simultaneous_coverage` | `point_msm__pointwise_joint_control` | control | the three terms of the point-treatment MSM projection: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8717 to 0.8940 | pass |
| `simultaneous_coverage` | `point_msm__t_band` | positive | the three terms of the point-treatment MSM projection: the default multiplier band with band_reference='t', which reads each estimate's critical value on a Student t reference with max(Kish, 5) degrees of freedom | joint coverage interval inside the calibration coverage band | joint coverage 0.9430 to 0.9580 | pass |
| `simultaneous_coverage` | `six_regimens__common_df_copula_diagnostic` | diagnostic | the six regimen means and five contrasts of the two-node law: the t band with one df, the floored smallest Kish count, for every estimate | none. The row is a diagnostic and states no verdict | joint coverage 0.9493 to 0.9635 | reported |
| `simultaneous_coverage` | `six_regimens__common_df_mvt_diagnostic` | diagnostic | the six regimen means and five contrasts of the two-node law: the band from the multivariate t with one shared df, the floored smallest Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9457 to 0.9603 | reported |
| `simultaneous_coverage` | `six_regimens__model_df_diagnostic` | diagnostic | the six regimen means and five contrasts of the two-node law: the t band with the Kish count of the covariate-cell means of the squared curve | none. The row is a diagnostic and states no verdict | joint coverage 0.9371 to 0.9529 | reported |
| `simultaneous_coverage` | `six_regimens__moment_df_diagnostic` | diagnostic | the six regimen means and five contrasts of the two-node law: the t band with the moment-matched df 2K / (1 - K/m) in place of the Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9352 to 0.9512 | reported |
| `simultaneous_coverage` | `six_regimens__normal_reference_control` | reference_control | the six regimen means and five contrasts of the two-node law: the default band, on the normal reference, from the same fits, draws and critical value | none. The paired gain of the t band over it is reported and gates nothing | joint coverage 0.9314 to 0.9479, paired gain of the t band 0.0054 to 0.0116 | pass |
| `simultaneous_coverage` | `six_regimens__pointwise_joint_control` | control | the six regimen means and five contrasts of the two-node law: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.6861 to 0.7178 | pass |
| `simultaneous_coverage` | `six_regimens__t_band` | positive | the six regimen means and five contrasts of the two-node law: the default multiplier band with band_reference='t', which reads each estimate's critical value on a Student t reference with max(Kish, 5) degrees of freedom | joint coverage interval inside the calibration coverage band | joint coverage 0.9401 to 0.9555 | pass |
| `simultaneous_coverage` | `strata__common_df_copula_diagnostic` | diagnostic | the six marginal and eighteen stratum parameters of the in-sample fit: the t band with one df, the floored smallest Kish count, for every estimate | none. The row is a diagnostic and states no verdict | joint coverage 0.9927 to 0.9975 | reported |
| `simultaneous_coverage` | `strata__common_df_mvt_diagnostic` | diagnostic | the six marginal and eighteen stratum parameters of the in-sample fit: the band from the multivariate t with one shared df, the floored smallest Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9911 to 0.9965 | reported |
| `simultaneous_coverage` | `strata__model_df_diagnostic` | diagnostic | the six marginal and eighteen stratum parameters of the in-sample fit: the t band with the Kish count of the covariate-cell means of the squared curve | none. The row is a diagnostic and states no verdict | joint coverage 0.9397 to 0.9552 | reported |
| `simultaneous_coverage` | `strata__moment_df_diagnostic` | diagnostic | the six marginal and eighteen stratum parameters of the in-sample fit: the t band with the moment-matched df 2K / (1 - K/m) in place of the Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9418 to 0.9570 | reported |
| `simultaneous_coverage` | `strata__normal_reference_control` | reference_control | the six marginal and eighteen stratum parameters of the in-sample fit: the default band, on the normal reference, from the same fits, draws and critical value | none. The paired gain of the t band over it is reported and gates nothing | joint coverage 0.9367 to 0.9526, paired gain of the t band 0.0063 to 0.0130 | pass |
| `simultaneous_coverage` | `strata__pointwise_joint_control` | control | the six marginal and eighteen stratum parameters of the in-sample fit: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.5734 to 0.6075 | pass |
| `simultaneous_coverage` | `strata__t_band` | positive | the six marginal and eighteen stratum parameters of the in-sample fit: the default multiplier band with band_reference='t', which reads each estimate's critical value on a Student t reference with max(Kish, 5) degrees of freedom | joint coverage interval inside the calibration coverage band | joint coverage 0.9468 to 0.9613 | pass |
| `simultaneous_coverage` | `strata_n8000__common_df_copula_diagnostic` | diagnostic | the six marginal and eighteen stratum parameters of the in-sample fit, at n = 8,000: the t band with one df, the floored smallest Kish count, for every estimate | none. The row is a diagnostic and states no verdict | joint coverage 0.9642 to 0.9760 | reported |
| `simultaneous_coverage` | `strata_n8000__common_df_mvt_diagnostic` | diagnostic | the six marginal and eighteen stratum parameters of the in-sample fit, at n = 8,000: the band from the multivariate t with one shared df, the floored smallest Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9632 to 0.9752 | reported |
| `simultaneous_coverage` | `strata_n8000__model_df_diagnostic` | diagnostic | the six marginal and eighteen stratum parameters of the in-sample fit, at n = 8,000: the t band with the Kish count of the covariate-cell means of the squared curve | none. The row is a diagnostic and states no verdict | joint coverage 0.9414 to 0.9567 | reported |
| `simultaneous_coverage` | `strata_n8000__moment_df_diagnostic` | diagnostic | the six marginal and eighteen stratum parameters of the in-sample fit, at n = 8,000: the t band with the moment-matched df 2K / (1 - K/m) in place of the Kish count | none. The row is a diagnostic and states no verdict | joint coverage 0.9420 to 0.9572 | reported |
| `simultaneous_coverage` | `strata_n8000__normal_reference_control` | reference_control | the six marginal and eighteen stratum parameters of the in-sample fit, at n = 8,000: the default band, on the normal reference, from the same fits, draws and critical value | none. The paired gain of the t band over it is reported and gates nothing | joint coverage 0.9399 to 0.9554, paired gain of the t band 0.0018 to 0.0057 | pass |
| `simultaneous_coverage` | `strata_n8000__pointwise_joint_control` | control | the six marginal and eighteen stratum parameters of the in-sample fit, at n = 8,000: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.6024 to 0.6360 | pass |
| `simultaneous_coverage` | `strata_n8000__t_band` | positive | the six marginal and eighteen stratum parameters of the in-sample fit, at n = 8,000: the default multiplier band with band_reference='t', which reads each estimate's critical value on a Student t reference with max(Kish, 5) degrees of freedom | joint coverage interval inside the calibration coverage band | joint coverage 0.9437 to 0.9587 | pass |
<!-- /generated -->

## Result

Every primary test passed, and all 33 counted property cells pass. Each of the eleven `t` bands
kept its joint-coverage interval inside $[0.92, 0.98]$, and each pointwise control fell below
0.95. The eleven normal-reference controls pass by construction, because their gain is reported,
and the other 44 rows are diagnostic arms with no verdict. The study is gated, so each cell
licenses `("mammen", "logit", "t")` on the support rows the table above names. The
[simultaneous bands](../inference.md#simultaneous-bands) section of the inference reference lists
the rows.

| cell | `t` band | default band | paired gain, 99% interval | declared value | median smallest $\nu_j$ | floor binds |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `categorical_ltmle` | 0.9461 | 0.9371 | 0.0059 to 0.0123 | 0.0001 | 63.5 | 0.0% |
| `cde_z0` | 0.9543 | 0.9382 | 0.0120 to 0.0204 | 0.0001 | 19.2 | 0.1% |
| `strata` | 0.9545 | 0.9450 | 0.0063 to 0.0130 | 0.0001 | 8.7 | 11.5% |
| `categorical_ltmle_n8000` | 0.9496 | 0.9479 | 0.0005 to 0.0034 | none | 257.6 | 0.0% |
| `cde_z0_n8000` | 0.9441 | 0.9411 | 0.0013 to 0.0050 | none | 76.5 | 0.0% |
| `strata_n8000` | 0.9516 | 0.9480 | 0.0018 to 0.0057 | none | 34.6 | 0.0% |
| `categorical_ltmle_crossfit` | 0.9502 | 0.9416 | 0.0055 to 0.0120 | none | 62.7 | 0.0% |
| `crossfit_strata` | 0.9620 | 0.9536 | 0.0054 to 0.0118 | none | 8.5 | 12.9% |
| `six_regimens` | 0.9482 | 0.9400 | 0.0054 to 0.0116 | none | 45.7 | 0.0% |
| `ordinary_binary` | 0.9496 | 0.9486 | 0.0002 to 0.0023 | none | 239.7 | 0.0% |
| `point_msm` | 0.9509 | 0.9480 | 0.0013 to 0.0048 | none | 106.1 | 0.0% |

**The paired gains gate nothing.** Each one is a reported reading beside the value declared for
it, and a gain below its declared value would license or block nothing. Every 99% gain interval
lies above zero, and each red cell's lower limit clears its declared $10^{-4}$. The `t` band
reaches nominal coverage on the three red shapes, where the default band of the same fits covers
0.9371, 0.9382 and 0.9450. Their 99% intervals at 5,600 replications, 0.9283 to 0.9452,
0.9295 to 0.9462 and 0.9367 to 0.9526, all lie inside $[0.92, 0.98]$, and the `strata` interval
includes 0.95. These controls are not the default band's registered cells and license nothing:
by owner decision the red verdicts stand on the registered 2,400-replication cells of
`default-simultaneous-bands`. At $n = 8000$ the gain shrinks toward zero, as the large-df expansion
$c_j/c - 1 \approx (c^2 + 1)/(4\nu_j)$ predicts. `tests/unit/test_band_t_reference_design.py`
rebuilds the table from `properties.csv` and `t-band-replicates.csv`.

The highest `t` band coverage is `crossfit_strata` at 0.9620, whose 99% upper limit stays below
the 0.98 ceiling. No replication of a licensing cell held a constant arm cell except two of
`crossfit_strata`, which count as not covered.

The diagnostic arms bracket the declared rule and state no verdict. The two common-df arms cover
at or above the `t` band on every cell, and above 0.99 on both stratified shapes at
$n = 2000$, where one sparse stratum sets the df of every estimate. The moment-matched and
model-assisted df arms cover at or below the `t` band on every cell.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | primary replications |
| `n` | 1000 | observations per primary replication |
| `independent_tests_total` | 7 | primary tests against truth |
| `independent_tests_passed` | 7 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 33 | repeated-sampling property cells |
| `property_cells_passed` | 33 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0292 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9500 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.9444 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0936 | highest bootstrap primary SE-ratio endpoint |
| `properties[simultaneous_coverage/categorical_ltmle__t_band]:coverage` | 0.9461 | joint coverage of the categorical LTMLE `t` band |
| `properties[simultaneous_coverage/categorical_ltmle__t_band]:coverage_ci_lower` | 0.9378 | its 99% lower endpoint |
| `properties[simultaneous_coverage/categorical_ltmle__normal_reference_control]:coverage` | 0.9371 | joint coverage of the default band of the same fits |
| `properties[simultaneous_coverage/categorical_ltmle__normal_reference_control]:coverage_gain_ci_lower` | 0.0059 | 99% lower endpoint of the paired gain, reported only |
| `properties[simultaneous_coverage/cde_z0__t_band]:coverage` | 0.9543 | joint coverage of the controlled direct-effect `t` band at $z = 0$ |
| `properties[simultaneous_coverage/cde_z0__t_band]:coverage_ci_lower` | 0.9466 | its 99% lower endpoint |
| `properties[simultaneous_coverage/cde_z0__normal_reference_control]:coverage` | 0.9382 | joint coverage of the default band of the same fits |
| `properties[simultaneous_coverage/cde_z0__normal_reference_control]:coverage_gain_ci_lower` | 0.0120 | 99% lower endpoint of the paired gain, reported only |
| `properties[simultaneous_coverage/strata__t_band]:coverage` | 0.9545 | joint coverage of the in-sample stratified `t` band |
| `properties[simultaneous_coverage/strata__t_band]:coverage_ci_lower` | 0.9468 | its 99% lower endpoint |
| `properties[simultaneous_coverage/strata__normal_reference_control]:coverage` | 0.9450 | joint coverage of the default band of the same fits |
| `properties[simultaneous_coverage/strata__normal_reference_control]:coverage_gain_ci_lower` | 0.0063 | 99% lower endpoint of the paired gain, reported only |
| `properties[simultaneous_coverage/crossfit_strata__t_band]:coverage` | 0.9620 | the highest `t` band coverage, cross-fitted strata |
| `properties[simultaneous_coverage/crossfit_strata__t_band]:coverage_ci_upper` | 0.9682 | its 99% upper endpoint |
| `properties[simultaneous_coverage/cde_z0_n8000__t_band]:coverage` | 0.9441 | the lowest `t` band coverage, controlled direct effect at $n = 8000$ |
| `properties[simultaneous_coverage/cde_z0_n8000__t_band]:coverage_ci_lower` | 0.9357 | its 99% lower endpoint |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration and joint-coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration and joint-coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | largest supported type-I rate. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:subject_failure_bound` | 0.0100 | share of `cleverly` fits a cell may lose |
| `margin:subject_failures_allowed` | 10 | `cleverly` fits a cell may lose |

## Limitations

| limitation | what it means for use |
| --- | --- |
| Each cell is measured at one law and one or two sizes | A licence covers the whole key of its support row, as a default-band licence does. A law, size or learner the study did not run is not measured |
| Plain fits only | No cell is clustered, weighted or repeated. A `t` band of any of those shapes reads experimental on every row |
| Joint coverage only | The gate reads the joint coverage of the band. Over-coverage of a sparse coordinate can offset under-coverage elsewhere, and per-coordinate tails are recorded, not gated |
| The paired gains are reported, not gated | The study does not show as a gate that the `t` reference widens the band where the default band is red. The deterministic widening check and the reported gains are that evidence |
| The default band is unchanged | The three red default-band cells stay red and stay owned. Their normal-reference controls here read inside $[0.92, 0.98]$ at 5,600 replications (`strata` 0.9367 to 0.9526, which includes 0.95), but they are not the registered cells and license nothing; by owner decision the red verdicts stand on the registered 2,400-replication cells. A fit that keeps the default band on those rows reads experimental |
| No comparator | No pinned implementation publishes a band with a per-estimate t reference |

## Reproduction

The [fixture](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/band_t_reference/regenerate.py)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/band_t_reference/manifest.json)
records the seeds, margins, cells, source hashes and result hashes. The run used commit
`a11bcf62`.
