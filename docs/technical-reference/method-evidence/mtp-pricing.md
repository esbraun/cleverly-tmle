# MTP pricing

This study validates a longitudinal modified treatment policy on a price at 2, 3 and 5 nodes. The
declared support of each price depends on the node's engagement. `LTMLE` and `SDR` read each plan on
the classifier route with an exact ratio, and the property cells read learned ratios. The
[modified treatment policies at a node](../longitudinal-tmle.md#modified-treatment-policies-at-a-node)
section states the estimator and its conditions. The
[sequential pricing tutorial](../../examples/sequential-pricing.ipynb) fits law P at 3 nodes with
`LTMLE` on one draw.

**Status of this page.** The study is registered, and its Python phases and its `lmtp` phase have
run. The study record sets `reference="lmtp"`, so the page publishes a paired row for each estimand.
The publication policy is `reporting`.
A red cell stays red on this page and in the [red-cell ledger](red-cells.md), with the owner row
that names it.

| part | what it is | status |
| --- | --- | --- |
| Python property evidence | exact laws, interval calibration with two controls, one double-robustness pair and two necessity pairs | the tables below give the verdicts. It decides whether the method ships |
| `lmtp` comparator runs | the pinned `lmtp` 1.5.4 fits the same samples, and the study pairs the results | 30 of 30 paired rows pass. The 30 exactness witnesses pass the 1e-6 gate |

## What was compared

The primary phase draws 1,000 samples of 5,000 rows for each of six scenarios.

| scenario | law and plans | pairing |
| --- | --- | --- |
| `pricing_ltmle_tau2`, `_tau3`, `_tau5` | law P at 2, 3 and 5 nodes, `LTMLE`, plans `natural_prices`, `price_up` and `price_down`. Each node reads the exact ratio on the classifier route | exact under a gap of $10^{-6}$ |
| `pricing_sdr_tau2`, `_tau3`, `_tau5` | the same draws, `SDR` | exact under a gap of $10^{-6}$ |

Law P has a price at each node. The declared support of the price is $[4 + 2 L_t, 8 + 2 L_t]$, where
$L_t$ is the engagement at the node. Engaged customers get and tolerate higher prices, and a high price
lowers the next engagement. A truncated normal draws each price given its history, and a Bernoulli law
draws the outcome. The revenue cell multiplies the outcome by the last price.

| plan | map | what it tests |
| --- | --- | --- |
| `natural_prices` | no change | the reference value |
| `price_up` | scale by 1.12, hold above $u_t / 1.12$ | the upper boundary binds |
| `price_down` | hold below $l_t / 0.85$, scale by 0.85 above | the lower boundary binds, with two pieces |

| setting | `cleverly` | R `lmtp` |
| --- | --- | --- |
| datasets | 1,000 samples of 5,000 rows per scenario | the identical rows |
| density ratio | the exact ratio, written beside the sample | the same ratio array, read from the replicate data |
| node regressions | a quasibinomial GLM at the last node and least squares at the others | `SL.glm` on the same columns |
| intervals | pointwise 95% Wald from the influence curve | the same, from `lmtp`'s influence curve |

The truths at 2 nodes come from quadrature split at every break of the integrand. The truths at 3 and
5 nodes come from Monte Carlo draws of Definition 1 of Díaz, Williams, Hoffman and Schenck (2023),
pinned with their Monte Carlo standard error (`tests/studies/mtp_pricing_common.py`).

The property cells draw the mean purchase under `price_up` at n = 2,000. They read a correctly
specified final model, except in the pairs. The learned ratio is one of two readings. The learned
density is a spline hazard with 320 bins on the position relative to the declared support. The
screened classifier is a quadratic logistic classifier, chosen by a screen that the study declares
before the pilot. The comparator adapters are `tests/canonical/lmtp_mtp_adapter.R` and
`tests/canonical/lmtp_sdr_adapter.R`.

The first run of the primary phase used n = 2,000. It read the four `tau = 5` `price_down` rows
below the coverage that the exact ratio supports: coverage 0.903 to 0.912, an SE ratio of 0.93, and a
bias under 0.04 standard deviations. A probe on fresh draws read an SE ratio of 0.887 at n = 2,000
and 0.968 at n = 4,000. The influence-curve variance runs short at the smaller size. The primary
therefore moved to n = 5,000, and the property cells kept their declared sizes.
`tests/studies/canonical_mtp_pricing.py` records the declaration.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2-node pricing law with engagement feedback on the declared price support, LTMLE | `ate_regimen[price_down vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 0.85, holding a price within 15% of the support's lower end" against "leave every observed price unchanged" | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000818 to 0.000549 | 0.9590 | 0.9772 | pass |
| 2-node pricing law with engagement feedback on the declared price support, LTMLE | `ate_regimen[price_down vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 0.85, holding a price within 15% of the support's lower end" against "leave every observed price unchanged" | R `lmtp` | -0.000818 to 0.000549 | 0.9590 | 0.9772 | pass |
| 2-node pricing law with engagement feedback on the declared price support, LTMLE | `ate_regimen[price_up vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 1.12, holding a price within 12% of the support's upper end" against "leave every observed price unchanged" | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000628 to 0.000542 | 0.9570 | 1.0186 | pass |
| 2-node pricing law with engagement feedback on the declared price support, LTMLE | `ate_regimen[price_up vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 1.12, holding a price within 12% of the support's upper end" against "leave every observed price unchanged" | R `lmtp` | -0.000628 to 0.000542 | 0.9570 | 1.0186 | pass |
| 2-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[natural_prices]` | mean outcome under the plan leave every observed price unchanged | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000640 to 0.000410 | 0.9410 | 0.9784 | pass |
| 2-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[natural_prices]` | mean outcome under the plan leave every observed price unchanged | R `lmtp` | -0.000640 to 0.000410 | 0.9410 | 0.9784 | pass |
| 2-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[price_down]` | mean outcome under the plan multiply each price by 0.85, holding a price within 15% of the support's lower end | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.0010 to 0.000530 | 0.9460 | 0.9678 | pass |
| 2-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[price_down]` | mean outcome under the plan multiply each price by 0.85, holding a price within 15% of the support's lower end | R `lmtp` | -0.0010 to 0.000530 | 0.9460 | 0.9678 | pass |
| 2-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[price_up]` | mean outcome under the plan multiply each price by 1.12, holding a price within 12% of the support's upper end | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000980 to 0.000664 | 0.9490 | 1.0166 | pass |
| 2-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[price_up]` | mean outcome under the plan multiply each price by 1.12, holding a price within 12% of the support's upper end | R `lmtp` | -0.000980 to 0.000664 | 0.9490 | 1.0166 | pass |
| 3-node pricing law with engagement feedback on the declared price support, LTMLE | `ate_regimen[price_down vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 0.85, holding a price within 15% of the support's lower end" against "leave every observed price unchanged" | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.0015 to 0.000458 | 0.9360 | 0.9543 | pass |
| 3-node pricing law with engagement feedback on the declared price support, LTMLE | `ate_regimen[price_down vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 0.85, holding a price within 15% of the support's lower end" against "leave every observed price unchanged" | R `lmtp` | -0.0015 to 0.000458 | 0.9360 | 0.9543 | pass |
| 3-node pricing law with engagement feedback on the declared price support, LTMLE | `ate_regimen[price_up vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 1.12, holding a price within 12% of the support's upper end" against "leave every observed price unchanged" | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000488 to 0.0013 | 0.9460 | 0.9542 | pass |
| 3-node pricing law with engagement feedback on the declared price support, LTMLE | `ate_regimen[price_up vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 1.12, holding a price within 12% of the support's upper end" against "leave every observed price unchanged" | R `lmtp` | -0.000488 to 0.0013 | 0.9460 | 0.9542 | pass |
| 3-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[natural_prices]` | mean outcome under the plan leave every observed price unchanged | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000316 to 0.000682 | 0.9420 | 1.0265 | pass |
| 3-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[natural_prices]` | mean outcome under the plan leave every observed price unchanged | R `lmtp` | -0.000316 to 0.000682 | 0.9420 | 1.0265 | pass |
| 3-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[price_down]` | mean outcome under the plan multiply each price by 0.85, holding a price within 15% of the support's lower end | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.0013 to 0.000663 | 0.9470 | 0.9817 | pass |
| 3-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[price_down]` | mean outcome under the plan multiply each price by 0.85, holding a price within 15% of the support's lower end | R `lmtp` | -0.0013 to 0.000663 | 0.9470 | 0.9817 | pass |
| 3-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[price_up]` | mean outcome under the plan multiply each price by 1.12, holding a price within 12% of the support's upper end | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000497 to 0.0016 | 0.9420 | 0.9622 | pass |
| 3-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[price_up]` | mean outcome under the plan multiply each price by 1.12, holding a price within 12% of the support's upper end | R `lmtp` | -0.000497 to 0.0016 | 0.9420 | 0.9622 | pass |
| 5-node pricing law with engagement feedback on the declared price support, LTMLE | `ate_regimen[price_down vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 0.85, holding a price within 15% of the support's lower end" against "leave every observed price unchanged" | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000974 to 0.0024 | 0.9160 | 0.9310 | **fail** |
| 5-node pricing law with engagement feedback on the declared price support, LTMLE | `ate_regimen[price_down vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 0.85, holding a price within 15% of the support's lower end" against "leave every observed price unchanged" | R `lmtp` | -0.000974 to 0.0024 | 0.9160 | 0.9310 | **fail** |
| 5-node pricing law with engagement feedback on the declared price support, LTMLE | `ate_regimen[price_up vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 1.12, holding a price within 12% of the support's upper end" against "leave every observed price unchanged" | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000889 to 0.0019 | 0.9380 | 0.9841 | pass |
| 5-node pricing law with engagement feedback on the declared price support, LTMLE | `ate_regimen[price_up vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 1.12, holding a price within 12% of the support's upper end" against "leave every observed price unchanged" | R `lmtp` | -0.000889 to 0.0019 | 0.9380 | 0.9841 | pass |
| 5-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[natural_prices]` | mean outcome under the plan leave every observed price unchanged | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000536 to 0.000467 | 0.9550 | 1.0208 | pass |
| 5-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[natural_prices]` | mean outcome under the plan leave every observed price unchanged | R `lmtp` | -0.000536 to 0.000467 | 0.9550 | 1.0208 | pass |
| 5-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[price_down]` | mean outcome under the plan multiply each price by 0.85, holding a price within 15% of the support's lower end | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000992 to 0.0023 | 0.9170 | 0.9479 | **fail** |
| 5-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[price_down]` | mean outcome under the plan multiply each price by 0.85, holding a price within 15% of the support's lower end | R `lmtp` | -0.000992 to 0.0023 | 0.9170 | 0.9479 | **fail** |
| 5-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[price_up]` | mean outcome under the plan multiply each price by 1.12, holding a price within 12% of the support's upper end | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.0010 to 0.0020 | 0.9430 | 0.9885 | pass |
| 5-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[price_up]` | mean outcome under the plan multiply each price by 1.12, holding a price within 12% of the support's upper end | R `lmtp` | -0.0010 to 0.0020 | 0.9430 | 0.9885 | pass |
| 2-node pricing law with engagement feedback on the declared price support, SDR | `ate_regimen[price_down vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 0.85, holding a price within 15% of the support's lower end" against "leave every observed price unchanged" | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000819 to 0.000548 | 0.9570 | 0.9775 | pass |
| 2-node pricing law with engagement feedback on the declared price support, SDR | `ate_regimen[price_down vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 0.85, holding a price within 15% of the support's lower end" against "leave every observed price unchanged" | R `lmtp` | -0.000819 to 0.000548 | 0.9570 | 0.9775 | pass |
| 2-node pricing law with engagement feedback on the declared price support, SDR | `ate_regimen[price_up vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 1.12, holding a price within 12% of the support's upper end" against "leave every observed price unchanged" | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000628 to 0.000543 | 0.9580 | 1.0185 | pass |
| 2-node pricing law with engagement feedback on the declared price support, SDR | `ate_regimen[price_up vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 1.12, holding a price within 12% of the support's upper end" against "leave every observed price unchanged" | R `lmtp` | -0.000628 to 0.000543 | 0.9580 | 1.0185 | pass |
| 2-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[natural_prices]` | mean outcome under the plan leave every observed price unchanged | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000640 to 0.000410 | 0.9410 | 0.9784 | pass |
| 2-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[natural_prices]` | mean outcome under the plan leave every observed price unchanged | R `lmtp` | -0.000640 to 0.000410 | 0.9410 | 0.9784 | pass |
| 2-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[price_down]` | mean outcome under the plan multiply each price by 0.85, holding a price within 15% of the support's lower end | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.0010 to 0.000529 | 0.9470 | 0.9687 | pass |
| 2-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[price_down]` | mean outcome under the plan multiply each price by 0.85, holding a price within 15% of the support's lower end | R `lmtp` | -0.0010 to 0.000529 | 0.9470 | 0.9687 | pass |
| 2-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[price_up]` | mean outcome under the plan multiply each price by 1.12, holding a price within 12% of the support's upper end | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000980 to 0.000664 | 0.9490 | 1.0166 | pass |
| 2-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[price_up]` | mean outcome under the plan multiply each price by 1.12, holding a price within 12% of the support's upper end | R `lmtp` | -0.000980 to 0.000664 | 0.9490 | 1.0166 | pass |
| 3-node pricing law with engagement feedback on the declared price support, SDR | `ate_regimen[price_down vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 0.85, holding a price within 15% of the support's lower end" against "leave every observed price unchanged" | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.0015 to 0.000443 | 0.9320 | 0.9545 | pass |
| 3-node pricing law with engagement feedback on the declared price support, SDR | `ate_regimen[price_down vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 0.85, holding a price within 15% of the support's lower end" against "leave every observed price unchanged" | R `lmtp` | -0.0015 to 0.000443 | 0.9320 | 0.9545 | pass |
| 3-node pricing law with engagement feedback on the declared price support, SDR | `ate_regimen[price_up vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 1.12, holding a price within 12% of the support's upper end" against "leave every observed price unchanged" | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000499 to 0.0012 | 0.9470 | 0.9553 | pass |
| 3-node pricing law with engagement feedback on the declared price support, SDR | `ate_regimen[price_up vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 1.12, holding a price within 12% of the support's upper end" against "leave every observed price unchanged" | R `lmtp` | -0.000499 to 0.0012 | 0.9470 | 0.9553 | pass |
| 3-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[natural_prices]` | mean outcome under the plan leave every observed price unchanged | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000316 to 0.000682 | 0.9420 | 1.0265 | pass |
| 3-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[natural_prices]` | mean outcome under the plan leave every observed price unchanged | R `lmtp` | -0.000316 to 0.000682 | 0.9420 | 1.0265 | pass |
| 3-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[price_down]` | mean outcome under the plan multiply each price by 0.85, holding a price within 15% of the support's lower end | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.0013 to 0.000647 | 0.9490 | 0.9830 | pass |
| 3-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[price_down]` | mean outcome under the plan multiply each price by 0.85, holding a price within 15% of the support's lower end | R `lmtp` | -0.0013 to 0.000647 | 0.9490 | 0.9830 | pass |
| 3-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[price_up]` | mean outcome under the plan multiply each price by 1.12, holding a price within 12% of the support's upper end | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000507 to 0.0016 | 0.9400 | 0.9635 | pass |
| 3-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[price_up]` | mean outcome under the plan multiply each price by 1.12, holding a price within 12% of the support's upper end | R `lmtp` | -0.000507 to 0.0016 | 0.9400 | 0.9635 | pass |
| 5-node pricing law with engagement feedback on the declared price support, SDR | `ate_regimen[price_down vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 0.85, holding a price within 15% of the support's lower end" against "leave every observed price unchanged" | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000981 to 0.0024 | 0.9190 | 0.9339 | **fail** |
| 5-node pricing law with engagement feedback on the declared price support, SDR | `ate_regimen[price_down vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 0.85, holding a price within 15% of the support's lower end" against "leave every observed price unchanged" | R `lmtp` | -0.000981 to 0.0024 | 0.9190 | 0.9339 | **fail** |
| 5-node pricing law with engagement feedback on the declared price support, SDR | `ate_regimen[price_up vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 1.12, holding a price within 12% of the support's upper end" against "leave every observed price unchanged" | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000906 to 0.0019 | 0.9400 | 0.9840 | pass |
| 5-node pricing law with engagement feedback on the declared price support, SDR | `ate_regimen[price_up vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 1.12, holding a price within 12% of the support's upper end" against "leave every observed price unchanged" | R `lmtp` | -0.000906 to 0.0019 | 0.9400 | 0.9840 | pass |
| 5-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[natural_prices]` | mean outcome under the plan leave every observed price unchanged | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000536 to 0.000467 | 0.9550 | 1.0208 | pass |
| 5-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[natural_prices]` | mean outcome under the plan leave every observed price unchanged | R `lmtp` | -0.000536 to 0.000467 | 0.9550 | 1.0208 | pass |
| 5-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[price_down]` | mean outcome under the plan multiply each price by 0.85, holding a price within 15% of the support's lower end | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.000997 to 0.0023 | 0.9230 | 0.9518 | **fail** |
| 5-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[price_down]` | mean outcome under the plan multiply each price by 0.85, holding a price within 15% of the support's lower end | R `lmtp` | -0.000997 to 0.0023 | 0.9230 | 0.9518 | **fail** |
| 5-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[price_up]` | mean outcome under the plan multiply each price by 1.12, holding a price within 12% of the support's upper end | `cleverly` LTMLE and SDR under price policies that bind at a declared support | -0.0011 to 0.0020 | 0.9440 | 0.9888 | pass |
| 5-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[price_up]` | mean outcome under the plan multiply each price by 1.12, holding a price within 12% of the support's upper end | R `lmtp` | -0.0011 to 0.0020 | 0.9440 | 0.9888 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

Every scenario is an exactness witness. Each of the 30 estimands pairs 1,000 fits with `lmtp`. The
largest gap is 2.8e-08 in the estimate and 4.1e-11 in the standard error, against a gate of 1e-6
(`exactness.csv`). The study fails a witness above the gate. The primary `LTMLE` fits bound the
submodel at `alpha` = 1 - 1e-5, which is the prediction bound of `lmtp`. The four `price_down` rows
at tau 5 are red on both sides with the same numbers, so the shortfall is not an implementation
difference ([red-cell ledger](red-cells.md)).

<!-- generated: agreement -->
Comparator-backed: a paired comparison with `lmtp` on the same samples. Each row reports estimate equivalence within the declared margin, the RMSE ratio and the coverage difference, and names its comparison kind.

| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | comparison kind | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2-node pricing law with engagement feedback on the declared price support, LTMLE | `ate_regimen[price_down vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 0.85, holding a price within 15% of the support's lower end" against "leave every observed price unchanged" | 7.190e-10 | 5.722e-07 | 1.0000 | 0 | matching implementation | 1.225e-08 vs 0.0500 | equivalent |
| 2-node pricing law with engagement feedback on the declared price support, LTMLE | `ate_regimen[price_up vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 1.12, holding a price within 12% of the support's upper end" against "leave every observed price unchanged" | -8.215e-11 | 7.635e-08 | 1.0000 | 0 | matching implementation | 5.701e-09 vs 0.0500 | equivalent |
| 2-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[natural_prices]` | mean outcome under the plan leave every observed price unchanged | 2.061e-10 | 2.136e-07 | 1.0000 | 0 | matching implementation | 2.710e-08 vs 0.0500 | equivalent |
| 2-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[price_down]` | mean outcome under the plan multiply each price by 0.85, holding a price within 15% of the support's lower end | 9.251e-10 | 6.451e-07 | 1.0000 | 0 | matching implementation | 2.474e-08 vs 0.0500 | equivalent |
| 2-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[price_up]` | mean outcome under the plan multiply each price by 1.12, holding a price within 12% of the support's upper end | 1.240e-10 | 8.205e-08 | 1.0000 | 0 | matching implementation | 2.244e-09 vs 0.0500 | equivalent |
| 3-node pricing law with engagement feedback on the declared price support, LTMLE | `ate_regimen[price_down vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 0.85, holding a price within 15% of the support's lower end" against "leave every observed price unchanged" | -9.665e-11 | 5.412e-08 | 1.0000 | 0 | matching implementation | 2.952e-09 vs 0.0500 | equivalent |
| 3-node pricing law with engagement feedback on the declared price support, LTMLE | `ate_regimen[price_up vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 1.12, holding a price within 12% of the support's upper end" against "leave every observed price unchanged" | -5.115e-11 | 3.192e-08 | 1.0000 | 0 | matching implementation | 1.259e-08 vs 0.0500 | equivalent |
| 3-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[natural_prices]` | mean outcome under the plan leave every observed price unchanged | 1.492e-10 | 1.626e-07 | 1.0000 | 0 | matching implementation | 3.803e-09 vs 0.0500 | equivalent |
| 3-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[price_down]` | mean outcome under the plan multiply each price by 0.85, holding a price within 15% of the support's lower end | 5.250e-11 | 2.872e-08 | 1.0000 | 0 | matching implementation | 1.965e-09 vs 0.0500 | equivalent |
| 3-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[price_up]` | mean outcome under the plan multiply each price by 1.12, holding a price within 12% of the support's upper end | 9.801e-11 | 5.015e-08 | 1.0000 | 0 | matching implementation | 1.854e-09 vs 0.0500 | equivalent |
| 5-node pricing law with engagement feedback on the declared price support, LTMLE | `ate_regimen[price_down vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 0.85, holding a price within 15% of the support's lower end" against "leave every observed price unchanged" | 1.215e-09 | 3.944e-07 | 1.0000 | 0 | matching implementation | 1.051e-08 vs 0.0500 | equivalent |
| 5-node pricing law with engagement feedback on the declared price support, LTMLE | `ate_regimen[price_up vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 1.12, holding a price within 12% of the support's upper end" against "leave every observed price unchanged" | -4.382e-11 | 1.681e-08 | 1.0000 | 0 | matching implementation | 1.514e-09 vs 0.0500 | equivalent |
| 5-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[natural_prices]` | mean outcome under the plan leave every observed price unchanged | 1.170e-10 | 1.269e-07 | 1.0000 | 0 | matching implementation | 3.532e-09 vs 0.0500 | equivalent |
| 5-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[price_down]` | mean outcome under the plan multiply each price by 0.85, holding a price within 15% of the support's lower end | 1.332e-09 | 4.368e-07 | 1.0000 | 0 | matching implementation | 1.110e-08 vs 0.0500 | equivalent |
| 5-node pricing law with engagement feedback on the declared price support, LTMLE | `ey_regimen[price_up]` | mean outcome under the plan multiply each price by 1.12, holding a price within 12% of the support's upper end | 7.319e-11 | 2.582e-08 | 1.0000 | 0 | matching implementation | 8.196e-10 vs 0.0500 | equivalent |
| 2-node pricing law with engagement feedback on the declared price support, SDR | `ate_regimen[price_down vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 0.85, holding a price within 15% of the support's lower end" against "leave every observed price unchanged" | 9.517e-17 | 7.575e-14 | 1.0000 | 0 | matching implementation | 1.402e-13 vs 0.0500 | equivalent |
| 2-node pricing law with engagement feedback on the declared price support, SDR | `ate_regimen[price_up vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 1.12, holding a price within 12% of the support's upper end" against "leave every observed price unchanged" | -3.912e-17 | 3.635e-14 | 1.0000 | 0 | matching implementation | 1.327e-13 vs 0.0500 | equivalent |
| 2-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[natural_prices]` | mean outcome under the plan leave every observed price unchanged | 0 | 0 | 1 | 0 | matching implementation | 1.532e-14 vs 0.0500 | equivalent |
| 2-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[price_down]` | mean outcome under the plan multiply each price by 0.85, holding a price within 15% of the support's lower end | 9.293e-17 | 6.485e-14 | 1.0000 | 0 | matching implementation | 1.809e-13 vs 0.0500 | equivalent |
| 2-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[price_up]` | mean outcome under the plan multiply each price by 1.12, holding a price within 12% of the support's upper end | -3.386e-17 | 2.241e-14 | 1.0000 | 0 | matching implementation | 3.952e-14 vs 0.0500 | equivalent |
| 3-node pricing law with engagement feedback on the declared price support, SDR | `ate_regimen[price_down vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 0.85, holding a price within 15% of the support's lower end" against "leave every observed price unchanged" | -7.367e-16 | 4.123e-13 | 1.0000 | 0 | matching implementation | 1.669e-13 vs 0.0500 | equivalent |
| 3-node pricing law with engagement feedback on the declared price support, SDR | `ate_regimen[price_up vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 1.12, holding a price within 12% of the support's upper end" against "leave every observed price unchanged" | -1.957e-16 | 1.222e-13 | 1.0000 | 0 | matching implementation | 6.884e-14 vs 0.0500 | equivalent |
| 3-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[natural_prices]` | mean outcome under the plan leave every observed price unchanged | 0 | 0 | 1 | 0 | matching implementation | 4.441e-16 vs 0.0500 | equivalent |
| 3-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[price_down]` | mean outcome under the plan multiply each price by 0.85, holding a price within 15% of the support's lower end | -7.365e-16 | 4.032e-13 | 1.0000 | 0 | matching implementation | 3.361e-13 vs 0.0500 | equivalent |
| 3-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[price_up]` | mean outcome under the plan multiply each price by 1.12, holding a price within 12% of the support's upper end | -1.864e-16 | 9.548e-14 | 1.0000 | 0 | matching implementation | 7.761e-14 vs 0.0500 | equivalent |
| 5-node pricing law with engagement feedback on the declared price support, SDR | `ate_regimen[price_down vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 0.85, holding a price within 15% of the support's lower end" against "leave every observed price unchanged" | -1.524e-15 | 4.931e-13 | 1.0000 | 0 | matching implementation | 1.236e-13 vs 0.0500 | equivalent |
| 5-node pricing law with engagement feedback on the declared price support, SDR | `ate_regimen[price_up vs natural_prices]` | difference in mean outcome between the plans "multiply each price by 1.12, holding a price within 12% of the support's upper end" against "leave every observed price unchanged" | -3.887e-16 | 1.487e-13 | 1.0000 | 0 | matching implementation | 1.710e-13 vs 0.0500 | equivalent |
| 5-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[natural_prices]` | mean outcome under the plan leave every observed price unchanged | 0 | 0 | 1 | 0 | matching implementation | 6.661e-16 vs 0.0500 | equivalent |
| 5-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[price_down]` | mean outcome under the plan multiply each price by 0.85, holding a price within 15% of the support's lower end | -1.517e-15 | 4.963e-13 | 1.0000 | 0 | matching implementation | 1.423e-13 vs 0.0500 | equivalent |
| 5-node pricing law with engagement feedback on the declared price support, SDR | `ey_regimen[price_up]` | mean outcome under the plan multiply each price by 1.12, holding a price within 12% of the support's upper end | -3.845e-16 | 1.354e-13 | 1.0000 | 0 | matching implementation | 1.592e-13 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `up_tau3__both_wrong_control` | control | the mean purchase under the 12% price rise at 3 nodes: the outcome is misspecified and the density ratio is one | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0951 to 0.0970, margin 0.0028, SE ratio 0.8773 | pass |
| `double_robustness` | `up_tau3__outcome_wrong_learned_classifier` | positive | the mean purchase under the 12% price rise at 3 nodes: the outcome is misspecified and the declared classifier estimates the density ratio | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000398 to 0.0029, margin 0.0050, SE ratio 1.0022 | pass |
| `interval_calibration` | `crossfit_ltmle_learned_classifier_tau3__correctly_specified` | positive | the mean purchase under the 12% price rise at 3 nodes, LTMLE cross-fitted over five folds: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9437 to 0.9570, SE ratio 0.9849 to 1.0285 | pass |
| `interval_calibration` | `crossfit_ltmle_learned_classifier_tau3__noise_control` | control | the mean purchase under the 12% price rise at 3 nodes, LTMLE cross-fitted over five folds: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8232 to 0.8458, SE ratio 0.6921 to 0.7226 | pass |
| `interval_calibration` | `crossfit_ltmle_learned_classifier_tau3__shrunken_se_control` | control | the mean purchase under the 12% price rise at 3 nodes, LTMLE cross-fitted over five folds: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8180 to 0.8410, SE ratio 0.6902 to 0.7197 | pass |
| `interval_calibration` | `crossfit_sdr_learned_classifier_tau3__correctly_specified` | positive | the mean purchase under the 12% price rise at 3 nodes, SDR cross-fitted over five folds: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9458 to 0.9589, SE ratio 0.9878 to 1.0317 | pass |
| `interval_calibration` | `crossfit_sdr_learned_classifier_tau3__noise_control` | control | the mean purchase under the 12% price rise at 3 nodes, SDR cross-fitted over five folds: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8233 to 0.8460, SE ratio 0.6943 to 0.7249 | pass |
| `interval_calibration` | `crossfit_sdr_learned_classifier_tau3__shrunken_se_control` | control | the mean purchase under the 12% price rise at 3 nodes, SDR cross-fitted over five folds: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8200 to 0.8429, SE ratio 0.6914 to 0.7224 | pass |
| `interval_calibration` | `revenue_learned_classifier_tau3__correctly_specified` | positive | the mean revenue under the 12% price rise at 3 nodes, learned classifier: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9393 to 0.9531, SE ratio 0.9814 to 1.0242 | pass |
| `interval_calibration` | `revenue_learned_classifier_tau3__noise_control` | control | the mean revenue under the 12% price rise at 3 nodes, learned classifier: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8179 to 0.8408, SE ratio 0.6894 to 0.7188 | pass |
| `interval_calibration` | `revenue_learned_classifier_tau3__shrunken_se_control` | control | the mean revenue under the 12% price rise at 3 nodes, learned classifier: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8223 to 0.8450, SE ratio 0.6865 to 0.7174 | pass |
| `interval_calibration` | `up_learned_classifier_tau2__correctly_specified` | positive | the mean purchase under the 12% price rise at 2 nodes, learned classifier: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9401 to 0.9538, SE ratio 0.9688 to 1.0111 | pass |
| `interval_calibration` | `up_learned_classifier_tau2__noise_control` | control | the mean purchase under the 12% price rise at 2 nodes, learned classifier: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8136 to 0.8368, SE ratio 0.6783 to 0.7074 | pass |
| `interval_calibration` | `up_learned_classifier_tau2__shrunken_se_control` | control | the mean purchase under the 12% price rise at 2 nodes, learned classifier: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8108 to 0.8341, SE ratio 0.6781 to 0.7080 | pass |
| `interval_calibration` | `up_learned_classifier_tau3__correctly_specified` | positive | the mean purchase under the 12% price rise at 3 nodes, learned classifier: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9401 to 0.9538, SE ratio 0.9695 to 1.0120 | pass |
| `interval_calibration` | `up_learned_classifier_tau3__noise_control` | control | the mean purchase under the 12% price rise at 3 nodes, learned classifier: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8230 to 0.8457, SE ratio 0.6953 to 0.7249 | pass |
| `interval_calibration` | `up_learned_classifier_tau3__shrunken_se_control` | control | the mean purchase under the 12% price rise at 3 nodes, learned classifier: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8116 to 0.8349, SE ratio 0.6783 to 0.7084 | pass |
| `interval_calibration` | `up_learned_classifier_tau5__correctly_specified` | positive | the mean purchase under the 12% price rise at 5 nodes, learned classifier: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9261 to 0.9413, SE ratio 0.9444 to 0.9878 | pass |
| `interval_calibration` | `up_learned_classifier_tau5__noise_control` | control | the mean purchase under the 12% price rise at 5 nodes, learned classifier: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8049 to 0.8285, SE ratio 0.6764 to 0.7060 | pass |
| `interval_calibration` | `up_learned_classifier_tau5__shrunken_se_control` | control | the mean purchase under the 12% price rise at 5 nodes, learned classifier: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7944 to 0.8185, SE ratio 0.6611 to 0.6913 | pass |
| `interval_calibration` | `up_learned_density_tau2__correctly_specified` | positive | the mean purchase under the 12% price rise at 2 nodes, learned density: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9530 to 0.9747, SE ratio 1.0265 to 1.1162 | **fail** |
| `interval_calibration` | `up_learned_density_tau2__noise_control` | control | the mean purchase under the 12% price rise at 2 nodes, learned density: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8363 to 0.8770, SE ratio 0.7238 to 0.7845 | pass |
| `interval_calibration` | `up_learned_density_tau2__shrunken_se_control` | control | the mean purchase under the 12% price rise at 2 nodes, learned density: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8257 to 0.8676, SE ratio 0.7187 to 0.7814 | pass |
| `interval_calibration` | `up_learned_density_tau3__correctly_specified` | positive | the mean purchase under the 12% price rise at 3 nodes, learned density: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9473 to 0.9704, SE ratio 1.0493 to 1.1408 | **fail** |
| `interval_calibration` | `up_learned_density_tau3__noise_control` | control | the mean purchase under the 12% price rise at 3 nodes, learned density: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8437 to 0.8836, SE ratio 0.7376 to 0.8016 | pass |
| `interval_calibration` | `up_learned_density_tau3__shrunken_se_control` | control | the mean purchase under the 12% price rise at 3 nodes, learned density: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8415 to 0.8817, SE ratio 0.7352 to 0.7988 | pass |
| `interval_calibration` | `up_learned_density_tau5__correctly_specified` | positive | the mean purchase under the 12% price rise at 5 nodes, learned density: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9485 to 0.9713, SE ratio 1.0428 to 1.1323 | **fail** |
| `interval_calibration` | `up_learned_density_tau5__noise_control` | control | the mean purchase under the 12% price rise at 5 nodes, learned density: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8162 to 0.8591, SE ratio 0.7238 to 0.7847 | pass |
| `interval_calibration` | `up_learned_density_tau5__shrunken_se_control` | control | the mean purchase under the 12% price rise at 5 nodes, learned density: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8431 to 0.8831, SE ratio 0.7310 to 0.7926 | pass |
| `interval_calibration` | `up_oracle_tau2__correctly_specified` | positive | the mean purchase under the 12% price rise at 2 nodes, exact ratio: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9389 to 0.9527, SE ratio 0.9810 to 1.0232 | pass |
| `interval_calibration` | `up_oracle_tau2__noise_control` | control | the mean purchase under the 12% price rise at 2 nodes, exact ratio: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8270 to 0.8495, SE ratio 0.6976 to 0.7292 | pass |
| `interval_calibration` | `up_oracle_tau2__shrunken_se_control` | control | the mean purchase under the 12% price rise at 2 nodes, exact ratio: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8129 to 0.8361, SE ratio 0.6863 to 0.7165 | pass |
| `interval_calibration` | `up_oracle_tau3__correctly_specified` | positive | the mean purchase under the 12% price rise at 3 nodes, exact ratio: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9404 to 0.9540, SE ratio 0.9759 to 1.0183 | pass |
| `interval_calibration` | `up_oracle_tau3__noise_control` | control | the mean purchase under the 12% price rise at 3 nodes, exact ratio: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8244 to 0.8471, SE ratio 0.6944 to 0.7253 | pass |
| `interval_calibration` | `up_oracle_tau3__shrunken_se_control` | control | the mean purchase under the 12% price rise at 3 nodes, exact ratio: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8096 to 0.8330, SE ratio 0.6830 to 0.7133 | pass |
| `interval_calibration` | `up_oracle_tau5__correctly_specified` | positive | the mean purchase under the 12% price rise at 5 nodes, exact ratio: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9301 to 0.9449, SE ratio 0.9671 to 1.0092 | pass |
| `interval_calibration` | `up_oracle_tau5__noise_control` | control | the mean purchase under the 12% price rise at 5 nodes, exact ratio: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8186 to 0.8415, SE ratio 0.6910 to 0.7215 | pass |
| `interval_calibration` | `up_oracle_tau5__shrunken_se_control` | control | the mean purchase under the 12% price rise at 5 nodes, exact ratio: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8044 to 0.8280, SE ratio 0.6768 to 0.7067 | pass |
| `ratio_necessity` | `up_tau3__learned_classifier` | positive | the mean purchase under the 12% price rise at 3 nodes: the declared classifier estimates the density ratio, with the outcome misspecified | bias interval inside the equivalence margin | bias -0.000013 to 0.0032, margin 0.0049 | pass |
| `ratio_necessity` | `up_tau3__unit_ratio_control` | control | the mean purchase under the 12% price rise at 3 nodes: the same fit replaces the density ratio with one, with the outcome misspecified | bias interval must fall entirely outside the margin | bias 0.0957 to 0.0975, margin 0.0028 | pass |
| `ratio_necessity` | `up_tau5__learned_classifier` | positive | the mean purchase under the 12% price rise at 5 nodes: the declared classifier estimates the density ratio, with the outcome misspecified | bias interval inside the equivalence margin | bias -0.0011 to 0.0023, margin 0.0053 | pass |
| `ratio_necessity` | `up_tau5__unit_ratio_control` | control | the mean purchase under the 12% price rise at 5 nodes: the same fit replaces the density ratio with one, with the outcome misspecified | bias interval must fall entirely outside the margin | bias 0.1273 to 0.1286, margin 0.0020 | pass |
<!-- /generated -->

The truths come from the exact law and its closed forms (`tests/studies/mtp_pricing_common.py`).

| family | cells | replications and size |
| --- | --- | --- |
| `interval_calibration` | twelve positives: the exact ratio, the learned density and the screened classifier at 2, 3 and 5 nodes, the revenue outcome at 3 nodes, and `LTMLE` and `SDR` cross-fitted over five folds at 3 nodes. Each positive has a shrunken-SE control and a noise control | the replications that the pooled rule sets, at n = 2,000 |
| `double_robustness` | 3 nodes, wrong outcome GLM. The screened classifier against the wrong GLM with a ratio of one | 1,000 pairs at n = 2,000 |
| `ratio_necessity` | 3 and 5 nodes, wrong outcome GLM. The screened classifier against a ratio of one | 1,000 pairs at n = 2,000 |

Each pair removes one term that can vanish at the truth, so each needs a nonzero witness. With a wrong
outcome regression the targeting step carries the whole correction, and a wrong ratio then moves the
estimate. The learned-density and classifier cells read a correct outcome model. They measure
interval calibration with a learned ratio and give no evidence that the ratio is consistent. The
double-robustness pair reads the ratio of the classifier alone.

The cross-fold pair `classifier_split` is not registered. Its pilot read a necessity power of 0.0002,
so the pair cannot tell a leak from no leak. In `tests/unit/test_classifier_ratio_controls.py`,
`test_t_c1_no_unit_trains_the_model_that_predicts_it` shows no leak under unit folds, and
`test_m_c1_folds_over_stacked_rows_leak` shows a leak under row folds.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | paired replications |
| `n` | 5000 | observations per paired replication |
| `independent_tests_passed` | 52 | truth tests passing |
| `independent_tests_total` | 60 | truth tests reported |
| `paired_tests_passed` | 30 | paired comparisons passing |
| `paired_tests_total` | 30 | paired comparisons reported |
| `property_cells_passed` | 39 | property cells passing |
| `property_cells_total` | 42 | property cells reported |
| `max_standardized_bias` | 0.0444 | largest primary standardized bias |
| `min_coverage` | 0.9160 | lowest primary coverage |
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
| `margin:necessity_displacement` | 0.1000 | minimum ratio necessity displacement |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 10 | the same bound, in primary replications |

## Limits

| limit | what it means for use |
| --- | --- |
| the four `tau = 5` `price_down` primary rows are experimental | the exact ratio gives coverage 0.916 to 0.923 and an SE ratio of 0.931 to 0.952 for `LTMLE` and `SDR` at the primary size. The package labels these fits experimental. The roadmap item `mtp-pricing-calibration` owns the question |
| the learned-density cells publish as `reporting` | the screen read SE ratios above 1.05: 1.052, 1.105 and 1.079 at 2, 3 and 5 nodes. The interval is conservative, with coverage 0.953 to 0.975. The row `mtp-pricing-learned-density` of the red-cell ledger owns it |
| the screened ratio learner is the quadratic logistic classifier | the study uses it at every node count, with `ratio="classifier"` |
| the cross-fold leak pair `classifier_split` is not registered | the pilot cannot discriminate. The fast tier shows the effect of the leak only |
| a continuous outcome needs `Targeting(q_bounds=(lower, upper))` | the revenue cell declares `q_bounds=(0, 10)`. The fit never clips an outcome |
| the comparison holds one set of node regressions | the study measures agreement only under the node regressions that "What was compared" names |
