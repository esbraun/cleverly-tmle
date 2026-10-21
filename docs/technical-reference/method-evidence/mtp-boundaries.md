# MTP boundaries

This study validates a longitudinal modified treatment policy on a continuous dose whose declared
support depends on the history. The policies bind at that support, change their Jacobian with the
dose, have two inverse branches, or send an interval of doses to a declared atom. `LTMLE` and
`SDR` read each policy at one fold on the density route. The
[modified treatment policies at a node](../longitudinal-tmle.md#modified-treatment-policies-at-a-node)
section states the estimator and its conditions.

**Status of this page.** The study is registered, and its Python phases and its `lmtp` phase have
run. The study record sets `reference="lmtp"`, so the page publishes a paired row for each estimand.
The publication policy is `reporting`.
A red cell stays red on this page and in the [red-cell ledger](red-cells.md), with the owner row
that names it.

| part | what it is | status |
| --- | --- | --- |
| Python property evidence | exact laws, interval calibration with two controls, and four necessity pairs | the tables below give the verdicts. It decides whether the method ships |
| `lmtp` comparator runs | the pinned `lmtp` 1.5.4 fits the same samples, and the study pairs the results | 24 of 24 paired rows pass. The 18 exactness witnesses pass the 1e-6 gate |

## What was compared

The primary phase draws 1,000 samples of 5,000 rows for each of four scenarios.

| scenario | law and plans | pairing |
| --- | --- | --- |
| `boundaries_ltmle` | law B, `LTMLE`, plans `natural` and `p1` to `p4`. Each node reads the oracle binned density, and the per-node ratio and the shifted values go beside the sample | exact under a gap of $10^{-6}$ |
| `boundaries_sdr` | the same draw as `boundaries_ltmle`, `SDR` | exact under a gap of $10^{-6}$ |
| `atom_ltmle` | law B', `LTMLE`, plans `natural` and `p5`. Node 2 reads the oracle mixed density: the exact atom probability, and the oracle hazard of the rows off the atom | margin-read. The outcome design of the package holds the atom indicator, which the shifted frame of `lmtp` cannot carry |
| `atom_sdr` | the same draw as `atom_ltmle`, `SDR` | margin-read, as above |

Law B has two nodes. Each dose lies on the support $[0, u]$ with $u = 6 + 2 W_2$, so the support
depends on a baseline covariate. A truncated normal draws each dose given its history. A Bernoulli
law draws the outcome. Law B' adds an atom at node 2: the dose equals the list price 3 with a
probability that depends on the history, and otherwise follows law B.

| plan | map | what it tests |
| --- | --- | --- |
| `natural` | no change | the reference value |
| `p1` | scale by 0.9 | a map that never binds at the support |
| `p2` | add 1 below $u - 1$, hold above | the upper boundary binds |
| `p3` | $u - D(u - a)$, with $D$ a smooth warp | one piece whose inverse slope runs from 0.63 to 2.5, so the Jacobian is not 1 |
| `p4` | add 1 below 3, subtract 1.5 above | two inverse branches |
| `p5` | $\max(3, 0.65 a)$ at node 2 on law B' | the interval below $3 / 0.65$ goes to the declared atom 3 |

| setting | `cleverly` | R `lmtp` |
| --- | --- | --- |
| datasets | 1,000 samples of 5,000 rows per scenario | the identical rows |
| density ratio | an oracle binned density with `ceil(320 (n / 2000)^(2/3))` bins (320 at n = 2,000), and the exact atom probability on law B' | the same ratio array, read from the replicate data |
| node regressions | a quasibinomial GLM at the last node and least squares at the first | `SL.glm` on the same columns |
| intervals | pointwise 95% Wald from the influence curve | the same, from `lmtp`'s influence curve |

The property rows read a correctly specified final model and a rich first-node basis. The `SDR`
property cells read a first-node learner that projects onto [0, 1]. The comparator adapters are
`tests/canonical/lmtp_mtp_adapter.R` and `tests/canonical/lmtp_sdr_adapter.R`. Both sides read the
columns `shift__<plan>__<node>__1` and `ratio__<plan>__<node>`, so neither side estimates a ratio
that the other does not see.

The first reading of the primary phase used n = 2,000. The two `boundaries_ltmle` rows of plan `p4`
read coverage 0.916 and 0.922 and an SE ratio of 0.925, with a bias under 0.03 standard deviations.
The `p4` calibration cell showed the same shortfall of the influence-curve variance at n = 2,000
and passed at n = 4,000. The primary therefore moved to n = 5,000, and the property cells kept
their declared sizes. At n = 5,000 the two `p4` rows read coverage 0.929 to 0.937 and an SE ratio
of 0.911 to 0.917, and they pass their declared rule. All 24 truth rows pass.
`tests/studies/canonical_mtp_boundaries.py` records the declaration.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the same law with a declared atom at the second dose, LTMLE | `ate_regimen[p5 vs natural]` | difference in mean outcome between the plans "set the second dose to the larger of 3 and 0.65 times the dose" against "leave the observed dose unchanged at both times" | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.0018 to 0.000616 | 0.9620 | 1.0280 | pass |
| the same law with a declared atom at the second dose, LTMLE | `ate_regimen[p5 vs natural]` | difference in mean outcome between the plans "set the second dose to the larger of 3 and 0.65 times the dose" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.0018 to 0.000625 | 0.9620 | 1.0288 | pass |
| the same law with a declared atom at the second dose, LTMLE | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.000166 to 0.000977 | 0.9530 | 0.9818 | pass |
| the same law with a declared atom at the second dose, LTMLE | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | R `lmtp` | -0.000166 to 0.000977 | 0.9530 | 0.9818 | pass |
| the same law with a declared atom at the second dose, LTMLE | `ey_regimen[p5]` | mean outcome under the plan set the second dose to the larger of 3 and 0.65 times the dose | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.0015 to 0.0011 | 0.9590 | 1.0420 | pass |
| the same law with a declared atom at the second dose, LTMLE | `ey_regimen[p5]` | mean outcome under the plan set the second dose to the larger of 3 and 0.65 times the dose | R `lmtp` | -0.0015 to 0.0011 | 0.9580 | 1.0424 | pass |
| the same law with a declared atom at the second dose, SDR | `ate_regimen[p5 vs natural]` | difference in mean outcome between the plans "set the second dose to the larger of 3 and 0.65 times the dose" against "leave the observed dose unchanged at both times" | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.0018 to 0.000642 | 0.9620 | 1.0275 | pass |
| the same law with a declared atom at the second dose, SDR | `ate_regimen[p5 vs natural]` | difference in mean outcome between the plans "set the second dose to the larger of 3 and 0.65 times the dose" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.0018 to 0.000654 | 0.9630 | 1.0281 | pass |
| the same law with a declared atom at the second dose, SDR | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.000166 to 0.000977 | 0.9530 | 0.9818 | pass |
| the same law with a declared atom at the second dose, SDR | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | R `lmtp` | -0.000166 to 0.000977 | 0.9530 | 0.9818 | pass |
| the same law with a declared atom at the second dose, SDR | `ey_regimen[p5]` | mean outcome under the plan set the second dose to the larger of 3 and 0.65 times the dose | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.0015 to 0.0012 | 0.9620 | 1.0413 | pass |
| the same law with a declared atom at the second dose, SDR | `ey_regimen[p5]` | mean outcome under the plan set the second dose to the larger of 3 and 0.65 times the dose | R `lmtp` | -0.0015 to 0.0012 | 0.9610 | 1.0415 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ate_regimen[p1 vs natural]` | difference in mean outcome between the plans "multiply the dose by 0.9 at both times" against "leave the observed dose unchanged at both times" | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.000417 to 0.000457 | 0.9490 | 1.0032 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ate_regimen[p1 vs natural]` | difference in mean outcome between the plans "multiply the dose by 0.9 at both times" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.000417 to 0.000457 | 0.9490 | 1.0032 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ate_regimen[p2 vs natural]` | difference in mean outcome between the plans "add 1 to the dose at both times, holding doses within 1 of the support's upper end" against "leave the observed dose unchanged at both times" | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.000804 to 0.000653 | 0.9480 | 1.0141 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ate_regimen[p2 vs natural]` | difference in mean outcome between the plans "add 1 to the dose at both times, holding doses within 1 of the support's upper end" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.000804 to 0.000653 | 0.9480 | 1.0141 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ate_regimen[p3 vs natural]` | difference in mean outcome between the plans "map the dose a to u - D(u - a) at both times, D a smooth warp whose slope runs from 0.4 to 1.6, u the support's upper end" against "leave the observed dose unchanged at both times" | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.000509 to 0.000634 | 0.9520 | 1.0078 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ate_regimen[p3 vs natural]` | difference in mean outcome between the plans "map the dose a to u - D(u - a) at both times, D a smooth warp whose slope runs from 0.4 to 1.6, u the support's upper end" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.000509 to 0.000634 | 0.9520 | 1.0078 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ate_regimen[p4 vs natural]` | difference in mean outcome between the plans "add 1 to a dose below 3 and subtract 1.5 from a dose at or above 3, at both times" against "leave the observed dose unchanged at both times" | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.0013 to 0.0027 | 0.9290 | 0.9169 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ate_regimen[p4 vs natural]` | difference in mean outcome between the plans "add 1 to a dose below 3 and subtract 1.5 from a dose at or above 3, at both times" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.0013 to 0.0027 | 0.9290 | 0.9169 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.000544 to 0.000552 | 0.9510 | 1.0169 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | R `lmtp` | -0.000544 to 0.000552 | 0.9510 | 1.0169 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ey_regimen[p1]` | mean outcome under the plan multiply the dose by 0.9 at both times | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.000706 to 0.000754 | 0.9530 | 0.9930 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ey_regimen[p1]` | mean outcome under the plan multiply the dose by 0.9 at both times | R `lmtp` | -0.000706 to 0.000754 | 0.9530 | 0.9930 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ey_regimen[p2]` | mean outcome under the plan add 1 to the dose at both times, holding doses within 1 of the support's upper end | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.000962 to 0.000819 | 0.9580 | 1.0167 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ey_regimen[p2]` | mean outcome under the plan add 1 to the dose at both times, holding doses within 1 of the support's upper end | R `lmtp` | -0.000962 to 0.000819 | 0.9580 | 1.0167 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ey_regimen[p3]` | mean outcome under the plan map the dose a to u - D(u - a) at both times, D a smooth warp whose slope runs from 0.4 to 1.6, u the support's upper end | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.000726 to 0.000858 | 0.9540 | 1.0240 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ey_regimen[p3]` | mean outcome under the plan map the dose a to u - D(u - a) at both times, D a smooth warp whose slope runs from 0.4 to 1.6, u the support's upper end | R `lmtp` | -0.000726 to 0.000858 | 0.9540 | 1.0240 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ey_regimen[p4]` | mean outcome under the plan add 1 to a dose below 3 and subtract 1.5 from a dose at or above 3, at both times | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.0014 to 0.0028 | 0.9320 | 0.9111 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ey_regimen[p4]` | mean outcome under the plan add 1 to a dose below 3 and subtract 1.5 from a dose at or above 3, at both times | R `lmtp` | -0.0014 to 0.0028 | 0.9320 | 0.9111 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ate_regimen[p1 vs natural]` | difference in mean outcome between the plans "multiply the dose by 0.9 at both times" against "leave the observed dose unchanged at both times" | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.000411 to 0.000463 | 0.9500 | 1.0031 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ate_regimen[p1 vs natural]` | difference in mean outcome between the plans "multiply the dose by 0.9 at both times" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.000411 to 0.000463 | 0.9500 | 1.0031 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ate_regimen[p2 vs natural]` | difference in mean outcome between the plans "add 1 to the dose at both times, holding doses within 1 of the support's upper end" against "leave the observed dose unchanged at both times" | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.000822 to 0.000641 | 0.9480 | 1.0110 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ate_regimen[p2 vs natural]` | difference in mean outcome between the plans "add 1 to the dose at both times, holding doses within 1 of the support's upper end" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.000822 to 0.000641 | 0.9480 | 1.0110 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ate_regimen[p3 vs natural]` | difference in mean outcome between the plans "map the dose a to u - D(u - a) at both times, D a smooth warp whose slope runs from 0.4 to 1.6, u the support's upper end" against "leave the observed dose unchanged at both times" | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.000518 to 0.000626 | 0.9500 | 1.0079 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ate_regimen[p3 vs natural]` | difference in mean outcome between the plans "map the dose a to u - D(u - a) at both times, D a smooth warp whose slope runs from 0.4 to 1.6, u the support's upper end" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.000518 to 0.000626 | 0.9500 | 1.0079 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ate_regimen[p4 vs natural]` | difference in mean outcome between the plans "add 1 to a dose below 3 and subtract 1.5 from a dose at or above 3, at both times" against "leave the observed dose unchanged at both times" | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.0013 to 0.0027 | 0.9370 | 0.9162 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ate_regimen[p4 vs natural]` | difference in mean outcome between the plans "add 1 to a dose below 3 and subtract 1.5 from a dose at or above 3, at both times" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.0013 to 0.0027 | 0.9370 | 0.9162 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.000544 to 0.000552 | 0.9510 | 1.0169 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | R `lmtp` | -0.000544 to 0.000552 | 0.9510 | 1.0169 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ey_regimen[p1]` | mean outcome under the plan multiply the dose by 0.9 at both times | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.000700 to 0.000759 | 0.9520 | 0.9929 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ey_regimen[p1]` | mean outcome under the plan multiply the dose by 0.9 at both times | R `lmtp` | -0.000700 to 0.000759 | 0.9520 | 0.9929 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ey_regimen[p2]` | mean outcome under the plan add 1 to the dose at both times, holding doses within 1 of the support's upper end | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.000980 to 0.000806 | 0.9570 | 1.0147 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ey_regimen[p2]` | mean outcome under the plan add 1 to the dose at both times, holding doses within 1 of the support's upper end | R `lmtp` | -0.000980 to 0.000806 | 0.9570 | 1.0147 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ey_regimen[p3]` | mean outcome under the plan map the dose a to u - D(u - a) at both times, D a smooth warp whose slope runs from 0.4 to 1.6, u the support's upper end | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.000734 to 0.000850 | 0.9550 | 1.0242 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ey_regimen[p3]` | mean outcome under the plan map the dose a to u - D(u - a) at both times, D a smooth warp whose slope runs from 0.4 to 1.6, u the support's upper end | R `lmtp` | -0.000734 to 0.000850 | 0.9550 | 1.0242 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ey_regimen[p4]` | mean outcome under the plan add 1 to a dose below 3 and subtract 1.5 from a dose at or above 3, at both times | `cleverly` LTMLE and SDR under modified treatment policies with declared supports and atoms | -0.0014 to 0.0029 | 0.9360 | 0.9105 | pass |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ey_regimen[p4]` | mean outcome under the plan add 1 to a dose below 3 and subtract 1.5 from a dose at or above 3, at both times | R `lmtp` | -0.0014 to 0.0029 | 0.9360 | 0.9105 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

The `boundaries_ltmle` and `boundaries_sdr` scenarios are exactness witnesses. Each of their 18
estimands pairs 1,000 fits with `lmtp`. The largest gap is 1.0e-07 in the estimate and 7.5e-10 in
the standard error, against a gate of 1e-6 (`exactness.csv`). The study fails a witness above the
gate.

The `atom_ltmle` and `atom_sdr` scenarios are margin-read. The package's node-2 design holds
the regressor 1{A2 = 3}, and `lmtp` cannot recompute that regressor at the shifted dose. Their
largest paired difference uses 0.0055 of the margin.

<!-- generated: agreement -->
Comparator-backed: a paired comparison with `lmtp` on the same samples. Each row reports estimate equivalence within the declared margin, the RMSE ratio and the coverage difference, and names its comparison kind.

| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | comparison kind | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| the same law with a declared atom at the second dose, LTMLE | `ate_regimen[p5 vs natural]` | difference in mean outcome between the plans "set the second dose to the larger of 3 and 0.65 times the dose" against "leave the observed dose unchanged at both times" | -0.000009 | 0.0039 | 1.0024 | 0 | matching implementation | 0.0028 vs 0.0500 | equivalent |
| the same law with a declared atom at the second dose, LTMLE | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | 3.709e-11 | 3.529e-08 | 1.0000 | 0 | matching implementation | 1.657e-08 vs 0.0500 | equivalent |
| the same law with a declared atom at the second dose, LTMLE | `ey_regimen[p5]` | mean outcome under the plan set the second dose to the larger of 3 and 0.65 times the dose | -0.000009 | 0.0036 | 1.0019 | 0.0010 | matching implementation | 0.0022 vs 0.0500 | equivalent |
| the same law with a declared atom at the second dose, SDR | `ate_regimen[p5 vs natural]` | difference in mean outcome between the plans "set the second dose to the larger of 3 and 0.65 times the dose" against "leave the observed dose unchanged at both times" | -0.000012 | 0.0055 | 1.0022 | -0.0010 | matching implementation | 0.0027 vs 0.0500 | equivalent |
| the same law with a declared atom at the second dose, SDR | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | 0 | 0 | 1 | 0 | matching implementation | 1.399e-14 vs 0.0500 | equivalent |
| the same law with a declared atom at the second dose, SDR | `ey_regimen[p5]` | mean outcome under the plan set the second dose to the larger of 3 and 0.65 times the dose | -0.000012 | 0.0051 | 1.0017 | 0.0010 | matching implementation | 0.0022 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ate_regimen[p1 vs natural]` | difference in mean outcome between the plans "multiply the dose by 0.9 at both times" against "leave the observed dose unchanged at both times" | 1.442e-11 | 1.796e-08 | 1.0000 | 0 | matching implementation | 2.227e-09 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ate_regimen[p2 vs natural]` | difference in mean outcome between the plans "add 1 to the dose at both times, holding doses within 1 of the support's upper end" against "leave the observed dose unchanged at both times" | -9.525e-11 | 7.113e-08 | 1.0000 | 0 | matching implementation | 3.040e-08 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ate_regimen[p3 vs natural]` | difference in mean outcome between the plans "map the dose a to u - D(u - a) at both times, D a smooth warp whose slope runs from 0.4 to 1.6, u the support's upper end" against "leave the observed dose unchanged at both times" | 1.710e-10 | 1.627e-07 | 1.0000 | 0 | matching implementation | 1.348e-08 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ate_regimen[p4 vs natural]` | difference in mean outcome between the plans "add 1 to a dose below 3 and subtract 1.5 from a dose at or above 3, at both times" against "leave the observed dose unchanged at both times" | 2.040e-11 | 5.561e-09 | 1.0000 | 0 | matching implementation | 3.410e-10 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | 1.098e-12 | 1.090e-09 | 1.0000 | 0 | matching implementation | 3.139e-10 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ey_regimen[p1]` | mean outcome under the plan multiply the dose by 0.9 at both times | 1.552e-11 | 1.157e-08 | 1.0000 | 0 | matching implementation | 6.935e-09 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ey_regimen[p2]` | mean outcome under the plan add 1 to the dose at both times, holding doses within 1 of the support's upper end | -9.415e-11 | 5.752e-08 | 1.0000 | 0 | matching implementation | 3.043e-08 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ey_regimen[p3]` | mean outcome under the plan map the dose a to u - D(u - a) at both times, D a smooth warp whose slope runs from 0.4 to 1.6, u the support's upper end | 1.721e-10 | 1.182e-07 | 1.0000 | 0 | matching implementation | 2.358e-09 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose on a history-dependent support, LTMLE | `ey_regimen[p4]` | mean outcome under the plan add 1 to a dose below 3 and subtract 1.5 from a dose at or above 3, at both times | 2.150e-11 | 5.537e-09 | 1.0000 | 0 | matching implementation | 2.948e-10 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ate_regimen[p1 vs natural]` | difference in mean outcome between the plans "multiply the dose by 0.9 at both times" against "leave the observed dose unchanged at both times" | 4.031e-18 | 5.022e-15 | 1.0000 | 0 | matching implementation | 1.377e-14 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ate_regimen[p2 vs natural]` | difference in mean outcome between the plans "add 1 to the dose at both times, holding doses within 1 of the support's upper end" against "leave the observed dose unchanged at both times" | -1.470e-16 | 1.093e-13 | 1.0000 | 0 | matching implementation | 4.052e-14 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ate_regimen[p3 vs natural]` | difference in mean outcome between the plans "map the dose a to u - D(u - a) at both times, D a smooth warp whose slope runs from 0.4 to 1.6, u the support's upper end" against "leave the observed dose unchanged at both times" | 2.851e-17 | 2.712e-14 | 1.0000 | 0 | matching implementation | 1.288e-14 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ate_regimen[p4 vs natural]` | difference in mean outcome between the plans "add 1 to a dose below 3 and subtract 1.5 from a dose at or above 3, at both times" against "leave the observed dose unchanged at both times" | -3.134e-16 | 8.434e-14 | 1.0000 | 0 | matching implementation | 1.532e-14 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | 0 | 0 | 1 | 0 | matching implementation | 4.441e-16 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ey_regimen[p1]` | mean outcome under the plan multiply the dose by 0.9 at both times | 9.992e-18 | 7.449e-15 | 1.0000 | 0 | matching implementation | 1.676e-14 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ey_regimen[p2]` | mean outcome under the plan add 1 to the dose at both times, holding doses within 1 of the support's upper end | -1.598e-16 | 9.735e-14 | 1.0000 | 0 | matching implementation | 4.730e-14 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ey_regimen[p3]` | mean outcome under the plan map the dose a to u - D(u - a) at both times, D a smooth warp whose slope runs from 0.4 to 1.6, u the support's upper end | 4.918e-17 | 3.379e-14 | 1.0000 | 0 | matching implementation | 1.021e-14 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose on a history-dependent support, SDR | `ey_regimen[p4]` | mean outcome under the plan add 1 to a dose below 3 and subtract 1.5 from a dose at or above 3, at both times | -3.106e-16 | 7.907e-14 | 1.0000 | 0 | matching implementation | 1.310e-14 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `atom_necessity` | `p5__declared` | positive | the two-node contrast of the map that sends an interval to the declared atom against the natural course: the ratio at the atom adds the mass of the interval sent to it, with the outcome misspecified | bias interval inside the equivalence margin | bias -0.0024 to 0.0014, margin 0.0059 | pass |
| `atom_necessity` | `p5__mass_dropped_control` | control | the two-node contrast of the map that sends an interval to the declared atom against the natural course: the same fit drops that mass, so the ratio at the atom is one | bias interval must fall entirely outside the margin | bias 0.0087 to 0.0135, margin 0.0073 | pass |
| `branch_necessity` | `p4__declared` | positive | the two-node contrast of the two-branch map against the natural course: the density ratio sums over both inverse branches, with the outcome misspecified | bias interval inside the equivalence margin | bias -0.0034 to 0.0026, margin 0.0093 | pass |
| `branch_necessity` | `p4__one_branch_control` | control | the two-node contrast of the two-branch map against the natural course: the same fit sums over the first branch only | bias interval must fall entirely outside the margin | bias -0.0311 to -0.0249, margin 0.0095 | pass |
| `interval_calibration` | `p1__correctly_specified` | positive | the two-node contrast of the nonbinding 0.9 scale against the natural course: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9470 to 0.9539, SE ratio 0.9897 to 1.0128 | pass |
| `interval_calibration` | `p1__noise_control` | control | the two-node contrast of the nonbinding 0.9 scale against the natural course: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8317 to 0.8434, SE ratio 0.7029 to 0.7191 | pass |
| `interval_calibration` | `p1__shrunken_se_control` | control | the two-node contrast of the nonbinding 0.9 scale against the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8254 to 0.8373, SE ratio 0.6926 to 0.7087 | pass |
| `interval_calibration` | `p2__correctly_specified` | positive | the two-node contrast of the shift held at the history-dependent support boundary against the natural course: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9439 to 0.9510, SE ratio 0.9712 to 1.0003 | pass |
| `interval_calibration` | `p2__noise_control` | control | the two-node contrast of the shift held at the history-dependent support boundary against the natural course: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8322 to 0.8439, SE ratio 0.7039 to 0.7212 | pass |
| `interval_calibration` | `p2__shrunken_se_control` | control | the two-node contrast of the shift held at the history-dependent support boundary against the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8218 to 0.8338, SE ratio 0.6795 to 0.7002 | pass |
| `interval_calibration` | `p3__correctly_specified` | positive | the two-node contrast of the warped map against the natural course: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9423 to 0.9495, SE ratio 0.9871 to 1.0098 | pass |
| `interval_calibration` | `p3__noise_control` | control | the two-node contrast of the warped map against the natural course: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8242 to 0.8361, SE ratio 0.6965 to 0.7125 | pass |
| `interval_calibration` | `p3__shrunken_se_control` | control | the two-node contrast of the warped map against the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8224 to 0.8344, SE ratio 0.6909 to 0.7068 | pass |
| `interval_calibration` | `p4__correctly_specified` | positive | the two-node contrast of the two-branch map against the natural course: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9346 to 0.9423, SE ratio 0.9466 to 0.9683 | pass |
| `interval_calibration` | `p4__noise_control` | control | the two-node contrast of the two-branch map against the natural course: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8068 to 0.8192, SE ratio 0.6799 to 0.6956 | pass |
| `interval_calibration` | `p4__shrunken_se_control` | control | the two-node contrast of the two-branch map against the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8023 to 0.8148, SE ratio 0.6625 to 0.6773 | pass |
| `interval_calibration` | `p4_sdr__correctly_specified` | positive | the two-branch contrast read by the sequentially doubly robust estimator: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9233 to 0.9467, SE ratio 0.9060 to 0.9775 | **fail** |
| `interval_calibration` | `p4_sdr__noise_control` | control | the two-branch contrast read by the sequentially doubly robust estimator: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.7999 to 0.8364, SE ratio 0.6628 to 0.7100 | pass |
| `interval_calibration` | `p4_sdr__shrunken_se_control` | control | the two-branch contrast read by the sequentially doubly robust estimator: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7809 to 0.8188, SE ratio 0.6345 to 0.6836 | pass |
| `interval_calibration` | `p5__correctly_specified` | positive | the two-node contrast of the map that sends an interval to the declared atom against the natural course: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9335 to 0.9412, SE ratio 0.9643 to 0.9863 | pass |
| `interval_calibration` | `p5__noise_control` | control | the two-node contrast of the map that sends an interval to the declared atom against the natural course: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8067 to 0.8191, SE ratio 0.6779 to 0.6934 | pass |
| `interval_calibration` | `p5__shrunken_se_control` | control | the two-node contrast of the map that sends an interval to the declared atom against the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8085 to 0.8209, SE ratio 0.6750 to 0.6902 | pass |
| `interval_calibration` | `p5_sdr__correctly_specified` | positive | the declared-atom contrast read by the sequentially doubly robust estimator: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9451 to 0.9521, SE ratio 0.9771 to 0.9988 | pass |
| `interval_calibration` | `p5_sdr__noise_control` | control | the declared-atom contrast read by the sequentially doubly robust estimator: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8144 to 0.8266, SE ratio 0.6844 to 0.7000 | pass |
| `interval_calibration` | `p5_sdr__shrunken_se_control` | control | the declared-atom contrast read by the sequentially doubly robust estimator: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8166 to 0.8287, SE ratio 0.6840 to 0.6990 | pass |
| `jacobian_necessity` | `p3__declared` | positive | the two-node contrast of the warped map against the natural course: the density ratio reads each piece's inverse slope, with an outcome model blind to the second dose | bias interval inside the equivalence margin | bias -0.0013 to 0.000545, margin 0.0028 | pass |
| `jacobian_necessity` | `p3__jacobian_dropped_control` | control | the two-node contrast of the warped map against the natural course: the same fit replaces each inverse slope with one | bias interval must fall entirely outside the margin | bias 0.0075 to 0.0085, margin 0.0014 | pass |
| `ratio_necessity` | `p2__declared` | positive | the two-node contrast of the shift held at the history-dependent support boundary against the natural course: the shifted-to-natural density ratio is used in the declared direction | bias interval inside the equivalence margin | bias -0.0017 to 0.000682, margin 0.0037 | pass |
| `ratio_necessity` | `p2__unit_ratio_control` | control | the two-node contrast of the shift held at the history-dependent support boundary against the natural course: the same fit replaces the density ratio with one, with the outcome misspecified | bias interval must fall entirely outside the margin | bias 0.0105 to 0.0121, margin 0.0025 | pass |
<!-- /generated -->

The truths come from the exact laws and their closed forms
(`tests/studies/mtp_boundaries_common.py`). Quadrature gives each truth, and the code checks it
against a rule of twice the order at $10^{-10}$.

| family | cells | replications and size |
| --- | --- | --- |
| `interval_calibration` | seven positives: `LTMLE` under `p1` to `p4` on law B and `p5` on law B', and `SDR` under `p4` and `p5`. Each positive has a shrunken-SE control and a noise control | 26,400 replications at n = 2,000, and at n = 4,000 for `p4`. The `p4_sdr` cell reports with 3,000 replications at n = 4,000 |
| `jacobian_necessity` | `p3`, dose-blind outcome. The declared fit against the same draw with every inverse slope replaced by 1 | 1,000 pairs at n = 2,000 |
| `branch_necessity` | `p4`, wrong outcome. The declared fit against the same draw with Equation (3) summed over the first branch only | 1,000 pairs at n = 2,000 |
| `atom_necessity` | `p5`, wrong outcome. The declared fit against the same draw with the atom's ratio taken without the mass of the interval sent to it | 1,000 pairs at n = 2,000 |
| `ratio_necessity` | `p2`, wrong outcome. The declared ratio against a ratio of one | 1,000 pairs at n = 2,000 |

Each necessity control removes one term that can vanish at the truth, so each needs a nonzero
witness. With a wrong outcome regression the targeting step carries the whole correction, and each
defect then moves the estimate. The Jacobian pair reads a dose-blind outcome model, because a
factor that is monotone in the dose is absorbed by the targeting step under the wrong model.
`tests/unit/test_mtp_boundaries_design.py` checks the sizes and the displacements before the run.

The `p4` cell of `LTMLE` moved to n = 4,000 because its pilot at n = 2,000 read a right-skewed
standard error from rare large weights. The `p4_sdr` cell stays outside the screen at both sizes,
so the study declares it `reporting` before the run. Its owner is the `SDR` limit in
[the interval under heavy cumulative density ratios](../longitudinal-tmle.md#the-interval-under-heavy-cumulative-density-ratios).

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | paired replications |
| `n` | 5000 | observations per paired replication |
| `independent_tests_passed` | 48 | truth tests passing |
| `independent_tests_total` | 48 | truth tests reported |
| `paired_tests_passed` | 24 | paired comparisons passing |
| `paired_tests_total` | 24 | paired comparisons reported |
| `property_cells_passed` | 28 | property cells passing |
| `property_cells_total` | 29 | property cells reported |
| `max_standardized_bias` | 0.0579 | largest primary standardized bias |
| `min_coverage` | 0.9290 | lowest primary coverage |
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
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:necessity_displacement` | 0.1000 | minimum necessity displacement of the four ratio and map pairs |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 10 | the same bound, in primary replications |

## Limits

| limit | what it means for use |
| --- | --- |
| two laws with two nodes and a binary outcome | the evidence covers a Bernoulli outcome, and a continuous outcome needs `q_bounds`. The study has no size ladder, so it claims no rate |
| the density is an oracle | the density route reads the generating pooled-hazard probabilities. The default estimated density is not tested here |
| the declared support is the true support | a declared support that is wrong but consistent with the data is outside this evidence |
| the `p4_sdr` cell reports | the `SDR` interval runs short of nominal under heavy cumulative density ratios. The package treats the interval as experimental, with the measured numbers of the red-cell owner `X40-sdr-heavy-ratios` |
| the `p4` standard error is short at n = 5,000 | the two `p4` rows of `boundaries_ltmle` read an SE ratio of 0.911 to 0.917 and coverage of 0.929 to 0.937. The rows pass the declared rule. Read an interval under this plan as slightly narrow |
| the primary rows use main-terms learners | the primary phase matches the comparator's `SL.glm`. The property cells read the correct outcome model |
| the atom covers one node | plan `p5` sends an interval to an atom at node 2 only. An atom with a near-zero mass is outside this evidence |
| the atom scenarios are margin-read | `lmtp` fits a different node-2 design at the atom, so the atom rows show agreement within the margin and not exactness |
