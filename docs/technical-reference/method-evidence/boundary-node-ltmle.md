# The boundary-node supplement of LTMLE

This study validates the interval of an `LTMLE` estimate that carries the boundary-node
supplement. When a node regression's fitted rows hold one value, the node drops out of the
influence curve. Inside the supplement's reach the estimate adds the node's variance at the exact
one-sided binomial bound of its hazard (`cleverly.inference.node_variance`). The
[node with no event](../longitudinal-tmle.md#a-node-with-no-event) section states the
construction and its reach.

**Status of this page.** The study is registered, and its run has completed. No other
implementation adds such a supplement, so the study has no comparator. The publication policy is
`reporting`. Two primary rows are red on their SE-ratio leg alone, as declared before the run.
They stay red in the [red-cell ledger](red-cells.md) under the owner `X54-supplement-width`.

| part | what it is | status |
| --- | --- | --- |
| primary | in-sample iterated-mean `LTMLE` on law A at n = 200, 1,000 replications: the risks under always and never at horizon 4 and their contrast | the accuracy table below. The SE-ratio legs of the always risk and the contrast are red by design: the supplement is a bound, so it is wider than the node's true variance |
| property family `boundary_node` | eight cells on 4,000 draws each. Every draw fits once and reads three intervals off that fit: the shipped one, the plug-in without the supplement, and an oracle that adds the flagged node's variance at the true hazards | the tables below give the verdicts |
| comparator runs | none | R `survtmle` reads such a node as zero and has no supplement, so it would reproduce only the plug-in arm |

## What was fitted

Law A has a baseline $W \sim N(0, 1)$, three treatment decisions before event nodes 1, 3 and 4
with known mechanisms $g_k = \operatorname{expit}(\gamma_k + 0.3 W)$, and hazards
$h_t = \operatorname{expit}(\alpha_t + 0.4 W - 0.5 A_{k(t)} + b)$ with every $\alpha_t$ set so the
hazard is about 0.08. The cluster effect $b$ (SD 0.5) is present only in the clustered cell. Node 4
holds 0.70 of the variance of the always risk and is flagged under always on about half the draws
at n = 200, while nodes 1 to 3 are flagged on at most 0.07.

Law B holds one baseline decision and runs the hazard path. Its flagged node carries about 0.03 of the variance, so law B shows what the
supplement costs on the hazard path and cannot show that it is needed. The zero variants set the
treated node-4 hazard to 0, so node 4 fires under always on every draw. The laws, their declared
grids and their exact figures are in `tests/studies/boundary_node_laws.py`, and
`tests/unit/test_boundary_node_study_design.py` recomputes each against a Monte Carlo draw.

The node regressions use the study's quasi-binomial GLM on main terms and the hazards an
unpenalized logistic regression, with the known treatment mechanisms. A firing draw is one where
node 4 of the target's regimen is flagged (always for the always risk and the contrast, never for
the never risk). A quiet draw is neither firing nor supplemented at another node.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| law A, three known treatment decisions before event nodes 1, 3 and 4 with every hazard near 0.08, in-sample iterated-mean LTMLE at n = 200: node 4 is flagged under always on about half the draws, and the shipped interval adds its variance at the exact one-sided bound of its hazard | `ate_regimen[always vs never @ t=4]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 4 | `cleverly` | -0.0037 to 0.0133 | 0.9470 | 1.1713 | **fail** |
| law A, three known treatment decisions before event nodes 1, 3 and 4 with every hazard near 0.08, in-sample iterated-mean LTMLE at n = 200: node 4 is flagged under always on about half the draws, and the shipped interval adds its variance at the exact one-sided bound of its hazard | `risk_regimen[always @ t=4]` | cumulative risk under the plan treat at both times at horizon t = 4 | `cleverly` | -0.000905 to 0.0124 | 0.9270 | 1.2605 | **fail** |
| law A, three known treatment decisions before event nodes 1, 3 and 4 with every hazard near 0.08, in-sample iterated-mean LTMLE at n = 200: node 4 is flagged under always on about half the draws, and the shipped interval adds its variance at the exact one-sided bound of its hazard | `risk_regimen[never @ t=4]` | cumulative risk under the plan treat at neither time at horizon t = 4 | `cleverly` | -0.0044 to 0.0063 | 0.9410 | 0.9506 | pass |
<!-- /generated -->

The red SE-ratio legs are the declared reading of `X54-supplement-width`: the supplement adds a
flagged node's variance at the one-sided bound of its hazard, not at the hazard, so its standard
error runs above the spread of the estimates. The coverage and bias legs of both rows pass, and
`unowned_primary_failures` in `tests/studies/boundary_node_ltmle.py` finds no red leg that the
reading does not hold.

## Agreement with the canonical implementation

None claimed. No canonical implementation is compared, because no other implementation adds a
boundary-node supplement. [0/0 external comparisons; the row rests on the other two columns]

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `boundary_node` | `clustered_j40__always__oracle_node` | diagnostic | law A with a cluster effect, 40 clusters of 10 rows, risk under always at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9254 to 0.9456; firing coverage 0.9636 to 0.9887 on 980 draws | reported |
| `boundary_node` | `clustered_j40__always__plugin` | diagnostic | law A with a cluster effect, 40 clusters of 10 rows, risk under always at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.8427 to 0.8714; firing coverage 0.6179 to 0.6968 on 980 draws | reported |
| `boundary_node` | `clustered_j40__always__supplemented` | positive | law A with a cluster effect, 40 clusters of 10 rows, risk under always at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin and the exact 99% lower end of coverage on the firing draws at least 0.90 | bias 0.000078 to 0.0048, margin 0.0145; coverage 0.9310 to 0.9504; firing coverage 0.9946 to 1 on 980 draws | pass |
| `boundary_node` | `clustered_j40__contrast__oracle_node` | diagnostic | law A with a cluster effect, 40 clusters of 10 rows, always-minus-never risk difference at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9289 to 0.9486; firing coverage 0.9437 to 0.9762 on 980 draws | reported |
| `boundary_node` | `clustered_j40__contrast__plugin` | diagnostic | law A with a cluster effect, 40 clusters of 10 rows, always-minus-never risk difference at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.8990 to 0.9225; firing coverage 0.8195 to 0.8791 on 980 draws | reported |
| `boundary_node` | `clustered_j40__contrast__supplemented` | positive | law A with a cluster effect, 40 clusters of 10 rows, always-minus-never risk difference at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin and the exact 99% lower end of coverage on the firing draws at least 0.90 | bias -0.0016 to 0.0044, margin 0.0182; coverage 0.9369 to 0.9555; firing coverage 0.9826 to 0.9979 on 980 draws | pass |
| `boundary_node` | `clustered_j40__never__oracle_node` | diagnostic | law A with a cluster effect, 40 clusters of 10 rows, risk under never at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9184 to 0.9395; firing coverage 0.6982 to 0.9734 on 42 draws | reported |
| `boundary_node` | `clustered_j40__never__plugin` | diagnostic | law A with a cluster effect, 40 clusters of 10 rows, risk under never at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.9160 to 0.9374; firing coverage 0.4571 to 0.8377 on 42 draws | reported |
| `boundary_node` | `clustered_j40__never__supplemented` | positive | law A with a cluster effect, 40 clusters of 10 rows, risk under never at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin | bias -0.000942 to 0.0030, margin 0.0121; coverage 0.9187 to 0.9398; firing coverage 0.7295 to 0.9835 on 42 draws | pass |
| `boundary_node` | `held_n200__always__oracle_node` | diagnostic | law B, one held baseline decision, hazard path, n = 200, risk under always at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9423 to 0.9601; firing coverage 0.9414 to 0.9665 on 1894 draws | reported |
| `boundary_node` | `held_n200__always__plugin` | diagnostic | law B, one held baseline decision, hazard path, n = 200, risk under always at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.9404 to 0.9585; firing coverage 0.9373 to 0.9633 on 1894 draws | reported |
| `boundary_node` | `held_n200__always__supplemented` | positive | law B, one held baseline decision, hazard path, n = 200, risk under always at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin and the exact 99% lower end of coverage on the firing draws at least 0.90 | bias -0.0014 to 0.0022, margin 0.0111; coverage 0.9497 to 0.9662; firing coverage 0.9577 to 0.9786 on 1894 draws | pass |
| `boundary_node` | `held_n200__contrast__oracle_node` | diagnostic | law B, one held baseline decision, hazard path, n = 200, always-minus-never risk difference at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9334 to 0.9525; firing coverage 0.9355 to 0.9619 on 1894 draws | reported |
| `boundary_node` | `held_n200__contrast__plugin` | diagnostic | law B, one held baseline decision, hazard path, n = 200, always-minus-never risk difference at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.9302 to 0.9497; firing coverage 0.9308 to 0.9582 on 1894 draws | reported |
| `boundary_node` | `held_n200__contrast__supplemented` | positive | law B, one held baseline decision, hazard path, n = 200, always-minus-never risk difference at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin and the exact 99% lower end of coverage on the firing draws at least 0.90 | bias -0.0021 to 0.0023, margin 0.0137; coverage 0.9415 to 0.9594; firing coverage 0.9492 to 0.9724 on 1894 draws | pass |
| `boundary_node` | `held_n200__never__oracle_node` | diagnostic | law B, one held baseline decision, hazard path, n = 200, risk under never at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9464 to 0.9635; firing coverage 0.9446 to 0.9676 on 2147 draws | reported |
| `boundary_node` | `held_n200__never__plugin` | diagnostic | law B, one held baseline decision, hazard path, n = 200, risk under never at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.9440 to 0.9614; firing coverage 0.9399 to 0.9639 on 2147 draws | reported |
| `boundary_node` | `held_n200__never__supplemented` | positive | law B, one held baseline decision, hazard path, n = 200, risk under never at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin and the exact 99% lower end of coverage on the firing draws at least 0.90 | bias -0.0010 to 0.0017, margin 0.0083; coverage 0.9522 to 0.9682; firing coverage 0.9557 to 0.9761 on 2147 draws | pass |
| `boundary_node` | `held_zero_n200__always__oracle_node` | diagnostic | law B with a true node-4 hazard of 0 under always, hazard path, n = 200, risk under always at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9399 to 0.9580; firing coverage 0.9399 to 0.9580 on 4000 draws | reported |
| `boundary_node` | `held_zero_n200__always__plugin` | diagnostic | law B with a true node-4 hazard of 0 under always, hazard path, n = 200, risk under always at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.9399 to 0.9580; firing coverage 0.9399 to 0.9580 on 4000 draws | reported |
| `boundary_node` | `held_zero_n200__always__supplemented` | positive | law B with a true node-4 hazard of 0 under always, hazard path, n = 200, risk under always at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin, the exact 99% lower end of coverage on the firing draws at least 0.90 and the exact 99% lower end of all-draw coverage at least 0.92 | bias -0.0022 to 0.0014, margin 0.0111; coverage 0.9579 to 0.9729; firing coverage 0.9579 to 0.9729 on 4000 draws | pass |
| `boundary_node` | `held_zero_n200__contrast__oracle_node` | diagnostic | law B with a true node-4 hazard of 0 under always, hazard path, n = 200, always-minus-never risk difference at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9399 to 0.9580; firing coverage 0.9399 to 0.9580 on 4000 draws | reported |
| `boundary_node` | `held_zero_n200__contrast__plugin` | diagnostic | law B with a true node-4 hazard of 0 under always, hazard path, n = 200, always-minus-never risk difference at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.9388 to 0.9571; firing coverage 0.9388 to 0.9571 on 4000 draws | reported |
| `boundary_node` | `held_zero_n200__contrast__supplemented` | positive | law B with a true node-4 hazard of 0 under always, hazard path, n = 200, always-minus-never risk difference at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin and the exact 99% lower end of coverage on the firing draws at least 0.90 | bias -0.0032 to 0.0013, margin 0.0137; coverage 0.9549 to 0.9705; firing coverage 0.9549 to 0.9705 on 4000 draws | pass |
| `boundary_node` | `held_zero_n200__never__oracle_node` | diagnostic | law B with a true node-4 hazard of 0 under always, hazard path, n = 200, risk under never at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9464 to 0.9635; firing coverage 0.9504 to 0.9721 on 2149 draws | reported |
| `boundary_node` | `held_zero_n200__never__plugin` | diagnostic | law B with a true node-4 hazard of 0 under always, hazard path, n = 200, risk under never at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.9404 to 0.9585; firing coverage 0.9389 to 0.9632 on 2149 draws | reported |
| `boundary_node` | `held_zero_n200__never__supplemented` | positive | law B with a true node-4 hazard of 0 under always, hazard path, n = 200, risk under never at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin and the exact 99% lower end of coverage on the firing draws at least 0.90 | bias -0.000821 to 0.0019, margin 0.0083; coverage 0.9524 to 0.9685; firing coverage 0.9621 to 0.9808 on 2149 draws | pass |
| `boundary_node` | `interior_n100__always__oracle_node` | diagnostic | law A in sample at n = 100, risk under always at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9122 to 0.9341; firing coverage 0.9912 to 0.9982 on 2749 draws | reported |
| `boundary_node` | `interior_n100__always__plugin` | diagnostic | law A in sample at n = 100, risk under always at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.8376 to 0.8668; firing coverage 0.8766 to 0.9074 on 2749 draws | reported |
| `boundary_node` | `interior_n100__always__supplemented` | positive | law A in sample at n = 100, risk under always at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin and the exact 99% lower end of coverage on the firing draws at least 0.90 | bias 0.0046 to 0.0149, margin 0.0317; coverage 0.9119 to 0.9339; firing coverage 0.9907 to 0.9980 on 2749 draws | pass |
| `boundary_node` | `interior_n100__contrast__oracle_node` | diagnostic | law A in sample at n = 100, always-minus-never risk difference at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9159 to 0.9374; firing coverage 0.9522 to 0.9712 on 2749 draws | reported |
| `boundary_node` | `interior_n100__contrast__plugin` | diagnostic | law A in sample at n = 100, always-minus-never risk difference at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.8536 to 0.8814; firing coverage 0.8704 to 0.9019 on 2749 draws | reported |
| `boundary_node` | `interior_n100__contrast__supplemented` | positive | law A in sample at n = 100, always-minus-never risk difference at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin and the exact 99% lower end of coverage on the firing draws at least 0.90 | bias -0.0020 to 0.0110, margin 0.0397; coverage 0.9418 to 0.9597; firing coverage 0.9860 to 0.9954 on 2749 draws | pass |
| `boundary_node` | `interior_n100__never__oracle_node` | diagnostic | law A in sample at n = 100, risk under never at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9278 to 0.9477; firing coverage 0.9761 to 0.9935 on 1280 draws | reported |
| `boundary_node` | `interior_n100__never__plugin` | diagnostic | law A in sample at n = 100, risk under never at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.8870 to 0.9117; firing coverage 0.8410 to 0.8906 on 1280 draws | reported |
| `boundary_node` | `interior_n100__never__supplemented` | positive | law A in sample at n = 100, risk under never at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin and the exact 99% lower end of coverage on the firing draws at least 0.90 | bias 0.0014 to 0.0091, margin 0.0237; coverage 0.9307 to 0.9502; firing coverage 0.9878 to 0.9988 on 1280 draws | pass |
| `boundary_node` | `interior_n200__always__oracle_node` | diagnostic | law A in sample at n = 200, risk under always at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9206 to 0.9414; firing coverage 0.9964 to 1.0000 on 2039 draws | reported |
| `boundary_node` | `interior_n200__always__plugin` | control | law A in sample at n = 200, risk under always at horizon 4: the same estimate at the curve's variance alone, the supplement removed | on the firing draws, the exact 99% McNemar lower end of supplemented minus plug-in coverage above 0, and the exact 99% lower end of oracle-node minus plug-in coverage at least 0.05 | firing coverage 0.8074 to 0.8507 on 2039 draws; gained 347, lost 0, McNemar lower 0.1650; oracle gained 346, lost 0, lower 0.1488 | pass |
| `boundary_node` | `interior_n200__always__supplemented` | positive | law A in sample at n = 200, risk under always at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin and the exact 99% lower end of coverage on the firing draws at least 0.90 | bias 0.0026 to 0.0097, margin 0.0218; coverage 0.9208 to 0.9416; firing coverage 0.9974 to 1 on 2039 draws | pass |
| `boundary_node` | `interior_n200__contrast__oracle_node` | diagnostic | law A in sample at n = 200, always-minus-never risk difference at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9168 to 0.9381; firing coverage 0.9450 to 0.9685 on 2039 draws | reported |
| `boundary_node` | `interior_n200__contrast__plugin` | diagnostic | law A in sample at n = 200, always-minus-never risk difference at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.8719 to 0.8981; firing coverage 0.8649 to 0.9018 on 2039 draws | reported |
| `boundary_node` | `interior_n200__contrast__supplemented` | positive | law A in sample at n = 200, always-minus-never risk difference at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin and the exact 99% lower end of coverage on the firing draws at least 0.90 | bias 0.000980 to 0.0100, margin 0.0276; coverage 0.9372 to 0.9557; firing coverage 0.9882 to 0.9976 on 2039 draws | pass |
| `boundary_node` | `interior_n200__never__oracle_node` | diagnostic | law A in sample at n = 200, risk under never at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9307 to 0.9502; firing coverage 0.9635 to 0.9964 on 424 draws | reported |
| `boundary_node` | `interior_n200__never__plugin` | diagnostic | law A in sample at n = 200, risk under never at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.9075 to 0.9300; firing coverage 0.7247 to 0.8302 on 424 draws | reported |
| `boundary_node` | `interior_n200__never__supplemented` | positive | law A in sample at n = 200, risk under never at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin | bias -0.0020 to 0.0033, margin 0.0163; coverage 0.9318 to 0.9511; firing coverage 0.9783 to 0.9998 on 424 draws | pass |
| `boundary_node` | `interior_n200_crossfit__always__oracle_node` | diagnostic | law A cross-fitted with five folds at n = 200, risk under always at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9510 to 0.9673; firing coverage 0.9974 to 1 on 2019 draws | reported |
| `boundary_node` | `interior_n200_crossfit__always__plugin` | diagnostic | law A cross-fitted with five folds at n = 200, risk under always at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.8732 to 0.8992; firing coverage 0.8341 to 0.8749 on 2019 draws | reported |
| `boundary_node` | `interior_n200_crossfit__always__supplemented` | positive | law A cross-fitted with five folds at n = 200, risk under always at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin and the exact 99% lower end of coverage on the firing draws at least 0.90 | bias -0.0014 to 0.0057, margin 0.0217; coverage 0.9510 to 0.9673; firing coverage 0.9974 to 1 on 2019 draws | pass |
| `boundary_node` | `interior_n200_crossfit__contrast__oracle_node` | diagnostic | law A cross-fitted with five folds at n = 200, always-minus-never risk difference at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9420 to 0.9598; firing coverage 0.9586 to 0.9787 on 2019 draws | reported |
| `boundary_node` | `interior_n200_crossfit__contrast__plugin` | diagnostic | law A cross-fitted with five folds at n = 200, always-minus-never risk difference at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.9162 to 0.9376; firing coverage 0.9083 to 0.9391 on 2019 draws | reported |
| `boundary_node` | `interior_n200_crossfit__contrast__supplemented` | positive | law A cross-fitted with five folds at n = 200, always-minus-never risk difference at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin and the exact 99% lower end of coverage on the firing draws at least 0.90 | bias -0.0020 to 0.0068, margin 0.0271; coverage 0.9576 to 0.9727; firing coverage 0.9930 to 0.9995 on 2019 draws | pass |
| `boundary_node` | `interior_n200_crossfit__never__oracle_node` | diagnostic | law A cross-fitted with five folds at n = 200, risk under never at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9407 to 0.9587; firing coverage 0.9516 to 0.9923 on 408 draws | reported |
| `boundary_node` | `interior_n200_crossfit__never__plugin` | diagnostic | law A cross-fitted with five folds at n = 200, risk under never at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.9234 to 0.9439; firing coverage 0.7673 to 0.8672 on 408 draws | reported |
| `boundary_node` | `interior_n200_crossfit__never__supplemented` | positive | law A cross-fitted with five folds at n = 200, risk under never at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin | bias -0.0029 to 0.0024, margin 0.0165; coverage 0.9412 to 0.9591; firing coverage 0.9585 to 0.9950 on 408 draws | pass |
| `boundary_node` | `interior_n400__always__oracle_node` | diagnostic | law A in sample at n = 400, risk under always at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9402 to 0.9582; firing coverage 0.9876 to 0.9993 on 1015 draws | reported |
| `boundary_node` | `interior_n400__always__plugin` | diagnostic | law A in sample at n = 400, risk under always at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.8680 to 0.8946; firing coverage 0.6906 to 0.7634 on 1015 draws | reported |
| `boundary_node` | `interior_n400__always__supplemented` | positive | law A in sample at n = 400, risk under always at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin and the exact 99% lower end of coverage on the firing draws at least 0.90 | bias 0.0016 to 0.0060, margin 0.0137; coverage 0.9413 to 0.9591; firing coverage 0.9948 to 1 on 1015 draws | pass |
| `boundary_node` | `interior_n400__contrast__oracle_node` | diagnostic | law A in sample at n = 400, always-minus-never risk difference at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9275 to 0.9474; firing coverage 0.9237 to 0.9617 on 1015 draws | reported |
| `boundary_node` | `interior_n400__contrast__plugin` | diagnostic | law A in sample at n = 400, always-minus-never risk difference at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.9083 to 0.9307; firing coverage 0.8469 to 0.9012 on 1015 draws | reported |
| `boundary_node` | `interior_n400__contrast__supplemented` | positive | law A in sample at n = 400, always-minus-never risk difference at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin and the exact 99% lower end of coverage on the firing draws at least 0.90 | bias 0.000198 to 0.0061, margin 0.0180; coverage 0.9391 to 0.9573; firing coverage 0.9764 to 0.9951 on 1015 draws | pass |
| `boundary_node` | `interior_n400__never__oracle_node` | diagnostic | law A in sample at n = 400, risk under never at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9278 to 0.9477; firing coverage 0.6909 to 0.9575 on 50 draws | reported |
| `boundary_node` | `interior_n400__never__plugin` | diagnostic | law A in sample at n = 400, risk under never at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.9232 to 0.9437; firing coverage 0.3337 to 0.7024 on 50 draws | reported |
| `boundary_node` | `interior_n400__never__supplemented` | positive | law A in sample at n = 400, risk under never at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin | bias -0.0012 to 0.0025, margin 0.0115; coverage 0.9262 to 0.9463; firing coverage 0.5526 to 0.8803 on 50 draws | pass |
| `boundary_node` | `zero_n200__always__oracle_node` | diagnostic | law A with a true node-4 hazard of 0 under always, n = 200, risk under always at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9489 to 0.9655; firing coverage 0.9489 to 0.9655 on 4000 draws | reported |
| `boundary_node` | `zero_n200__always__plugin` | diagnostic | law A with a true node-4 hazard of 0 under always, n = 200, risk under always at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.9334 to 0.9525; firing coverage 0.9334 to 0.9525 on 4000 draws | reported |
| `boundary_node` | `zero_n200__always__supplemented` | positive | law A with a true node-4 hazard of 0 under always, n = 200, risk under always at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin, the exact 99% lower end of coverage on the firing draws at least 0.90 and the exact 99% lower end of all-draw coverage at least 0.92 | bias -0.0019 to 0.0014, margin 0.0104; coverage 0.9987 to 1; firing coverage 0.9987 to 1 on 4000 draws | pass |
| `boundary_node` | `zero_n200__contrast__oracle_node` | diagnostic | law A with a true node-4 hazard of 0 under always, n = 200, always-minus-never risk difference at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9348 to 0.9536; firing coverage 0.9348 to 0.9536 on 4000 draws | reported |
| `boundary_node` | `zero_n200__contrast__plugin` | diagnostic | law A with a true node-4 hazard of 0 under always, n = 200, always-minus-never risk difference at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.9235 to 0.9439; firing coverage 0.9235 to 0.9439 on 4000 draws | reported |
| `boundary_node` | `zero_n200__contrast__supplemented` | positive | law A with a true node-4 hazard of 0 under always, n = 200, always-minus-never risk difference at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin and the exact 99% lower end of coverage on the firing draws at least 0.90 | bias -0.0042 to 0.0020, margin 0.0190; coverage 0.9977 to 1.0000; firing coverage 0.9977 to 1.0000 on 4000 draws | pass |
| `boundary_node` | `zero_n200__never__oracle_node` | diagnostic | law A with a true node-4 hazard of 0 under always, n = 200, risk under never at horizon 4: the same estimate at the curve's variance plus the flagged nodes' variance at the true row hazards | none. The row is a diagnostic and states no verdict | coverage 0.9283 to 0.9481; firing coverage 0.9433 to 0.9890 on 396 draws | reported |
| `boundary_node` | `zero_n200__never__plugin` | diagnostic | law A with a true node-4 hazard of 0 under always, n = 200, risk under never at horizon 4: the same estimate at the curve's variance alone, the supplement removed | none. The row is a diagnostic and states no verdict | coverage 0.9112 to 0.9332; firing coverage 0.7550 to 0.8586 on 396 draws | reported |
| `boundary_node` | `zero_n200__never__supplemented` | positive | law A with a true node-4 hazard of 0 under always, n = 200, risk under never at horizon 4: the shipped interval, the curve's variance plus the supplement | bias inside the margin | bias -0.0018 to 0.0035, margin 0.0163; coverage 0.9297 to 0.9493; firing coverage 0.9609 to 0.9961 on 396 draws | pass |
<!-- /generated -->

| clause | rows | what it reads |
| --- | --- | --- |
| `bias` | every supplemented row | the 99% bias interval inside 0.25 SD |
| `firing floor` | every supplemented row with 600 or more expected firing draws (19 rows) | the exact 99% lower end of coverage on the firing draws at least 0.90 |
| `zero-cell coverage` | the always risk of both zero variants | the exact 99% lower end of all-draw coverage at least 0.92 |
| `paired gain` | the plug-in row of `interior_n200`, always | on the firing draws, the exact McNemar lower end of supplemented minus plug-in coverage above 0 |
| `node variance matters` | the same row | on the same firing draws, the exact 99% lower end of oracle minus plug-in coverage at least 0.05 |

The two paired clauses together attribute the change. The first says the supplement changes
coverage where the node fires. The second says the node's true variance changes it by more than
the gap between nominal coverage and the floor. The oracle interval contains the plug-in interval,
so the oracle can only gain. On law B the same reading is near 0, which is why law B carries no
paired control.

## Reported readings

These readings carry no verdict.

- On the control's 2,039 firing draws the plug-in covers 0.830, and its exact 99% upper end,
  0.851, lies below 0.95 (`firing_below_nominal`). So the plug-in falls short of nominal where
  the node fires, as well as short of the oracle.
- The quiet coverage of the always risk and the contrast at n = 100 and 200 has an exact lower end
  below 0.90 (`rare_event_limit`). A node flagged with probability $f$ holds about $-\ln f$
  expected events, so a quiet draw carries one or two events at the node that holds most of the
  variance. That is the ordinary interval's rare-event limit, recorded under
  [F31](../../roadmap.md#f31-inference-at-a-boundary-node-estimate), not a property of the
  supplement.
- For the risk under always, the supplemented interval is 1.6 to 1.8 times as wide as the oracle
  on the firing draws of the law A cells, 3.2 times in the zero cell and 1.07 on law B
  (`width_ratio_to_oracle`).
  At n = 100 that reading is NaN for the always risk and the contrast. Those cells hold withheld
  draws whose standard errors are NaN, and the reading takes a plain median. The fixed formula is
  the median over firing draws where both standard errors are finite, and the study publishes the
  committed value until its next declared run.

## Sizing and the first declared run

Every size was fixed before the first declared run from the 2,000-draw pilot in
`tests/studies/boundary_node_pilot.py`, on a seed stream no declared cell reads. The first
declared run (declaration `91cb39ae`, artifacts `cbbd634c`) followed the declared route for a
failed paired clause. Its clause (ii) compared the plug-in's coverage on firing draws with its
coverage on the non-firing draws of the same cell. Those draws under-cover for the rare-event
reason, so the clause compared two shortfalls.

The redeclaration replaced it with
`node variance matters`, wrote a withheld draw as a miss with a NaN standard error, and drew on
fresh seeds and stream labels. Every cell, size, margin and owner stayed as declared. The rows on
this page are the rerun's; the first run's passing cells are history, not evidence.

## Red-cell route

The study is registered under `reporting`. The two red primary SE-ratio legs are declared before
the run under `X54-supplement-width` in `tests/canonical/boundary_node_ltmle/regenerate.py`, which
also states the route of every other outcome.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | primary replications |
| `n` | 200 | observations per replication |
| `independent_tests_passed` | 1 | truth tests passing |
| `independent_tests_total` | 3 | truth tests reported |
| `property_cells_passed` | 25 | gated property cells passing |
| `property_cells_total` | 25 | gated property cells reported |
| `max_standardized_bias` | 0.0705 | largest primary standardized bias |
| `min_coverage` | 0.9270 | lowest primary coverage |
| `properties[boundary_node/interior_n200__always__plugin]:firing_coverage` | 0.8298 | plug-in coverage on the control's firing draws |
| `properties[boundary_node/interior_n200__always__plugin]:mcnemar_lower` | 0.1650 | clause (i) lower end |
| `properties[boundary_node/interior_n200__always__plugin]:oracle_gain_lower` | 0.1488 | clause (ii') lower end |
| `properties[boundary_node/interior_n200__always__supplemented]:firing_coverage` | 1 | supplemented coverage on the same firing draws |
| `properties[boundary_node/held_zero_n200__always__supplemented]:coverage` | 0.9660 | all-draw coverage of the law B zero cell |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound. No cell of this study reads it |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound. No cell of this study reads it |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound. No cell of this study reads it |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound. No cell of this study reads it |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs. No cell of this study reads it |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound. No cell of this study reads it |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | minimum power lower bound. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | expected root-n slope. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude. No cell of this study reads it |
| `margin:subject_failure_bound` | 0.0500 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 50 | the same bound, in primary replications |

## Limitations

- **A bound, not the node's variance.** The supplement adds a flagged node's variance at the
  one-sided bound of its hazard, so the interval is wider than one built on the true hazard and its
  SE ratio runs above 1.2 where the node holds most of the variance.
- **Two laws.** Law A on the iterated-mean path at n = 100 to 400, cross-fitted, clustered and with
  a zero hazard; law B on the hazard path at n = 200. Law B cannot show that the supplement is
  needed on the hazard path.
- **Known treatment mechanisms and correct learners.** Every fit reads the known mechanisms and a
  correctly specified outcome model.
- **The quiet draws keep the ordinary interval.** A draw without a flagged node shows the interval
  the package shipped before, and in these laws it under-covers at n = 100 and 200 for the
  rare-event reason that F31 keeps.
- **No comparator.** No other implementation adds the supplement.
