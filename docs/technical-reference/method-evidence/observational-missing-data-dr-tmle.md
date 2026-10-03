# Observational missing-data DR-TMLE

This study tests the composite-indicator construction for an observational missing outcome
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
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1: two-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ate` | average treatment effect | `cleverly` composite-indicator missing-data DR-TMLE | -0.0039 to 0.0036 | 0.9400 | 0.9831 | pass |
| L1: two-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ate` | average treatment effect | R `drtmle` with an NA treatment and a missing outcome | -0.0039 to 0.0036 | 0.9387 | 0.9827 | pass |
| L1: two-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ey0` | counterfactual mean under no treatment | `cleverly` composite-indicator missing-data DR-TMLE | -0.0027 to 0.0029 | 0.9325 | 0.9829 | pass |
| L1: two-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ey0` | counterfactual mean under no treatment | R `drtmle` with an NA treatment and a missing outcome | -0.0027 to 0.0029 | 0.9363 | 0.9829 | pass |
| L1: two-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ey1` | counterfactual mean under treatment | `cleverly` composite-indicator missing-data DR-TMLE | -0.0025 to 0.0023 | 0.9513 | 1.0149 | pass |
| L1: two-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ey1` | counterfactual mean under treatment | R `drtmle` with an NA treatment and a missing outcome | -0.0025 to 0.0023 | 0.9513 | 1.0145 | pass |
| L1: two-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `or` | marginal odds ratio, reported on the log scale | `cleverly` composite-indicator missing-data DR-TMLE | -0.0134 to 0.0183 | 0.9387 | 0.9823 | pass |
| L1: two-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `or` | marginal odds ratio, reported on the log scale | R `drtmle` with an NA treatment and a missing outcome | -0.0134 to 0.0183 | 0.9387 | 0.9819 | pass |
| L1: two-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `rr` | marginal risk ratio, reported on the log scale | `cleverly` composite-indicator missing-data DR-TMLE | -0.0067 to 0.0105 | 0.9413 | 0.9806 | pass |
| L1: two-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `rr` | marginal risk ratio, reported on the log scale | R `drtmle` with an NA treatment and a missing outcome | -0.0066 to 0.0106 | 0.9413 | 0.9803 | pass |
| L1: two-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `tmle_ate` | average treatment effect of the composite TMLE, paired with R `drtmle`'s out$tmle | `cleverly` composite-indicator missing-data DR-TMLE | -0.0039 to 0.0036 | 0.9375 | 0.9775 | pass |
| L1: two-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `tmle_ate` | average treatment effect of the composite TMLE, paired with R `drtmle`'s out$tmle | R `drtmle` with an NA treatment and a missing outcome | -0.0039 to 0.0036 | 0.9375 | 0.9775 | pass |
| L1: two-arm binary-outcome observational law with MAR outcomes | `ate` | average treatment effect | `cleverly` composite-indicator missing-data DR-TMLE | -0.0046 to 0.0015 | 0.9487 | 1.0050 | pass |
| L1: two-arm binary-outcome observational law with MAR outcomes | `ate` | average treatment effect | R `drtmle` with an NA treatment and a missing outcome | -0.0046 to 0.0015 | 0.9487 | 1.0048 | pass |
| L1: two-arm binary-outcome observational law with MAR outcomes | `ey0` | counterfactual mean under no treatment | `cleverly` composite-indicator missing-data DR-TMLE | -0.0017 to 0.0027 | 0.9475 | 1.0005 | pass |
| L1: two-arm binary-outcome observational law with MAR outcomes | `ey0` | counterfactual mean under no treatment | R `drtmle` with an NA treatment and a missing outcome | -0.0017 to 0.0027 | 0.9487 | 1.0006 | pass |
| L1: two-arm binary-outcome observational law with MAR outcomes | `ey1` | counterfactual mean under treatment | `cleverly` composite-indicator missing-data DR-TMLE | -0.0032 to 0.0011 | 0.9500 | 0.9952 | pass |
| L1: two-arm binary-outcome observational law with MAR outcomes | `ey1` | counterfactual mean under treatment | R `drtmle` with an NA treatment and a missing outcome | -0.0032 to 0.0011 | 0.9500 | 0.9946 | pass |
| L1: two-arm binary-outcome observational law with MAR outcomes | `or` | marginal odds ratio, reported on the log scale | `cleverly` composite-indicator missing-data DR-TMLE | -0.0171 to 0.0084 | 0.9450 | 1.0047 | pass |
| L1: two-arm binary-outcome observational law with MAR outcomes | `or` | marginal odds ratio, reported on the log scale | R `drtmle` with an NA treatment and a missing outcome | -0.0172 to 0.0084 | 0.9450 | 1.0045 | pass |
| L1: two-arm binary-outcome observational law with MAR outcomes | `rr` | marginal risk ratio, reported on the log scale | `cleverly` composite-indicator missing-data DR-TMLE | -0.0087 to 0.0049 | 0.9525 | 1.0064 | pass |
| L1: two-arm binary-outcome observational law with MAR outcomes | `rr` | marginal risk ratio, reported on the log scale | R `drtmle` with an NA treatment and a missing outcome | -0.0087 to 0.0048 | 0.9525 | 1.0063 | pass |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ate[low vs high]` | difference in counterfactual means, low versus high | `cleverly` composite-indicator missing-data DR-TMLE | -0.0030 to 0.0060 | 0.9437 | 0.9702 | pass |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ate[low vs high]` | difference in counterfactual means, low versus high | R `drtmle` with an NA treatment and a missing outcome | -0.0030 to 0.0060 | 0.9437 | 0.9712 | pass |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ate[mid vs high]` | difference in counterfactual means, mid versus high | `cleverly` composite-indicator missing-data DR-TMLE | -0.0020 to 0.0064 | 0.9475 | 0.9844 | pass |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ate[mid vs high]` | difference in counterfactual means, mid versus high | R `drtmle` with an NA treatment and a missing outcome | -0.0020 to 0.0064 | 0.9475 | 0.9849 | pass |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ey[high]` | counterfactual mean under treatment arm 'high' | `cleverly` composite-indicator missing-data DR-TMLE | -0.0054 to 0.0013 | 0.9225 | 0.9342 | **fail** |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ey[high]` | counterfactual mean under treatment arm 'high' | R `drtmle` with an NA treatment and a missing outcome | -0.0054 to 0.0013 | 0.9237 | 0.9349 | **fail** |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ey[low]` | counterfactual mean under treatment arm 'low' | `cleverly` composite-indicator missing-data DR-TMLE | -0.0036 to 0.0025 | 0.9450 | 0.9988 | pass |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ey[low]` | counterfactual mean under treatment arm 'low' | R `drtmle` with an NA treatment and a missing outcome | -0.0036 to 0.0025 | 0.9450 | 0.9994 | pass |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ey[mid]` | counterfactual mean under treatment arm 'mid' | `cleverly` composite-indicator missing-data DR-TMLE | -0.0025 to 0.0028 | 0.9437 | 0.9859 | pass |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ey[mid]` | counterfactual mean under treatment arm 'mid' | R `drtmle` with an NA treatment and a missing outcome | -0.0025 to 0.0028 | 0.9437 | 0.9859 | pass |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | `cleverly` composite-indicator missing-data DR-TMLE | -0.0208 to 0.0195 | 0.9437 | 0.9692 | pass |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | R `drtmle` with an NA treatment and a missing outcome | -0.0207 to 0.0195 | 0.9450 | 0.9700 | pass |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `or[mid vs high]` | marginal odds ratio, mid versus high, reported on the log scale | `cleverly` composite-indicator missing-data DR-TMLE | -0.0130 to 0.0240 | 0.9437 | 0.9822 | pass |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `or[mid vs high]` | marginal odds ratio, mid versus high, reported on the log scale | R `drtmle` with an NA treatment and a missing outcome | -0.0130 to 0.0240 | 0.9437 | 0.9827 | pass |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | `cleverly` composite-indicator missing-data DR-TMLE | -0.0104 to 0.0085 | 0.9450 | 0.9840 | pass |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | R `drtmle` with an NA treatment and a missing outcome | -0.0104 to 0.0085 | 0.9450 | 0.9849 | pass |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `rr[mid vs high]` | marginal risk ratio, mid versus high, reported on the log scale | `cleverly` composite-indicator missing-data DR-TMLE | -0.0036 to 0.0101 | 0.9537 | 0.9885 | pass |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `rr[mid vs high]` | marginal risk ratio, mid versus high, reported on the log scale | R `drtmle` with an NA treatment and a missing outcome | -0.0036 to 0.0101 | 0.9537 | 0.9890 | pass |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `tmle_ate_mid` | difference in counterfactual means, mid versus high, of the composite TMLE, paired with R `drtmle`'s out$tmle | `cleverly` composite-indicator missing-data DR-TMLE | -0.0019 to 0.0064 | 0.9475 | 0.9839 | pass |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `tmle_ate_mid` | difference in counterfactual means, mid versus high, of the composite TMLE, paired with R `drtmle`'s out$tmle | R `drtmle` with an NA treatment and a missing outcome | -0.0019 to 0.0064 | 0.9475 | 0.9839 | pass |
<!-- /generated -->

One arm mean is red in both implementations: `ey[high]` of the three-arm scenario. Its coverage
is 0.9225 in the package and 0.9237 in R `drtmle`, and each 99% interval ends below the 0.90
floor. The two implementations differ by 4e-6 on average. `high` has the smallest composite
mechanism of the law, 0.070 where `W = 0`. The roadmap owner
[`composite-high-arm`](../../roadmap.md#red-cell-owners) records the reading. The row stays red
under the `reporting` policy.

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L1: two-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ate` | average treatment effect | -0.000015 | 0.0024 | 1.0007 | 0.0012 | 0.0013 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ey0` | counterfactual mean under no treatment | 0.000003 | 0.000578 | 1.0019 | -0.0037 | 0.0010 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ey1` | counterfactual mean under treatment | -0.000012 | 0.0031 | 1.0002 | 0 | 0.000654 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `or` | marginal odds ratio, reported on the log scale | -0.000134 | 0.0021 | 1.0008 | 0 | 0.0013 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `rr` | marginal risk ratio, reported on the log scale | -0.000030 | 0.0013 | 1.0011 | 0 | 0.0012 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `tmle_ate` | average treatment effect of the composite TMLE, paired with R `drtmle`'s out$tmle | 1.668e-09 | 2.704e-07 | 1.0000 | 0 | 5.707e-09 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome observational law with MAR outcomes | `ate` | average treatment effect | 0.000009 | 0.0017 | 1.0002 | 0 | 0.000534 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome observational law with MAR outcomes | `ey0` | counterfactual mean under no treatment | -0.000003 | 0.000798 | 1.0004 | -0.0012 | 0.000595 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome observational law with MAR outcomes | `ey1` | counterfactual mean under treatment | 0.000006 | 0.0016 | 0.9999 | 0 | 0.0017 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome observational law with MAR outcomes | `or` | marginal odds ratio, reported on the log scale | 0.000070 | 0.0014 | 1.0003 | 0 | 0.000530 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome observational law with MAR outcomes | `rr` | marginal risk ratio, reported on the log scale | 0.000027 | 0.0016 | 1.0004 | 0 | 0.000460 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ate[low vs high]` | difference in counterfactual means, low versus high | -0.000007 | 0.000913 | 1.0015 | 0 | 0.000717 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ate[mid vs high]` | difference in counterfactual means, mid versus high | -0.000001 | 0.000205 | 1.0013 | 0 | 0.000935 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ey[high]` | counterfactual mean under treatment arm 'high' | 0.000004 | 0.000791 | 1.0014 | -0.0012 | 0.000952 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ey[low]` | counterfactual mean under treatment arm 'low' | -0.000002 | 0.000479 | 1.0014 | 0 | 0.000684 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `ey[mid]` | counterfactual mean under treatment arm 'mid' | 0.000003 | 0.000665 | 1.0008 | 0 | 0.000972 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | -6.250e-08 | 0.000006 | 1.0017 | -0.0012 | 0.000738 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `or[mid vs high]` | marginal odds ratio, mid versus high, reported on the log scale | 0.000002 | 0.000088 | 1.0014 | 0 | 0.000914 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | -0.000004 | 0.000479 | 1.0015 | 0 | 0.000658 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `rr[mid vs high]` | marginal risk ratio, mid versus high, reported on the log scale | 0.000002 | 0.000208 | 1.0012 | 0 | 0.000917 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome observational law with a MAR outcome and a MAR treatment whose recording depends on the treatment | `tmle_ate_mid` | difference in counterfactual means, mid versus high, of the composite TMLE, paired with R `drtmle`'s out$tmle | -2.193e-10 | 3.188e-08 | 1.0000 | 0 | 4.432e-09 vs 0.0500 | equivalent |
<!-- /generated -->

## Repeated-sampling properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `corrected_mar_inference` | `composite_binary_ate__both_correct` | positive | the ATE of the two-arm law with a MAR outcome and treatment: the outcome regression and the composite mechanism are correct | bias interval inside the margin, coverage clears the floor, SE ratio inside the band, SE ratio must remain between 0.1 and 10.0 | bias -0.0041 to 0.0036, coverage 0.9208 to 0.9637, SE ratio 0.9675 | pass |
| `corrected_mar_inference` | `composite_binary_ate__both_wrong` | control | the ATE of the two-arm law with a MAR outcome and treatment: the outcome regression and the composite mechanism are both misspecified | bias interval must fall entirely outside the margin, and its distance from zero must clear the declared floor, SE ratio must remain between 0.1 and 10.0 | bias -0.0530 to -0.0451, coverage 0.7308 to 0.8084, SE ratio 0.9468 | pass |
| `corrected_mar_inference` | `composite_binary_ate__mechanism_drift` | positive | the ATE of the two-arm law with a MAR outcome and treatment: the outcome regression is correct and the treatment factor and treatment observation factor of the composite mechanism are misspecified | bias interval inside the margin, coverage clears the floor, SE ratio inside the band, SE ratio must remain between 0.1 and 10.0 | bias -0.0063 to 0.000835, coverage 0.9416 to 0.9776, SE ratio 1.1390 | pass |
| `corrected_mar_inference` | `composite_binary_ate__outcome_drift` | positive | the ATE of the two-arm law with a MAR outcome and treatment: the outcome regression is misspecified and the composite mechanism is correct | bias interval inside the margin, coverage clears the floor, SE ratio inside the band, SE ratio must remain between 0.1 and 10.0 | bias -0.0016 to 0.0060, coverage 0.9267 to 0.9677, SE ratio 1.0077 | pass |
| `corrected_mar_inference` | `composite_observational_ate__both_correct` | positive | the ATE of the two-arm law with an observational MAR outcome: the outcome regression and the composite mechanism are correct | bias interval inside the margin, coverage clears the floor, SE ratio inside the band, SE ratio must remain between 0.1 and 10.0 | bias -0.0044 to 0.0018, coverage 0.9208 to 0.9637, SE ratio 0.9767 | pass |
| `corrected_mar_inference` | `composite_observational_ate__both_wrong` | control | the ATE of the two-arm law with an observational MAR outcome: the outcome regression and the composite mechanism are both misspecified | bias interval must fall entirely outside the margin, and its distance from zero must clear the declared floor, SE ratio must remain between 0.1 and 10.0 | bias -0.0941 to -0.0871, coverage 0.2068 to 0.2863, SE ratio 0.8904 | pass |
| `corrected_mar_inference` | `composite_observational_ate__mechanism_drift` | positive | the ATE of the two-arm law with an observational MAR outcome: the outcome regression is correct and the treatment factor and treatment observation factor of the composite mechanism are misspecified | bias interval inside the margin, coverage clears the floor, SE ratio inside the band, SE ratio must remain between 0.1 and 10.0 | bias -0.0022 to 0.0038, coverage 0.9297 to 0.9698, SE ratio 1.0626 | pass |
| `corrected_mar_inference` | `composite_observational_ate__outcome_drift` | positive | the ATE of the two-arm law with an observational MAR outcome: the outcome regression is misspecified and the composite mechanism is correct | bias interval inside the margin, coverage clears the floor, SE ratio inside the band, SE ratio must remain between 0.1 and 10.0 | bias -0.0029 to 0.0034, coverage 0.9282 to 0.9688, SE ratio 1.0063 | pass |
| `corrected_mar_inference` | `composite_three_arm_ate_low__both_correct` | positive | the difference low versus high of the three-arm law with a MAR outcome and treatment: the outcome regression and the composite mechanism are correct | bias interval inside the margin, coverage clears the floor, SE ratio inside the band, SE ratio must remain between 0.1 and 10.0 | bias -0.0065 to 0.0026, coverage 0.9238 to 0.9657, SE ratio 0.9564 | pass |
| `corrected_mar_inference` | `composite_three_arm_ate_low__both_wrong` | control | the difference low versus high of the three-arm law with a MAR outcome and treatment: the outcome table's arm columns rolled, a uniform treatment factor, and a wrong treatment observation factor: a drift of this contrast's own | bias interval must fall entirely outside the margin, and its distance from zero must clear the declared floor, SE ratio must remain between 0.1 and 10.0 | bias 0.0166 to 0.0255, coverage 0.9035 to 0.9512, SE ratio 1.0048 | pass |
| `corrected_mar_inference` | `composite_three_arm_ate_low__mechanism_drift` | positive | the difference low versus high of the three-arm law with a MAR outcome and treatment: the outcome regression is correct and the treatment factor and treatment observation factor of the composite mechanism are misspecified | bias interval inside the margin, coverage clears the floor, SE ratio inside the band, SE ratio must remain between 0.1 and 10.0 | bias -0.0014 to 0.0076, coverage 0.9208 to 0.9637, SE ratio 0.9703 | pass |
| `corrected_mar_inference` | `composite_three_arm_ate_low__outcome_drift` | positive | the difference low versus high of the three-arm law with a MAR outcome and treatment: the outcome regression is misspecified and the composite mechanism is correct | bias interval inside the margin, coverage clears the floor, SE ratio inside the band, SE ratio must remain between 0.1 and 10.0 | bias -0.0071 to 0.0020, coverage 0.9267 to 0.9677, SE ratio 0.9798 | pass |
| `corrected_mar_inference` | `composite_three_arm_ate_mid__both_correct` | positive | the difference mid versus high of the three-arm law with a MAR outcome and treatment: the outcome regression and the composite mechanism are correct | bias interval inside the margin, coverage clears the floor, SE ratio inside the band, SE ratio must remain between 0.1 and 10.0 | bias -0.0061 to 0.0023, coverage 0.9092 to 0.9554, SE ratio 0.9852 | pass |
| `corrected_mar_inference` | `composite_three_arm_ate_mid__both_wrong` | control | the difference mid versus high of the three-arm law with a MAR outcome and treatment: the outcome regression and the composite mechanism are both misspecified | bias interval must fall entirely outside the margin, and its distance from zero must clear the declared floor, SE ratio must remain between 0.1 and 10.0 | bias -0.0797 to -0.0711, coverage 0.5644 to 0.6543, SE ratio 1.0003 | pass |
| `corrected_mar_inference` | `composite_three_arm_ate_mid__mechanism_drift` | positive | the difference mid versus high of the three-arm law with a MAR outcome and treatment: the outcome regression is correct and the treatment factor and treatment observation factor of the composite mechanism are misspecified | bias interval inside the margin, coverage clears the floor, SE ratio inside the band, SE ratio must remain between 0.1 and 10.0 | bias 0.000142 to 0.0081, coverage 0.9356 to 0.9737, SE ratio 1.0469 | pass |
| `corrected_mar_inference` | `composite_three_arm_ate_mid__outcome_drift` | positive | the difference mid versus high of the three-arm law with a MAR outcome and treatment: the outcome regression is misspecified and the composite mechanism is correct | bias interval inside the margin, coverage clears the floor, SE ratio inside the band, SE ratio must remain between 0.1 and 10.0 | bias -0.0072 to 0.0013, coverage 0.9223 to 0.9647, SE ratio 0.9826 | pass |
| `correction_necessity` | `composite_cycle__closed_score` | positive | composite-indicator correction cycle: the largest equation-(9) and equation-(10) scores over the arms after the cycle, under the outcome drift | the upper confidence endpoint is below the declared fraction of the initial-score lower endpoint | score 4.071e-11 to 4.475e-11 | pass |
| `correction_necessity` | `composite_cycle__initial_score_control` | control | composite-indicator correction cycle: the largest equation-(10) score over the arms before the cycle is run | the lower confidence endpoint clears the declared unresolved-score floor | score 0.0024 to 0.0027 | pass |
| `interval_calibration` | `ate__correctly_specified` | positive | average treatment effect: all four required nuisance functions are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9255 to 0.9511, SE ratio 0.9305 to 1.0022, empirical efficiency ratio 0.9975 to 1.0741, reported efficiency ratio 0.9975 to 1.0018 | pass |
| `interval_calibration` | `ate__noise_control` | control | average treatment effect: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8046 to 0.8449, SE ratio 0.6677 to 0.7185, empirical efficiency ratio 1.3914 to 1.4974, reported efficiency ratio 0.9976 to 1.0019 | pass |
| `interval_calibration` | `ate__shrunken_se_control` | control | average treatment effect: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7894 to 0.8310, SE ratio 0.6524 to 0.7016, empirical efficiency ratio 0.9973 to 1.0724, reported efficiency ratio 0.6983 to 0.7013 | pass |
| `ordinary_targeting` | `binary` | positive | the composite TMLE on the two-arm law with a MAR outcome and treatment | bias interval inside the margin, coverage lower bound at least 0.90, SE ratio inside (0.80, 1.20) | bias -0.0034 to 0.0042, coverage 0.9136 to 0.9585, SE ratio 0.9659 | pass |
| `ordinary_targeting` | `binary_msm` | positive | the composite TMLE of the linear arm MSM slope on the two-arm law with a MAR outcome and treatment | bias interval inside the margin, coverage lower bound at least 0.90, SE ratio inside (0.80, 1.20) | bias -0.0021 to 0.0051, coverage 0.9252 to 0.9667, SE ratio 1.0247 | pass |
| `ordinary_targeting` | `binary_regime` | positive | the composite TMLE of a known W-dependent regime mean on the two-arm law with a MAR outcome and treatment | bias interval inside the margin, coverage lower bound at least 0.90, SE ratio inside (0.80, 1.20) | bias -0.0027 to 0.0018, coverage 0.9121 to 0.9575, SE ratio 1.0148 | pass |
| `ordinary_targeting` | `three_arm` | positive | the composite TMLE on the three-arm law, mid versus high | bias interval inside the margin, coverage lower bound at least 0.90, SE ratio inside (0.80, 1.20) | bias -0.0036 to 0.0043, coverage 0.9386 to 0.9757, SE ratio 1.0442 | pass |
| `power` | `ate` | positive | the same test applied to the binary missing-treatment law's ATE of 0.21, at n = 1,000 | rejection lower bound clears the minimum power | rejection 0.9350, 0.9145 to 0.9520 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000450, coverage 0.9182 to 0.9549, SE ratio 0.9668 | pass |
| `root_n_and_efficiency` | `n_500` | control | bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.000127, coverage 0.9258 to 0.9606, SE ratio 0.9742 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0014, coverage 0.9296 to 0.9634, SE ratio 0.9808 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5292 to -0.4748 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5017 to -0.4969 | pass |
| `simultaneous_coverage` | `composite_binary__pointwise_joint_control` | control | the two arm means, the difference, the risk ratio and the odds ratio of the two-arm law with a MAR outcome and treatment: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8474 to 0.8836 | pass |
| `simultaneous_coverage` | `composite_binary__simultaneous_band` | positive | the two arm means, the difference, the risk ratio and the odds ratio of the two-arm law with a MAR outcome and treatment: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9250 to 0.9507 | pass |
| `simultaneous_coverage` | `composite_observational__pointwise_joint_control` | control | the two arm means, the difference, the risk ratio and the odds ratio of the two-arm law with an observational MAR outcome: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8659 to 0.9000 | pass |
| `simultaneous_coverage` | `composite_observational__simultaneous_band` | positive | the two arm means, the difference, the risk ratio and the odds ratio of the two-arm law with an observational MAR outcome: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9337 to 0.9578 | pass |
| `simultaneous_coverage` | `composite_three_arm__pointwise_joint_control` | control | the three arm means, two differences, two risk ratios and two odds ratios of the three-arm law with a MAR outcome and treatment: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.7994 to 0.8402 | pass |
| `simultaneous_coverage` | `composite_three_arm__simultaneous_band` | positive | the three arm means, two differences, two risk ratios and two odds ratios of the three-arm law with a MAR outcome and treatment: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9383 to 0.9615 | pass |
| `treatment_complete_case` | `drop_unrecorded__control` | control | the rows with an unrecorded treatment dropped, and the rest fitted with delta= alone | bias interval must fall entirely outside the margin | bias -0.0487 to -0.0422, margin 0.0089 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0533, 0.0380 to 0.0723 | pass |
<!-- /generated -->

The study declared every budget, seed, law, learner, size, drift and floor before any verdict. The
policy is `reporting`: a red cell stays red at its budget and margin, and the
[red-cell ledger](red-cells.md) records it.

| family | what it tests |
| --- | --- |
| `corrected_mar_inference` | each contrast with both nuisances correct, the outcome regression wrong, the composite mechanism wrong, and both wrong. The outcome drift is the constant regression $\bar Q = 0.5$. The mechanism drift replaces $g(a \mid \Delta_A = 1, W)$ and, where the treatment can be missing, $P(\Delta_A = 1 \mid W)$. The both-wrong control must clear a declared floor of half its large-sample bias. Under the shared drifts the three-arm `low` contrast moves by at most 0.010, inside the margin. Its control therefore runs under its own drift: the outcome table's arm columns rolled, a uniform treatment factor, and the wrong observation factor |
| `ordinary_targeting` | the composite TMLE, `guard=()`, on the two missing-treatment scenarios. Two further cells fit the binary missing-treatment scenario: the mean of a known `W`-dependent regime, and the slope of the linear arm MSM |
| `root_n_and_efficiency`, `root_n_rate` | the `ate` of the binary missing-treatment scenario at n = 500, 2,000 and 8,000, and the efficiency ratio against the exact EIF SD |
| `interval_calibration` | the same `ate` against the calibration bands, with a shrunken-SE and a noise control |
| `simultaneous_coverage` | the default band over every reported name of each scenario, and its pointwise joint control |
| `type_i_error`, `power` | the `ate` at a sharp null, and at n = 1,000 |
| `correction_necessity` | the extra scores of every arm after and before the correction cycle, under the outcome drift |
| `treatment_complete_case` | a control that drops the rows with an unrecorded treatment. It must miss the truth |

Each property row and each package primary row records how the fit's DR-TMLE outer loop ended,
in `exit_reason` and `rounds`. `fit-exits.csv` publishes the primary fits' records. A composite
TMLE fit has no outer loop and records `none`. A fit that reaches `max_outer=100` rounds records
`cap`. It counts as it is when its scores pass. The study does not raise `max_outer` after a result.
Of the 2,400 primary DR-TMLE fits, 49 reached the cap and 9 stalled, 47 and 8 of them on the
three-arm scenario.

The power cell runs at n = 1,000, not at the plan's n = 500. At n = 500 the planned power of the
two-sided test is 0.738, and at n = 1,000 it is 0.957. The design of the band cells is in
`tests/unit/test_simultaneous_cell_design.py`: pointwise joint coverage 0.8832, 0.8835 and 0.8220,
and control power 1.0 at 2,400 replications.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | paired replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_passed` | 40 | truth tests passing |
| `independent_tests_total` | 42 | truth tests reported |
| `paired_tests_passed` | 21 | paired comparisons passing |
| `paired_tests_total` | 21 | paired comparisons reported |
| `property_cells_passed` | 39 | property cells passing |
| `property_cells_total` | 39 | property cells reported |
| `max_standardized_bias` | 0.0565 | largest primary standardized bias |
| `min_coverage` | 0.9225 | lowest primary coverage |
| `max_margin_utilization` | 0.0031 | largest paired similarity-margin share |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | nominal test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound |
| `margin:minimum_power` | 0.8000 | minimum power lower bound |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:efficiency_ratio_lower` | 0.9000 | efficiency-ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | efficiency-ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:correction_score_ratio` | 0.0100 | maximum closed-to-initial score-endpoint ratio |
| `margin:uncorrected_score_floor` | 0.0010 | unresolved initial-score floor |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit. No cell of this study reads it |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit. No cell of this study reads it |

## Limits

- The study covers a binary outcome, one three-level baseline covariate, and finite-support
  oracle primaries. The reduced regressions are linear and logistic models fitted in sample.
- Partial guards, the bivariate reduction, weights and clusters have no coverage cell of their
  own. The exact laws of `tests/unit/test_composite_missing_data.py` cover the guards, the
  reductions and a weight. The [weighted](weighted-point-treatment-tmle.md) and
  [clustered](clustered-point-treatment-cv-tmle.md) studies cover the weight and cluster algebra,
  which the composite does not change.
- The composite is tilted inside $[10^{-6}, 0.99]$ at `g_bounds=(0.01, 0.99)` and
  `nuisance_bound=0.01` with both indicators, against $10^{-4}$ with a missing outcome alone.
  The default `g_bounds="auto"` floors the treatment factor at $5 / (\sqrt{n} \ln n)$ instead.
  No scenario reaches that floor.
- No instrument here can detect a violation of the treatment condition, $Y(a)$ independent of
  $\Delta_A$ given $(A, W)$.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_composite/README.md),
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_composite/manifest.json),
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_composite/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_composite/equivalence.csv),
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_composite/properties.csv)
carry the protocol, provenance, and every published row.
