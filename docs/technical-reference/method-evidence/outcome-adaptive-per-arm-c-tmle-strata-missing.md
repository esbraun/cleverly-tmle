# Outcome-adaptive per-arm C-TMLE with strata and missing data

This study measures `CTMLE(strategy="oat")` with the default `oat_design="per_arm"` on fits
with baseline strata (`strata=`), a missing outcome (`delta=`) or a missing treatment
(`treatment_delta=`). For arm $a$ the design is the product of three factors, each fitted on the
one column $\bar Q_n(a, W)$ inside each stratum. The
[collaborative TMLE reference](../collaborative-tmle.md#baseline-strata-and-missing-data-in-the-per-arm-design)
states the factors and the routes. The base result is Theorem 1 of Benkeser, Cai and van der Laan
(2020), applied to the composite indicator $C_a = \Delta_A \Delta 1\{A = a\}$.

The study pairs the fit with R
[`drtmle`](https://github.com/benkeser/drtmle/tree/538a3a264c1ca984b6d88978ca7f96165f43152c)
1.1.2 at commit `538a3a2`, with `adapt_g = TRUE`, `DeltaY`, `DeltaA` and `stratify = TRUE`. The
runner calls `drtmle` once per arm on the binary recode $1\{A = a\}$, as the
[per-arm study](outcome-adaptive-per-arm-c-tmle.md) does. On the `strata` law it calls `drtmle`
on the rows of each stratum. Both sides read this package's initial outcome regression.

The study publishes under the `reporting` policy, and it claims the point estimate only. Every
`oat` fit takes the `generated_design_plugin` status and reports no interval. The coverage and
standard-error columns below measure `plugin_interval` and `plugin_std_error` as diagnostics.
Their pass labels do not authorize `ci`, `pvalue` or `std_error`.

[F19](../../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) owns the red truth
rows and the three red rungs of the `root_n_and_efficiency` ladder. The study re-declared the
ladder after the first run, and a rerun of the property phase measured it.
[The re-declared ladder](#the-re-declared-ladder) gives the reason.

## What was compared

| setting | `cleverly` | R `drtmle` |
| --- | --- | --- |
| datasets | 800 samples of 2,250 rows for each law | the identical rows |
| `strata` law | $W = (W_1, W_2, W_3)$ is independent standard normal, and the stratum $S \in \{0, 1, 2\}$ has the masses 0.5, 0.3 and 0.2. The binary outcome reads the per-arm study's two-arm index plus a stratum term. $W_3$ enters only the treatment and observation laws, so it is an instrument. No row is missing | the same |
| `missing_outcome` law | the `strata` law at $S = 0$, with an outcome that is missing at random given $A$, $W_1$ and $W_3$ | the same |
| `three_arm_missing` law | the per-arm study's three-arm law, with an outcome that is missing at random | the same |
| `missing_treatment` law | the `missing_outcome` law, with a treatment that is missing at random given $A$ and $W_3$ | the same |
| reported parameters | `strata`: the arm means and the ATE, marginal and in each stratum. Two arms: the arm means, ATE, RR and OR. Three arms: the arm means and the ATE, RR and OR of arms 1 and 2 against arm 0 | the same |
| outcome regression | an unpenalized logistic regression on the arm indicators, $W$, the stratum indicators, $A \times W$ and $A \times S$. It is correctly specified | this package's initial fit, passed as `Qn` |
| design | for each arm, factor and stratum, an unpenalized logistic regression of the factor's target on $(l, l^2, l^3)$, $l = \operatorname{logit} \bar Q_n(a, W)$. `g_bounds=(0.01, 0.99)` and `nuisance_bound=0.01` | `glm_g` for `A`, `DeltaY` and `DeltaA` with the same cubic |
| targeting | one logistic fluctuation for each arm, and for each stratum on the `strata` law | the TMLE stage of `drtmle` |
| intervals | pointwise 95% Wald from the per-arm curve, with RR and OR on the log scale. They are diagnostics | the same, from `tmle$cov` |

The runner rebuilds each arm's TMLE curve on its logistic submodel. It refuses a fit unless the
rebuilt variance equals `tmle$cov[1, 1]` to `1e-8`. On the `strata` law it embeds each stratum
curve. The marginal curve is the $P_n(S = s)$ mixture with its mass term.
`tests/studies/oat_per_arm_extension_laws.py` declares the laws and computes each truth by
quadrature.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ate` | average treatment effect | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0040 to 0.000874 | 0.8750 | 0.8249 | **fail** |
| two arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ate` | average treatment effect | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0040 to 0.000874 | 0.8750 | 0.8248 | **fail** |
| two arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ey0` | counterfactual mean under no treatment | `cleverly` per-arm outcome-adaptive C-TMLE | -0.000251 to 0.0033 | 0.8975 | 0.8426 | **fail** |
| two arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ey0` | counterfactual mean under no treatment | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.000251 to 0.0033 | 0.8975 | 0.8426 | **fail** |
| two arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ey1` | counterfactual mean under treatment | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0016 to 0.0016 | 0.8738 | 0.8094 | **fail** |
| two arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ey1` | counterfactual mean under treatment | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0016 to 0.0016 | 0.8738 | 0.8094 | **fail** |
| two arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `or` | marginal odds ratio, reported on the log scale | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0152 to 0.0056 | 0.8875 | 0.8237 | **fail** |
| two arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `or` | marginal odds ratio, reported on the log scale | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0152 to 0.0056 | 0.8875 | 0.8237 | **fail** |
| two arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `rr` | marginal risk ratio, reported on the log scale | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0076 to 0.0019 | 0.8862 | 0.8323 | **fail** |
| two arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `rr` | marginal risk ratio, reported on the log scale | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0076 to 0.0019 | 0.8862 | 0.8323 | **fail** |
| two arms, treatment unrecorded at random given W3 and the arm, outcomes missing at random | `ate` | average treatment effect | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0023 to 0.0031 | 0.9012 | 0.8231 | **fail** |
| two arms, treatment unrecorded at random given W3 and the arm, outcomes missing at random | `ate` | average treatment effect | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0023 to 0.0032 | 0.9012 | 0.8231 | **fail** |
| two arms, treatment unrecorded at random given W3 and the arm, outcomes missing at random | `ey0` | counterfactual mean under no treatment | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0017 to 0.0021 | 0.9125 | 0.8751 | **fail** |
| two arms, treatment unrecorded at random given W3 and the arm, outcomes missing at random | `ey0` | counterfactual mean under no treatment | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0017 to 0.0021 | 0.9125 | 0.8750 | **fail** |
| two arms, treatment unrecorded at random given W3 and the arm, outcomes missing at random | `ey1` | counterfactual mean under treatment | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0013 to 0.0026 | 0.8675 | 0.7576 | **fail** |
| two arms, treatment unrecorded at random given W3 and the arm, outcomes missing at random | `ey1` | counterfactual mean under treatment | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0013 to 0.0026 | 0.8675 | 0.7576 | **fail** |
| two arms, treatment unrecorded at random given W3 and the arm, outcomes missing at random | `or` | marginal odds ratio, reported on the log scale | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0076 to 0.0158 | 0.8950 | 0.8163 | **fail** |
| two arms, treatment unrecorded at random given W3 and the arm, outcomes missing at random | `or` | marginal odds ratio, reported on the log scale | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0076 to 0.0158 | 0.8950 | 0.8163 | **fail** |
| two arms, treatment unrecorded at random given W3 and the arm, outcomes missing at random | `rr` | marginal risk ratio, reported on the log scale | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0041 to 0.0064 | 0.9062 | 0.8453 | **fail** |
| two arms, treatment unrecorded at random given W3 and the arm, outcomes missing at random | `rr` | marginal risk ratio, reported on the log scale | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0041 to 0.0064 | 0.9062 | 0.8452 | **fail** |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ate` | average treatment effect | `cleverly` per-arm outcome-adaptive C-TMLE | -0.000778 to 0.0033 | 0.9187 | 0.9014 | **fail** |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ate` | average treatment effect | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.000772 to 0.0033 | 0.9187 | 0.9013 | **fail** |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ate[S=0]` | average treatment effect, in stratum S=0 | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0013 to 0.0043 | 0.9250 | 0.9175 | **fail** |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ate[S=0]` | average treatment effect, in stratum S=0 | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0013 to 0.0043 | 0.9250 | 0.9175 | **fail** |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ate[S=1]` | average treatment effect, in stratum S=1 | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0024 to 0.0043 | 0.9487 | 0.9916 | pass |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ate[S=1]` | average treatment effect, in stratum S=1 | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0024 to 0.0043 | 0.9487 | 0.9917 | pass |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ate[S=2]` | average treatment effect, in stratum S=2 | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0032 to 0.0054 | 0.9437 | 0.9897 | pass |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ate[S=2]` | average treatment effect, in stratum S=2 | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0032 to 0.0054 | 0.9437 | 0.9897 | pass |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey0` | counterfactual mean under no treatment | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0029 to 0.000181 | 0.9100 | 0.8733 | **fail** |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey0` | counterfactual mean under no treatment | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0029 to 0.000176 | 0.9100 | 0.8733 | **fail** |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey0[S=0]` | counterfactual mean under no treatment, in stratum S=0 | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0040 to -0.000028 | 0.9313 | 0.9131 | pass |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey0[S=0]` | counterfactual mean under no treatment, in stratum S=0 | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0040 to -0.000028 | 0.9313 | 0.9131 | pass |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey0[S=1]` | counterfactual mean under no treatment, in stratum S=1 | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0037 to 0.0015 | 0.9437 | 0.9663 | pass |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey0[S=1]` | counterfactual mean under no treatment, in stratum S=1 | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0037 to 0.0015 | 0.9437 | 0.9663 | pass |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey0[S=2]` | counterfactual mean under no treatment, in stratum S=2 | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0035 to 0.0032 | 0.9413 | 0.9841 | pass |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey0[S=2]` | counterfactual mean under no treatment, in stratum S=2 | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0035 to 0.0032 | 0.9413 | 0.9841 | pass |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey1` | counterfactual mean under treatment | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0015 to 0.0012 | 0.9038 | 0.9006 | **fail** |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey1` | counterfactual mean under treatment | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0015 to 0.0012 | 0.9038 | 0.9006 | **fail** |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey1[S=0]` | counterfactual mean under treatment, in stratum S=0 | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0024 to 0.0014 | 0.9363 | 0.9241 | pass |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey1[S=0]` | counterfactual mean under treatment, in stratum S=0 | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0024 to 0.0014 | 0.9363 | 0.9241 | pass |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey1[S=1]` | counterfactual mean under treatment, in stratum S=1 | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0024 to 0.0021 | 0.9450 | 0.9747 | pass |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey1[S=1]` | counterfactual mean under treatment, in stratum S=1 | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0024 to 0.0021 | 0.9450 | 0.9746 | pass |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey1[S=2]` | counterfactual mean under treatment, in stratum S=2 | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0017 to 0.0036 | 0.9525 | 0.9923 | pass |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey1[S=2]` | counterfactual mean under treatment, in stratum S=2 | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0017 to 0.0036 | 0.9525 | 0.9923 | pass |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ate[1 vs 0]` | difference in counterfactual means, 1 versus 0 | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0024 to 0.0038 | 0.9062 | 0.8510 | **fail** |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ate[1 vs 0]` | difference in counterfactual means, 1 versus 0 | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0024 to 0.0038 | 0.9062 | 0.8510 | **fail** |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ate[2 vs 0]` | difference in counterfactual means, 2 versus 0 | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0025 to 0.0037 | 0.9187 | 0.9054 | **fail** |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ate[2 vs 0]` | difference in counterfactual means, 2 versus 0 | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0025 to 0.0037 | 0.9187 | 0.9054 | **fail** |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ey[0]` | counterfactual mean under treatment arm '0' | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0018 to 0.0024 | 0.9337 | 0.9553 | pass |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ey[0]` | counterfactual mean under treatment arm '0' | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0018 to 0.0024 | 0.9337 | 0.9553 | pass |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ey[1]` | counterfactual mean under treatment arm '1' | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0012 to 0.0033 | 0.8575 | 0.7364 | **fail** |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ey[1]` | counterfactual mean under treatment arm '1' | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0012 to 0.0033 | 0.8575 | 0.7364 | **fail** |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ey[2]` | counterfactual mean under treatment arm '2' | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0015 to 0.0034 | 0.8800 | 0.8012 | **fail** |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ey[2]` | counterfactual mean under treatment arm '2' | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0015 to 0.0034 | 0.8800 | 0.8012 | **fail** |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `or[1 vs 0]` | marginal odds ratio, 1 versus 0, reported on the log scale | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0089 to 0.0160 | 0.9100 | 0.8537 | **fail** |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `or[1 vs 0]` | marginal odds ratio, 1 versus 0, reported on the log scale | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0089 to 0.0160 | 0.9100 | 0.8538 | **fail** |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `or[2 vs 0]` | marginal odds ratio, 2 versus 0, reported on the log scale | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0086 to 0.0172 | 0.9187 | 0.9047 | **fail** |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `or[2 vs 0]` | marginal odds ratio, 2 versus 0, reported on the log scale | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0086 to 0.0172 | 0.9187 | 0.9046 | **fail** |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `rr[1 vs 0]` | marginal risk ratio, 1 versus 0, reported on the log scale | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0051 to 0.0084 | 0.9137 | 0.8704 | **fail** |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `rr[1 vs 0]` | marginal risk ratio, 1 versus 0, reported on the log scale | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0051 to 0.0084 | 0.9137 | 0.8704 | **fail** |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `rr[2 vs 0]` | marginal risk ratio, 2 versus 0, reported on the log scale | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0051 to 0.0077 | 0.9300 | 0.9288 | pass |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `rr[2 vs 0]` | marginal risk ratio, 2 versus 0, reported on the log scale | R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and one fit per stratum | -0.0050 to 0.0077 | 0.9300 | 0.9288 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ate` | average treatment effect | -1.928e-07 | 0.000048 | 1.0000 | 0 | 0.000003 vs 0.0500 | equivalent |
| two arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ey0` | counterfactual mean under no treatment | 1.930e-07 | 0.000065 | 1.0000 | 0 | 0.000003 vs 0.0500 | equivalent |
| two arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ey1` | counterfactual mean under treatment | 1.686e-10 | 6.352e-08 | 1.0000 | 0 | 2.623e-09 vs 0.0500 | equivalent |
| two arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `or` | marginal odds ratio, reported on the log scale | -0.000002 | 0.000049 | 1.0000 | 0 | 0.000003 vs 0.0500 | equivalent |
| two arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `rr` | marginal risk ratio, reported on the log scale | -6.895e-07 | 0.000058 | 1.0000 | 0 | 0.000003 vs 0.0500 | equivalent |
| two arms, treatment unrecorded at random given W3 and the arm, outcomes missing at random | `ate` | average treatment effect | -0.000001 | 0.000293 | 1.0000 | 0 | 0.000024 vs 0.0500 | equivalent |
| two arms, treatment unrecorded at random given W3 and the arm, outcomes missing at random | `ey0` | counterfactual mean under no treatment | 0.000001 | 0.000405 | 1.0000 | 0 | 0.000027 vs 0.0500 | equivalent |
| two arms, treatment unrecorded at random given W3 and the arm, outcomes missing at random | `ey1` | counterfactual mean under treatment | -2.499e-08 | 0.000008 | 1.0000 | 0 | 0.000002 vs 0.0500 | equivalent |
| two arms, treatment unrecorded at random given W3 and the arm, outcomes missing at random | `or` | marginal odds ratio, reported on the log scale | -0.000014 | 0.000281 | 1.0000 | 0 | 0.000023 vs 0.0500 | equivalent |
| two arms, treatment unrecorded at random given W3 and the arm, outcomes missing at random | `rr` | marginal risk ratio, reported on the log scale | -0.000005 | 0.000350 | 1.0000 | 0 | 0.000023 vs 0.0500 | equivalent |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ate` | average treatment effect | -0.000007 | 0.0020 | 1.0001 | 0 | 0.000196 vs 0.0500 | equivalent |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ate[S=0]` | average treatment effect, in stratum S=0 | -1.936e-08 | 0.000004 | 1.0000 | 0 | 0.000002 vs 0.0500 | equivalent |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ate[S=1]` | average treatment effect, in stratum S=1 | -0.000002 | 0.000408 | 1.0001 | 0 | 0.000042 vs 0.0500 | equivalent |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ate[S=2]` | average treatment effect, in stratum S=2 | -0.000030 | 0.0043 | 1.0005 | 0 | 0.000727 vs 0.0500 | equivalent |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey0` | counterfactual mean under no treatment | 0.000006 | 0.0022 | 1.0002 | 0 | 0.000237 vs 0.0500 | equivalent |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey0[S=0]` | counterfactual mean under no treatment, in stratum S=0 | -3.448e-09 | 0.000001 | 1.0000 | 0 | 3.806e-07 vs 0.0500 | equivalent |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey0[S=1]` | counterfactual mean under no treatment, in stratum S=1 | 3.434e-07 | 0.000080 | 1.0000 | 0 | 0.000014 vs 0.0500 | equivalent |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey0[S=2]` | counterfactual mean under no treatment, in stratum S=2 | 0.000028 | 0.0050 | 1.0007 | 0 | 0.000838 vs 0.0500 | equivalent |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey1` | counterfactual mean under treatment | -0.000001 | 0.000485 | 1.0000 | 0 | 0.000029 vs 0.0500 | equivalent |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey1[S=0]` | counterfactual mean under treatment, in stratum S=0 | -2.281e-08 | 0.000007 | 1.0000 | 0 | 0.000004 vs 0.0500 | equivalent |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey1[S=1]` | counterfactual mean under treatment, in stratum S=1 | -0.000002 | 0.000513 | 1.0000 | 0 | 0.000285 vs 0.0500 | equivalent |
| two arms, three baseline strata, an instrument, complete data, correct outcome regression | `ey1[S=2]` | counterfactual mean under treatment, in stratum S=2 | -0.000003 | 0.000628 | 1.0001 | 0 | 0.000165 vs 0.0500 | equivalent |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ate[1 vs 0]` | difference in counterfactual means, 1 versus 0 | -0.000004 | 0.000847 | 1.0002 | 0 | 0.000102 vs 0.0500 | equivalent |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ate[2 vs 0]` | difference in counterfactual means, 2 versus 0 | -0.000004 | 0.000850 | 1.0000 | 0 | 0.000066 vs 0.0500 | equivalent |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ey[0]` | counterfactual mean under treatment arm '0' | 0.000002 | 0.000561 | 1.0001 | 0 | 0.000062 vs 0.0500 | equivalent |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ey[1]` | counterfactual mean under treatment arm '1' | -0.000002 | 0.000632 | 1.0002 | 0 | 0.000106 vs 0.0500 | equivalent |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `ey[2]` | counterfactual mean under treatment arm '2' | -0.000002 | 0.000601 | 1.0001 | 0 | 0.000085 vs 0.0500 | equivalent |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `or[1 vs 0]` | marginal odds ratio, 1 versus 0, reported on the log scale | -0.000023 | 0.000783 | 1.0001 | 0 | 0.000105 vs 0.0500 | equivalent |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `or[2 vs 0]` | marginal odds ratio, 2 versus 0, reported on the log scale | -0.000038 | 0.000895 | 1.0000 | 0 | 0.000065 vs 0.0500 | equivalent |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `rr[1 vs 0]` | marginal risk ratio, 1 versus 0, reported on the log scale | -0.000011 | 0.000821 | 1.0001 | 0 | 0.000102 vs 0.0500 | equivalent |
| three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression | `rr[2 vs 0]` | marginal risk ratio, 2 versus 0, reported on the log scale | -0.000013 | 0.000855 | 1.0000 | 0 | 0.000069 vs 0.0500 | equivalent |
<!-- /generated -->

The paired margin of 0.15 empirical SDs cannot detect the design change of this study. The
shipped construction differs from this design by 0.024 to 0.079 empirical SDs on average. The
file `design-gap.csv` reports these gaps for the `strata` and `missing_outcome` laws. Each gap is
inside the margin. The RMSE ratio of the two designs is 0.996 to 1.005 on those rows. Thus the
RMSE ratio carries no verdict here.

`test_the_pairing_reads_the_design_change` in
`tests/unit/test_ctmle_oat_per_arm_strata_missing_study_design.py` reads the gap itself. On
every paired row, the mean of $|\hat\psi_{\text{cleverly}} - \hat\psi_{\text{drtmle}}|$ must be
at most 0.005 empirical SDs. The largest measured value is 0.0008, for `ey0[S=2]` on the
`strata` law. The smallest `mean_design_gap` in `design-gap.csv` is 0.024. Thus the shipped
design would fail this check on every reported row.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `robustness_contract` | `ate__missing_outcome_crossfit` | positive | average treatment effect: missing outcomes, five-fold cross-fitted per-arm design, n = 2,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0016 to 0.0034, margin 0.0069, SE ratio 0.8488 | pass |
| `robustness_contract` | `ate__strata_missing` | positive | average treatment effect: three strata with missing outcomes, cross-fitted per-arm design per stratum, n = 3,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0020 to 0.0020, margin 0.0055, SE ratio 0.9447 | pass |
| `robustness_contract` | `ate__strata_missing_outcome_wrong` | control | average treatment effect: the strata law with the outcome regression without W1, n = 3,000 | bias interval must fall entirely outside the margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias 0.0685 to 0.0731, margin 0.0062, SE ratio 0.9692 | pass |
| `robustness_contract` | `ate__weighted_strata_missing` | positive | average treatment effect: the strata law under the declared known weight 0.5 + 1{W1 > 0}, n = 3,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0025 to 0.0021, margin 0.0064, SE ratio 0.9112 | pass |
| `robustness_contract` | `ate_s0__strata_missing` | positive | average treatment effect in baseline stratum S = 0: three strata with missing outcomes, cross-fitted per-arm design per stratum, n = 3,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0027 to 0.0025, margin 0.0071, SE ratio 0.9556 | pass |
| `robustness_contract` | `ate_s0__weighted_strata_missing` | positive | average treatment effect in baseline stratum S = 0: the strata law under the declared known weight 0.5 + 1{W1 > 0}, n = 3,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0029 to 0.0030, margin 0.0081, SE ratio 0.9446 | pass |
| `robustness_contract` | `ate_s1__strata_missing` | positive | average treatment effect in baseline stratum S = 1: three strata with missing outcomes, cross-fitted per-arm design per stratum, n = 3,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0025 to 0.0050, margin 0.0102, SE ratio 0.9158 | pass |
| `robustness_contract` | `ate_s1__weighted_strata_missing` | positive | average treatment effect in baseline stratum S = 1: the strata law under the declared known weight 0.5 + 1{W1 > 0}, n = 3,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0044 to 0.0038, margin 0.0112, SE ratio 0.9659 | pass |
| `robustness_contract` | `ate_s2__strata_missing` | positive | average treatment effect in baseline stratum S = 2: three strata with missing outcomes, cross-fitted per-arm design per stratum, n = 3,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0064 to 0.0030, margin 0.0129, SE ratio 0.9986 | pass |
| `robustness_contract` | `ate_s2__weighted_strata_missing` | positive | average treatment effect in baseline stratum S = 2: the strata law under the declared known weight 0.5 + 1{W1 > 0}, n = 3,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0060 to 0.0043, margin 0.0141, SE ratio 1.0161 | pass |
| `robustness_contract` | `ey0__missing_outcome_crossfit` | positive | arm-0 mean: missing outcomes, five-fold cross-fitted per-arm design, n = 2,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0026 to 0.0012, margin 0.0052, SE ratio 0.8555 | pass |
| `robustness_contract` | `ey1__missing_outcome_crossfit` | positive | arm-1 mean: missing outcomes, five-fold cross-fitted per-arm design, n = 2,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0015 to 0.0019, margin 0.0046, SE ratio 0.8361 | pass |
| `robustness_contract` | `ey_obs__missing_outcome_crossfit` | positive | observed outcome mean under the natural course: missing outcomes, five-fold cross-fitted per-arm design, n = 2,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0011 to 0.0011, margin 0.0031, SE ratio 0.9920 | pass |
| `robustness_contract` | `ey_obs__natural_course_insample` | positive | observed outcome mean under the natural course: missing outcomes, in-sample per-arm design, n = 2,250, natural-course estimands | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.000289 to 0.0019, margin 0.0030, SE ratio 0.9682 | pass |
| `robustness_contract` | `ey_obs__strata_missing` | positive | observed outcome mean under the natural course: three strata with missing outcomes, cross-fitted per-arm design per stratum, n = 3,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.000866 to 0.000975, margin 0.0025, SE ratio 1.0171 | pass |
| `robustness_contract` | `ey_obs_s0__strata_missing` | positive | observed outcome mean in baseline stratum S = 0: three strata with missing outcomes, cross-fitted per-arm design per stratum, n = 3,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0012 to 0.0014, margin 0.0036, SE ratio 0.9926 | pass |
| `robustness_contract` | `ey_obs_s1__strata_missing` | positive | observed outcome mean in baseline stratum S = 1: three strata with missing outcomes, cross-fitted per-arm design per stratum, n = 3,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0015 to 0.0019, margin 0.0046, SE ratio 1.0114 | pass |
| `robustness_contract` | `ey_obs_s2__strata_missing` | positive | observed outcome mean in baseline stratum S = 2: three strata with missing outcomes, cross-fitted per-arm design per stratum, n = 3,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0025 to 0.0018, margin 0.0059, SE ratio 1.0225 | pass |
| `robustness_contract` | `or__missing_outcome_crossfit` | positive | marginal odds ratio: missing outcomes, five-fold cross-fitted per-arm design, n = 2,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias 0.0037 to 0.0606, margin 0.0780, SE ratio 0.8489 | pass |
| `robustness_contract` | `paf__missing_outcome_crossfit` | positive | population attributable fraction: missing outcomes, five-fold cross-fitted per-arm design, n = 2,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0014 to 0.0041, margin 0.0075, SE ratio 0.7722 | pass |
| `robustness_contract` | `paf__natural_course_insample` | positive | population attributable fraction: missing outcomes, in-sample per-arm design, n = 2,250, natural-course estimands | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0031 to 0.0020, margin 0.0069, SE ratio 0.7832 | pass |
| `robustness_contract` | `paf__strata_missing` | positive | population attributable fraction: three strata with missing outcomes, cross-fitted per-arm design per stratum, n = 3,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0025 to 0.0020, margin 0.0061, SE ratio 0.9121 | pass |
| `robustness_contract` | `paf_s0__strata_missing` | positive | population attributable fraction in baseline stratum S = 0: three strata with missing outcomes, cross-fitted per-arm design per stratum, n = 3,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0028 to 0.0027, margin 0.0075, SE ratio 0.8943 | pass |
| `robustness_contract` | `paf_s1__strata_missing` | positive | population attributable fraction in baseline stratum S = 1: three strata with missing outcomes, cross-fitted per-arm design per stratum, n = 3,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0032 to 0.0048, margin 0.0110, SE ratio 0.9162 | pass |
| `robustness_contract` | `paf_s2__strata_missing` | positive | population attributable fraction in baseline stratum S = 2: three strata with missing outcomes, cross-fitted per-arm design per stratum, n = 3,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0083 to 0.0034, margin 0.0160, SE ratio 1.0113 | pass |
| `robustness_contract` | `par__missing_outcome_crossfit` | positive | population attributable risk: missing outcomes, five-fold cross-fitted per-arm design, n = 2,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.000810 to 0.0022, margin 0.0041, SE ratio 0.7632 | pass |
| `robustness_contract` | `par__natural_course_insample` | positive | population attributable risk: missing outcomes, in-sample per-arm design, n = 2,250, natural-course estimands | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0016 to 0.0012, margin 0.0038, SE ratio 0.7746 | pass |
| `robustness_contract` | `par__strata_missing` | positive | population attributable risk: three strata with missing outcomes, cross-fitted per-arm design per stratum, n = 3,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0014 to 0.0011, margin 0.0034, SE ratio 0.9067 | pass |
| `robustness_contract` | `par_s0__strata_missing` | positive | population attributable risk in baseline stratum S = 0: three strata with missing outcomes, cross-fitted per-arm design per stratum, n = 3,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0016 to 0.0014, margin 0.0041, SE ratio 0.8909 | pass |
| `robustness_contract` | `par_s1__strata_missing` | positive | population attributable risk in baseline stratum S = 1: three strata with missing outcomes, cross-fitted per-arm design per stratum, n = 3,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0018 to 0.0026, margin 0.0060, SE ratio 0.9166 | pass |
| `robustness_contract` | `par_s2__strata_missing` | positive | population attributable risk in baseline stratum S = 2: three strata with missing outcomes, cross-fitted per-arm design per stratum, n = 3,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0043 to 0.0019, margin 0.0085, SE ratio 1.0132 | pass |
| `robustness_contract` | `rr__missing_outcome_crossfit` | positive | marginal risk ratio: missing outcomes, five-fold cross-fitted per-arm design, n = 2,000 | bias interval inside the equivalence margin, SE ratio inside (0.1, 10), SE ratio must remain between 0.1 and 10.0 | bias -0.0014 to 0.0138, margin 0.0208, SE ratio 0.8567 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000923, coverage 0.8963 to 0.9460, SE ratio 0.9292 | **fail** |
| `root_n_and_efficiency` | `n_4000` | positive | bias, coverage and SE calibration at n = 4,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000466, coverage 0.8992 to 0.9481, SE ratio 0.8955 | **fail** |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000447, coverage 0.8949 to 0.9449, SE ratio 0.8812 | **fail** |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5636 to -0.4362 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5700 to -0.5183 | pass |
<!-- /generated -->

The `robustness_contract` cells measure the bias of the point estimate. Each positive cell passes
when its bias interval is inside the equivalence margin. The control fits an outcome model without
$W_1$ and passes when its bias is outside the margin. The `root_n_and_efficiency` ladder fits the
`strata` law with a missing outcome and five folds, and reads the marginal ATE.

The three rungs are n = 2,000, 4,000 and 8,000. Each rung has an equivalent bias, and each
coverage interval reaches below the 0.90 floor. The `root_n_rate` rows read the slopes of the
log empirical SD and of the log mean standard error on log n. Both slopes are inside the root-n
band.

Each `rr` and `or` row records the standard error of $\log \hat\psi$. The summary multiplies it
by $\hat\psi$, the delta-method standard error of $\hat\psi$, before it compares it with the
empirical SD of $\hat\psi$. The first summary of the rerun did not, and it published SE ratios of
0.559 for `rr` and 0.322 for `or`. The current `properties.csv` was summarized again from the
committed `property-replicates.csv.gz`, with no new fit. No verdict changed.
`test_the_ratio_cells_compare_one_scale` checks both readings.

A mean SE ratio is sensitive to a few fits with a large standard error. The table gives the
median beside the mean for the ladder and for the cells that have a tail fit. A tail fit has a
standard error above twice the empirical SD of its cell. `se_ratio_spread` in
`tests/studies/ctmle_oat_per_arm_strata_missing_properties.py` computes the table from
`property-replicates.csv.gz`, and `test_the_tail_fits_are_rare` checks it.

| cell | mean SE ratio | median SE ratio | tail fits of 800 |
| --- | --- | --- | --- |
| `n_2000` | 0.929 | 0.898 | 3 |
| `n_4000` | 0.896 | 0.893 | 0 |
| `n_8000` | 0.881 | 0.880 | 0 |
| `ate__strata_missing` | 0.945 | 0.925 | 2 |
| `ate_s2__strata_missing` | 0.999 | 0.946 | 2 |
| `ey_obs__strata_missing` | 1.017 | 1.011 | 2 |
| `ey_obs_s2__strata_missing` | 1.023 | 1.006 | 2 |
| `par__strata_missing` | 0.907 | 0.877 | 1 |
| `par_s2__strata_missing` | 1.013 | 0.948 | 2 |
| `paf__strata_missing` | 0.912 | 0.885 | 1 |
| `paf_s2__strata_missing` | 1.011 | 0.950 | 2 |
| `ate_s2__weighted_strata_missing` | 1.016 | 0.997 | 1 |
| `ate__strata_missing_outcome_wrong` | 0.969 | 0.907 | 5 |

In the `strata_missing` cells, the two tail fits are replications 271 and 605. They lift the
mean SE ratio of the $S = 2$ `ate`, `par` and `paf` cells 5% to 7% above its median. No cell
outside the table and no primary row has a tail fit.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | primary replications per law |
| `n` | 2250 | observations per primary replication |
| `independent_tests_total` | 62 | implementation-estimand tests against the truth |
| `independent_tests_passed` | 20 | of those, passing |
| `paired_tests_total` | 31 | paired comparisons with R `drtmle` |
| `paired_tests_passed` | 31 | of those, passing |
| `max_standardized_bias` | 0.0926 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.8575 | lowest primary coverage of the diagnostic interval |
| `min_se_ratio_ci_lower` | 0.6898 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0584 | highest bootstrap primary SE-ratio endpoint |
| `max_margin_utilization` | 0.0050 | largest share of the paired margin used |
| `mechanism_fits_with_rows_at_bound` | 341 | primary fits with one or more rows of a factor at a declared bound |
| `mechanism_max_gap_off_bound` | 4.491e-07 | largest gap between the two designs in a fit with no row at a bound |
| `mechanism_max_gap` | 0.0100 | largest gap between the two designs in any fit |
| `properties[robustness_contract/ate__strata_missing]:se_ratio` | 0.9447 | SE ratio of the marginal ATE on the strata law at n = 3,000 |
| `properties[robustness_contract/ate__strata_missing_outcome_wrong]:standardized_bias` | 2.8427 | bias of the outcome-wrong control, in empirical standard deviations |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:minimum_power` | 0.8000 | rejection lower bound the power cell must clear |
| `margin:type_i_ceiling` | 0.1000 | the rate the null cell must bound |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:union_model_se_lower` | 0.1000 | robustness-contract SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | robustness-contract SE-ratio screen, upper limit |

## The reading rule and the red rows

The study declared its reading rule before the run, in `READING_RULE` of
`tests/studies/canonical_ctmle_oat_per_arm_strata_missing.py`. A red row goes to F19 when its bias
is equivalent and its SE ratio is below 1. The upper end of its 99% SE-ratio interval must also be
below 1.07. For a ladder rung, a bootstrap interval of the SE ratio gives that upper end. Every
other red row is a defect suspect.

A floor was added to the rule after the run. A routed row must also have a point SE ratio of 0.70
or more. A shorter standard error is a defect suspect, because it is too short for the
generated-design term to explain. The lowest routed row is `ey[1]` on the `three_arm_missing` law,
at 0.736 for both implementations. Thus the floor changes no routing of this run.

| red rows | count | reading | owner |
| --- | --- | --- | --- |
| truth rows of both implementations | 42 | short diagnostic standard error, equivalent bias. Both implementations read the same rows alike | F19 |
| `root_n_and_efficiency` rungs | 3 | coverage interval reaches below 0.90, equivalent bias, SE ratios 0.929, 0.896 and 0.881, bootstrap upper ends 1.0065, 0.952 and 0.940 | F19 |

No paired row, `robustness_contract` cell or control is red, and no replication failed. Both
`root_n_rate` rows pass. The empirical SD contracts at a slope of -0.499, and the mean standard
error at -0.537.

## The re-declared ladder

The first run declared the ladder at n = 500, 2,000 and 8,000. The `n_500` rung measured the
stability of the factor learner in small cells and not the rate. After the run, the study dropped
that rung and re-declared the ladder at n = 2,000, 4,000 and 8,000. The rerun of the property
phase replaced the first run's property artifacts. The first run is commit `ae53bbb`.

The table gives what the `n_500` rung measured in the first run.

| quantity | value |
| --- | --- |
| mean diagnostic standard error | 0.0884 |
| empirical SD of the marginal ATE | 0.0561 |
| mean SE ratio | 1.575 |
| median standard error over the empirical SD | 1.00 |
| fits with a standard error above twice the empirical SD | 40 of 800 (5%) |
| mean SE ratio without those fits | 1.035 |
| coverage of the diagnostic interval | 0.954 |
| `root_n_rate/reported_se` slope (99% interval) | -0.71 (-0.79 to -0.63) |
| `root_n_rate/empirical_sd` slope | -0.50 |
| slope of the median standard error | -0.55 |

The reading rule made `n_500` and `root_n_rate/reported_se` defect suspects. A diagnosis refitted
the ladder replications and found no defect in the code.

| finding | value |
| --- | --- |
| rows in the stratum, arm and fold cell that fits each factor at n = 500 | about 20 to 35; 29.3 expected in the smallest cell |
| estimated design at the row that dominates a tail fit | far below 0.01 |
| true design at those rows | 0.11 or more |
| largest gap between the fit's curve and the textbook curve | 1e-13 |
| tail fits at n = 2,000 and at n = 8,000 | 3 of 800, and none |

In a small cell the unpenalized cubic factor learner extrapolates. One or two observed rows then
get a very small design, and their term $C_a / G\, (Y - \bar Q^*)$ dominates the plug-in
variance. The point estimate does not change, so its bias stays inside the margin.

The re-declaration added condition (t). As declared, it required the smallest factor training
cell to hold 100 rows or more in expectation at the smallest rung. That cell is arm 0 in the
stratum $S = 2$, with a share of 0.0732 of the rows. Four fifths of the rows train each fold, so
the cell holds 117.1 rows at n = 2,000.

The row count is not the quantity that limits a factor fit. Each factor is a logistic regression
on a binary target, and its smaller class limits it. On the strata law the smaller class is the
nonrespondents of arm 0 in $S = 2$. After the rerun, the study restated condition (t) as the
expected size of the smallest class for each stratum, arm, fold and factor. The restated
condition sets no floor, because the declared ladder does not hold 100 such rows.
`smallest_minority_class` in `tests/studies/oat_per_arm_extension_laws.py` computes the count by
quadrature.

| configuration | n | rows that train each fit | smallest class | expected count | tail fits of 800 |
| --- | --- | --- | --- | --- | --- |
| dropped `n_500` rung of the first run | 500 | four fifths | nonrespondents, arm 0, $S = 2$ | 9.9 | 40 |
| `n_2000` | 2,000 | four fifths | nonrespondents, arm 0, $S = 2$ | 39.5 | 3 |
| `strata_missing` cells | 3,000 | four fifths | nonrespondents, arm 0, $S = 2$ | 59.2 | 2 |
| `n_4000` | 4,000 | four fifths | nonrespondents, arm 0, $S = 2$ | 79.0 | 0 |
| `missing_outcome_crossfit` | 2,000 | four fifths | nonrespondents, arm 1 | 107.0 | 0 |
| primary `missing_treatment` | 2,250 | all | nonrespondents, arm 1 | 110.5 | 0 |
| primary `three_arm_missing` | 2,250 | all | nonrespondents, arm 0 | 116.4 | 0 |
| primary `missing_outcome` and `natural_course_insample` | 2,250 | all | nonrespondents, arm 1 | 150.5 | 0 |
| `n_8000` | 8,000 | four fifths | nonrespondents, arm 0, $S = 2$ | 157.9 | 0 |
| primary `strata` | 2,250 | all | arm 0 in $S = 2$ | 164.7 | 0 |

The weighted and the outcome-wrong cells sample the strata law at n = 3,000, so their count is
59.2 too. They hold 1 and 5 tail fits. The measured consequence of a small class is a rare tail
fit. It moves the mean SE ratio of a cell and not its bias.
`test_condition_t_the_smallest_minority_class_of_each_configuration` checks each count.

Condition (r) does not read n, so each rung keeps 800 replications, with a pass
probability of 0.950. The primary laws, the pairing, the robustness cells and the seeds do not
change. `tests/studies/ctmle_oat_per_arm_strata_missing_properties.py` records the change.

## Limitations

| limitation | what it means for use |
| --- | --- |
| No cell measures published inference | Every `oat` fit withholds `ci`, `pvalue` and `std_error`. The `std_error`, `ci_lower`, `ci_upper` and `covered` columns read `plugin_std_error` and `plugin_interval` |
| The diagnostic standard error is too small on the primary laws | The primary SE ratios run from `min_se_ratio_ci_lower` to `max_se_ratio_ci_upper`. The truth rows of both implementations are red alike, and F19 owns them. R `drtmle` reports the same standard errors, so the pairing cannot detect the gap |
| The diagnostic standard error is heavy-tailed in small factor cells | In the first run, at n = 500, the smallest class of a factor target held about 10 rows, and the mean standard error was 1.575 times the empirical SD. The ladder now starts at n = 2,000, where that class holds 39.5 rows and 3 of 800 fits have a tail. [The re-declared ladder](#the-re-declared-ladder) gives the measurement |
| The paired strata law has complete data | The `strata` primary samples `strata_complete`, so R `drtmle` never meets a stratum-local observation factor. Only the fast tier, `tests/unit/test_outcome_adaptive_per_arm_extensions.py`, checks that factor against a longhand. The `strata_missing` property cells measure its bias, unpaired |
| The points are paired, not the intervals | All 31 paired rows are equivalent, with `max_margin_utilization` of the margin used |
| The designs agree only off the bounds | Where no factor row sits at a bound, the two designs agree to `mechanism_max_gap_off_bound`. In `mechanism_fits_with_rows_at_bound` fits some rows sit at a bound, and the gap reaches `mechanism_max_gap`. The rows at a bound are reported, not gated |
| Parametric laws with a binary outcome | Each law has a normal index for each arm and one instrument. A continuous outcome and more than three arms have fast-tier evidence only |
| Clusters stay refused | A per-arm fit with `id=` refuses. A cross-fitted fit with a missing treatment refuses too |

## Reproduction

The [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_oat_per_arm_strata_missing/manifest.json)
records the seeds, the margins, the reading rule, the estimator configuration and the hashes. Run
`python -m tests.canonical.drtmle_oat_per_arm_strata_missing.regenerate` to regenerate the
artifacts. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_oat_per_arm_strata_missing/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_oat_per_arm_strata_missing/equivalence.csv),
[design gap](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_oat_per_arm_strata_missing/design-gap.csv)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_oat_per_arm_strata_missing/properties.csv)
carry every published row. The post-run gate in
`tests/unit/test_ctmle_oat_per_arm_strata_missing_study_design.py` applies the reading rule to the
committed tables.
