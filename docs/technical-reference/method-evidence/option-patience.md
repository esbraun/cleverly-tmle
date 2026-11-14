# Patience on the cross-validated C-TMLE path

This study validates the point estimate of `CTMLE(patience=p)`, which stops the cross-validated
scoring of the candidate path after `p` candidates whose risk is not a new minimum and selects
the smallest risk among the scored candidates. The
[patience section](../collaborative-tmle.md#patience) states the option. A selector path
withholds its interval (F18), so the study reads point estimates, stop shares and paired
displacements.

What an early stop can skip depends on how the path is ordered. On the greedy path each step adds
the candidate that lowers the training risk most, so a candidate that removes bias on its own
enters before the counter can stop. On a pre-ordered path (`"ordered"`, `"discrete"`, `"lasso"`,
`"learners"`) the order is fixed before any risk is read, and the counter can stop before a needed
candidate with a probability that does not shrink with $n$. The study measures both.

R `ctmle`'s `ctmleDiscrete(patience=)` counter is the implementation reference, transcribed in
`tests/unit/test_patience.py`, so no canonical implementation is compared. The study reads every
cell against its exact truth.

## What was fitted

| law | construction | fits |
| --- | --- | --- |
| LC1 | the `canonical-ctmle-selector` binary law with three covariates and that study's settings | greedy with patience 1, the smallest patience whose pilot stop share reached 0.5, and ordered and discrete with patience 1, each beside its whole path |
| LC2 | the `canonical-ctmle-sequences` lasso and learners laws with that study's grid and candidates | lasso and learners with patience 2, each beside its whole path |
| LC3 | $W_1, Z_1, Z_2, Z_3 \sim N(0, 1)$, $A \sim \operatorname{Bernoulli}(\operatorname{expit}(0.8(Z_1 + Z_2 + Z_3) + 0.6 W_1))$, $Y \sim \operatorname{Bernoulli}(\operatorname{expit}(-0.3 + 0.5 A + W_1))$. The outcome regression is a logistic regression on $(A, Z_1, Z_2, Z_3)$, which omits the confounder $W_1$ | ordered with the instruments first and patience 2, and greedy with patience 2, each beside its whole path; `g_bounds=(0.01, 0.99)` |

Every fit is in sample at n = 2,000 with five selection folds and two inner folds, no penalty,
`max_iter=100`, `tol=1e-10`, and the ATE as the selection estimand. The laws and fits are in
`tests/studies/_option_patience_laws.py`, and `tests/unit/test_option_patience_design.py`
recomputes every declared number. `tests/unit/test_option_patience_witness.py` checks each
behaviour on fixed samples with a mutation control.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| law LC1 (the canonical-ctmle-selector binary law, three covariates), the ATE, n = 2,000: the greedy path with patience=1, in sample; the point estimate only | `ate` | average treatment effect | `cleverly` | -0.0011 to 0.0015 | 0.9519 | 0.9899 | pass |
<!-- /generated -->

The primary row is point-only, so it passes on its bias alone. Its coverage and SE ratio come
from the plug-in diagnostic of the working mechanism, which is not an interval the package
reports.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `greedy_late_confounder` | `lc3_greedy` | positive | law LC3 (confounder W1 omitted by the outcome regression, three instruments), the ATE, n = 2,000: the greedy path with patience=2 minus the greedy whole path on the same draw | the 99% interval of the mean paired difference inside 0.25 SD of the whole-path estimates | mean difference 0.000272 to 0.000993, 0.0223 whole-path SD, identical on 0.9825 | pass |
| `late_confounder_stop` | `lc3_ordered` | positive | law LC3 (confounder W1 omitted by the outcome regression, three instruments), the ATE, n = 2,000: the ordered path with patience=2, in sample, the instruments ordered first: the share of draws whose selected treatment model leaves out W1 | the lower limit of the share's exact 99% interval at least 0.3 | share 0.6773 to 0.7364, bias on those draws 3.3159 whole-path SD | pass |
| `ordered_late_confounder` | `lc3_ordered` | control | law LC3 (confounder W1 omitted by the outcome regression, three instruments), the ATE, n = 2,000: the ordered path with patience=2, in sample, the instruments ordered first | the 99% interval of the standardized bias must lie wholly beyond the margin | bias 0.0758 to 0.0834, margin 0.0147 | pass |
| `ordered_whole_path` | `lc3_ordered` | positive | law LC3 (confounder W1 omitted by the outcome regression, three instruments), the ATE, n = 2,000: the ordered path scoring the whole path, in sample, on the draws of the ordered fit with patience | bias interval inside the equivalence margin | bias -0.0015 to 0.0029, margin 0.0086 | pass |
| `patience_bias` | `lc1_greedy` | positive | law LC1 (the canonical-ctmle-selector binary law, three covariates), the ATE, n = 2,000: the greedy path with patience=1, in sample | bias interval inside the equivalence margin | bias -0.0013 to 0.0014, margin 0.0051 | pass |
| `patience_witness` | `lc1_greedy` | positive | law LC1 (the canonical-ctmle-selector binary law, three covariates), the ATE, n = 2,000: the greedy path with patience=1, in sample: the share of draws the counter stopped before the path's end | the lower limit of the share's exact 99% interval at least 0.3 | share 0.8210 to 0.8681 | pass |
| `reported_displacement` | `lc1_discrete` | diagnostic | law LC1 (the canonical-ctmle-selector binary law, three covariates), the ATE, n = 2,000: the discrete path with patience minus its whole path on the same draw | none. The row is a diagnostic and states no verdict | mean difference -0.000038 to 0.000065, 0.000657 whole-path SD, identical on 0.7731 | reported |
| `reported_displacement` | `lc1_greedy` | diagnostic | law LC1 (the canonical-ctmle-selector binary law, three covariates), the ATE, n = 2,000: the greedy path with patience minus its whole path on the same draw | none. The row is a diagnostic and states no verdict | mean difference -0.000048 to 0.000054, 0.000141 whole-path SD, identical on 0.7775 | reported |
| `reported_displacement` | `lc1_ordered` | diagnostic | law LC1 (the canonical-ctmle-selector binary law, three covariates), the ATE, n = 2,000: the ordered path with patience minus its whole path on the same draw | none. The row is a diagnostic and states no verdict | mean difference -0.000040 to 0.000065, 0.000590 whole-path SD, identical on 0.7856 | reported |
| `reported_displacement` | `lc2_lasso` | diagnostic | law LC2 (the canonical-ctmle-sequences lasso law), the ATE, n = 2,000: the lasso path with patience minus its whole path on the same draw | none. The row is a diagnostic and states no verdict | mean difference 0.0262 to 0.0293, 0.7538 whole-path SD, identical on 0.0731 | reported |
| `reported_displacement` | `lc2_learners` | diagnostic | law LC2 (the canonical-ctmle-sequences learners law), the ATE, n = 2,000: the learners path with patience minus its whole path on the same draw | none. The row is a diagnostic and states no verdict | mean difference 0 to 0, 0 whole-path SD, identical on 1 | reported |
| `reported_estimate` | `lc1_discrete_patience` | diagnostic | law LC1 (the canonical-ctmle-selector binary law, three covariates), the ATE, n = 2,000: the discrete path with patience=1, in sample | none. The row is a diagnostic and states no verdict | bias -0.0013 to 0.0014, coverage 0.9444, SE ratio 0.9831 | reported |
| `reported_estimate` | `lc1_discrete_whole` | diagnostic | law LC1 (the canonical-ctmle-selector binary law, three covariates), the ATE, n = 2,000: the discrete path scoring the whole path, in sample | none. The row is a diagnostic and states no verdict | bias -0.0013 to 0.0014, coverage 0.9513, SE ratio 0.9944 | reported |
| `reported_estimate` | `lc1_greedy_whole` | diagnostic | law LC1 (the canonical-ctmle-selector binary law, three covariates), the ATE, n = 2,000: the greedy path scoring the whole path, in sample | none. The row is a diagnostic and states no verdict | bias -0.0013 to 0.0014, coverage 0.9494, SE ratio 0.9895 | reported |
| `reported_estimate` | `lc1_ordered_patience` | diagnostic | law LC1 (the canonical-ctmle-selector binary law, three covariates), the ATE, n = 2,000: the ordered path with patience=1, in sample | none. The row is a diagnostic and states no verdict | bias -0.0013 to 0.0014, coverage 0.9450, SE ratio 0.9835 | reported |
| `reported_estimate` | `lc1_ordered_whole` | diagnostic | law LC1 (the canonical-ctmle-selector binary law, three covariates), the ATE, n = 2,000: the ordered path scoring the whole path, in sample | none. The row is a diagnostic and states no verdict | bias -0.0013 to 0.0014, coverage 0.9500, SE ratio 0.9943 | reported |
| `reported_estimate` | `lc2_lasso_patience` | diagnostic | law LC2 (the canonical-ctmle-sequences lasso law), the ATE, n = 2,000: the lasso path with patience=2, in sample | none. The row is a diagnostic and states no verdict | bias 0.0245 to 0.0285, coverage 0.6644, SE ratio 0.7011 | reported |
| `reported_estimate` | `lc2_lasso_whole` | diagnostic | law LC2 (the canonical-ctmle-sequences lasso law), the ATE, n = 2,000: the lasso path scoring the whole path, in sample | none. The row is a diagnostic and states no verdict | bias -0.0036 to 0.0011, coverage 0.9069, SE ratio 0.8871 | reported |
| `reported_estimate` | `lc2_learners_patience` | diagnostic | law LC2 (the canonical-ctmle-sequences learners law), the ATE, n = 2,000: the learners path with patience=2, in sample | none. The row is a diagnostic and states no verdict | bias -0.0027 to 0.0016, coverage 0.9350, SE ratio 0.9816 | reported |
| `reported_estimate` | `lc2_learners_whole` | diagnostic | law LC2 (the canonical-ctmle-sequences learners law), the ATE, n = 2,000: the learners path scoring the whole path, in sample | none. The row is a diagnostic and states no verdict | bias -0.0027 to 0.0016, coverage 0.9350, SE ratio 0.9816 | reported |
| `reported_estimate` | `lc3_greedy_patience` | diagnostic | law LC3 (confounder W1 omitted by the outcome regression, three instruments), the ATE, n = 2,000: the greedy path with patience=2, in sample | none. The row is a diagnostic and states no verdict | bias 0.0106 to 0.0143, coverage 0.8844, SE ratio 0.8851 | reported |
| `reported_estimate` | `lc3_greedy_whole` | diagnostic | law LC3 (confounder W1 omitted by the outcome regression, three instruments), the ATE, n = 2,000: the greedy path scoring the whole path, in sample | none. The row is a diagnostic and states no verdict | bias 0.0100 to 0.0137, coverage 0.8856, SE ratio 0.8794 | reported |
| `reported_stop` | `lc1_discrete` | diagnostic | law LC1 (the canonical-ctmle-selector binary law, three covariates), the ATE, n = 2,000: the discrete path with patience, the share of draws the counter stopped before the path's end | none. The row is a diagnostic and states no verdict | share 0.7881 to 0.8388 | reported |
| `reported_stop` | `lc1_ordered` | diagnostic | law LC1 (the canonical-ctmle-selector binary law, three covariates), the ATE, n = 2,000: the ordered path with patience, the share of draws the counter stopped before the path's end | none. The row is a diagnostic and states no verdict | share 0.7822 to 0.8334 | reported |
| `reported_stop` | `lc2_lasso` | diagnostic | law LC2 (the canonical-ctmle-sequences lasso law), the ATE, n = 2,000: the lasso path with patience, the share of draws the counter stopped before the path's end | none. The row is a diagnostic and states no verdict | share 0.9085 to 0.9427 | reported |
| `reported_stop` | `lc2_learners` | diagnostic | law LC2 (the canonical-ctmle-sequences learners law), the ATE, n = 2,000: the learners path with patience, the share of draws the counter stopped before the path's end | none. The row is a diagnostic and states no verdict | share 0.000211 to 0.0068 | reported |
| `reported_stop` | `lc3_greedy` | diagnostic | law LC3 (confounder W1 omitted by the outcome regression, three instruments), the ATE, n = 2,000: the greedy path with patience, the share of draws the counter stopped before the path's end | none. The row is a diagnostic and states no verdict | share 0.2136 to 0.2692 | reported |
<!-- /generated -->

On LC1 the counter stops the greedy path early on 85% of draws, and the estimate stays unbiased:
its standardized bias is 0.002. That pair is the claim of `C12.greedy`. On LC3 the greedy fit with
patience returns the whole path's estimate on 98% of draws, and the mean paired difference is 0.02
whole-path SDs.

The LC3 ordered cells show why a pre-ordered path stays experimental. With patience 2, 71% of the
ordered fits stop before $W_1$, those fits miss by 3.32 whole-path SDs, and the cell's
standardized bias is 1.35, beyond the margin as its control role requires. The whole ordered path
on the same draws is unbiased (0.020).

## Reported readings

These rows carry no verdict.

**The greedy path on LC3.** The greedy whole path keeps a standardized bias of 0.42, and 0.44
with patience 2. The path selects $W_1$ with a varying subset of the instruments, a logistic model
of $A$ on such a subset is misspecified, and the outcome regression omits $W_1$, so neither
selected nuisance is consistent. The bias belongs to the greedy estimator on this law, not to
`patience=`, and `C01.greedy` and `C12.greedy` both state it as a limit.

**The other pre-ordered paths.** On LC1 the ordered and discrete fits with patience 1 stop early
on 81% of draws and stay unbiased, because the needed covariate comes early in their order. On
LC2 the lasso fit with patience 2 stops early on 93% of draws, and its standardized bias is 0.85
against $-0.03$ for its whole path. The learners fit with patience 2 almost never stops (0.2% of
draws), so it equals its whole path.

## Sizing

Every cell and the primary phase run 1,600 replicates. The bias cells' budget is above the 1,000
a primary needs, and each witness share is gated at an exact lower limit of at least 0.3. The
pilot is `tests/studies/option_patience_pilot.py`, on a stream no declared cell reads, and the
design test recomputes each rule.

## Red-cell route

The study is registered under `reporting`, and no cell is red. The declared reading is in
`tests/canonical/option_patience/regenerate.py`.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | primary replications |
| `n` | 2000 | observations per replication |
| `independent_tests_passed` | 1 | truth tests passing |
| `independent_tests_total` | 1 | truth tests reported |
| `property_cells_passed` | 6 | gated property cells passing |
| `property_cells_total` | 6 | gated property cells reported |
| `max_standardized_bias` | 0.0092 | largest primary standardized bias |
| `min_coverage` | 0.9519 | lowest primary coverage of the plug-in diagnostic |
| `properties[patience_witness/lc1_greedy]:coverage` | 0.8456 | share of LC1 greedy fits the counter stopped early |
| `properties[patience_bias/lc1_greedy]:standardized_bias` | 0.0022 | standardized bias of the LC1 greedy fit with patience |
| `properties[greedy_late_confounder/lc3_greedy]:paired_over_whole_sd` | 0.0223 | mean paired difference on LC3, in whole-path SDs |
| `properties[greedy_late_confounder/lc3_greedy]:identical_share` | 0.9825 | share of LC3 draws where the greedy estimates are identical |
| `properties[late_confounder_stop/lc3_ordered]:coverage` | 0.7075 | share of LC3 ordered fits with patience that leave out W1 |
| `properties[late_confounder_stop/lc3_ordered]:conditional_bias_over_sd` | 3.3159 | bias of those fits, in whole-path SDs |
| `properties[ordered_late_confounder/lc3_ordered]:standardized_bias` | 1.3513 | standardized bias of the LC3 ordered fit with patience |
| `properties[ordered_whole_path/lc3_ordered]:standardized_bias` | 0.0197 | standardized bias of the LC3 ordered whole path |
| `properties[reported_estimate/lc3_greedy_whole]:standardized_bias` | 0.4165 | standardized bias of the LC3 greedy whole path |
| `properties[reported_estimate/lc3_greedy_patience]:standardized_bias` | 0.4434 | standardized bias of the LC3 greedy fit with patience |
| `properties[reported_stop/lc1_ordered]:coverage` | 0.8087 | stop share of the LC1 ordered fit |
| `properties[reported_stop/lc1_discrete]:coverage` | 0.8144 | stop share of the LC1 discrete fit |
| `properties[reported_stop/lc2_lasso]:coverage` | 0.9269 | stop share of the LC2 lasso fit |
| `properties[reported_stop/lc2_learners]:coverage` | 0.0019 | stop share of the LC2 learners fit |
| `properties[reported_estimate/lc2_lasso_patience]:standardized_bias` | 0.8518 | standardized bias of the LC2 lasso fit with patience |
| `properties[reported_estimate/lc2_lasso_whole]:standardized_bias` | -0.0341 | standardized bias of the LC2 lasso whole path |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor. No cell of this study reads it |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold. No cell of this study reads it |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen. No cell of this study reads it |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen. No cell of this study reads it |
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
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 16 | the same bound, in primary replications |

## Limitations

- **Points only.** A selector path withholds its interval (F18), so no cell reads an interval.
- **The greedy path keeps the greedy estimator's bias.** On LC3, where the outcome regression
  omits a confounder and the selected treatment model is misspecified given the selected set,
  the greedy estimate is biased by 0.42 SD with or without patience.
- **Greedy is a heuristic, not a theorem.** When only two candidates together remove the
  confounding, no single greedy step lowers the risk at order one, and the counter can stop
  before the pair. No law here has that structure.
- **A pre-ordered path stays experimental permanently.** The counter can stop before a needed
  candidate at every sample size, as the LC3 ordered and LC2 lasso cells show.
- **One sample size, in sample.** Every cell is at n = 2,000 without cross-fitting, on binary
  outcomes and family P1.
- **No comparator.** The R `ctmle` counter is transcribed in the fast tier, not run here.

## Reproduction

`tests/canonical/option_patience/regenerate.py` regenerates the study and states the run form.
It runs no reference process, because the study has no comparator. The manifest records the
seeds, the margins, the estimator configuration and the source and result hashes, and the run
log records the commit, the clean tree and the wall time.
