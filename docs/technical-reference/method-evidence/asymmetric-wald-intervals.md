# Asymmetric Wald intervals

This study measures the two asymmetric 95% intervals that `cleverly` reports for a stratum or
sparse-arm TMLE estimate. A bounded difference of two binary-outcome means of one fit takes
correlated MOVER on the two means' range-logit intervals (`cleverly.inference.mover_ci`, with
the correlation `cleverly.inference.implied_correlation` reads from the three variances). An
unclustered continuous-outcome target takes the skew-widened t
(`cleverly.inference.skew_widened_ci`): the hull of the Student t interval at
`max(small_sample_df, 5)` and its Cornish-Fisher correction by the curve's acceleration
(`cleverly.inference.curve_acceleration`). A bounded mean keeps the range-logit normal interval
that [small-sample-wald](small-sample-wald-intervals.md) supports, and no interval is shown at a
constant arm cell or an estimate on its bound.

The run was declared before the default read either construction. The study computed each
interval from the public functions (`tests/studies/asymmetric_wald.py`, `rule_interval`). Every
gated cell passed and no falsifier fired, so the default now reads them
([small-sample scale](../inference.md#small-sample-scale)), and
`tests/unit/test_asymmetric_default.py` checks that the shipped interval of a fit equals the one
the study measured.

**No canonical implementation is compared.** No published implementation applies either
construction to a TMLE influence curve, so no comparison was run. The absence does not block the
row, and a zero-row equivalence artifact records it. The independent evidence is the exact or
closed-form truths, the unit witnesses of `tests/unit/test_asymmetric_intervals.py`, the
controls, the falsifiers and two deliberate-mutation witnesses.

## What was tested

| setting | declaration |
| --- | --- |
| primary | 1,000 fits of the binary-outcome law with three unequal strata of `canonical-stratified-tmle` at n = 2,000, its `ate[V=2]` under MOVER, a withheld interval as the point. The shipped identity interval of the same fits, from before the default read MOVER, is in `primary-shipped.csv`, which no verdict reads |
| held-out cells | five laws declared before any estimator was fitted on them, each with exact or closed-form truths (`tests/studies/asymmetric_wald_laws.py`). D1: a sparse treated arm near 0 in a stratum of about 600 rows, n = 3,000. D2: a sparse untreated arm near 1 with a continuous covariate and no strata, n = 2,000. C1: right-skewed gamma errors scaled by arm, n = 2,500. C2: a left-skewed Beta outcome with a coarse covariate, n = 2,000. C3: symmetric Laplace errors, the no-harm control, n = 2,500. D1 and D2 are new laws in known mechanism classes: a binary arm near its bound, in a stratum (the `rare` diagnosis law) and without strata (study 1's H1) |
| owner cells | the cells refuted in study 1, on fresh seeds: `ho_sparse_treat_ate` (H1, owner `F3-difference-near-bound`), `ho_hurdle_ey1` (H2) and `lognormal06_ey1` (the lognormal law of skewness 2.0), both owner `F3-skewed-continuous`. These laws were read when the constructions were chosen, so they are diagnosis laws here |
| reported rows | the other targets of each law, the Student t and lognormal 6.2 laws, the boundary law at n = 1,000 and 2,000, the rare-outcome difference, an identity-MSM coefficient, and every control re-reading of the gated cells |
| verdicts | each gated cell draws 4,000 fits. On the first 2,000, the lower leg's 99% Clopper-Pearson lower end is at least 0.92, counting lost and withheld fits as misses, and the shown fits' 99% upper end is at most 0.98. On all 4,000, each side's one-sided miss rate has a 99% lower end of at most 0.040. Each owner cell also needs the exact conditional 99.5% lower bound of its paired gain over today's interval above 1/2 |
| falsifiers | MOVER on `d1_ate` or `d2_ate`; MOVER against the identity interval on both pooled; the skew-widened t on `c1_ey1`, `c1_ate` or `c2_ey1`; the shipped means rule on `d1_ey1` or `d2_ey0`; and a width clause on C3, the 99% bootstrap upper bound of the mean width over the t interval above 1.08 |
| seeds | `seed=2026111301` and `resampling_seed=2026111302`; every group draws from its own stream, keyed by its label |
| policy | `reporting` |

The groups, laws, sizes, fits, controls, falsifiers and design numbers were fixed before the run,
at commit `cd4ac53c`. At perfect calibration the 12 gated cells and the three paired gains pass
together with probability at least 0.9720. With the owner cells at their exploratory readings the
designed chance of a red cell was at most 0.097, almost all of it `ho_hurdle_ey1`.
`tests/unit/test_asymmetric_wald_design.py` recomputes every design number from the law module.
`tests/canonical/asymmetric_wald/regenerate.py` states the declaration and the red-cell reading.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the binary-outcome law with three unequal baseline strata of `canonical-stratified-tmle`, whose stratum V = 2 has a treated mean near 1, n = 2,000 | `ate[V=2]` | average treatment effect in stratum V = 2, with MOVER's interval (a withheld one counts as a miss) | `cleverly` stratified TMLE with the PR 3 interval rule: correlated MOVER on the two arms' range-logit intervals for a bounded difference, the range-logit normal interval for a bounded mean, and the skew-widened t at max(df, 5) for a continuous outcome, with no interval when an arm's stratum holds one outcome value or an estimate sits on a bound | -0.0032 to 0.0018 | 0.9510 | 1.0009 | pass |
<!-- /generated -->

## Theory properties

| family | what it asks | how it fails |
| --- | --- | --- |
| `heldout_coverage` | the shipped interval covers on laws no estimator was fitted on before the declaration, with no one-sided excess | a coverage leg leaves (0.92, 0.98), or a side's miss rate has a 99% lower end above 0.040 |
| `owner_coverage` | the correction repairs the owner cells study 1 refuted | as above, or the paired gain over today's interval does not exceed 1/2 |
| `control` | each gated cell's fits read under the identity normal interval, the range-logit difference interval, a range-logit t, the t without widening and the two-sided Cornish-Fisher shift | reported; the falsifiers of `falsifiers.csv` read them |
| `reported` | the constructions on other targets and laws | reported |
| `witness` | the skew-widened t with the acceleration's sign flipped, and MOVER with each arm's sides swapped | reported; each must fail its heavy-side tail clause |

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `control` | `c1_ate__capped_t_control` | diagnostic | held-out law C1, right-skewed gamma errors scaled by arm, effect in V = 2, n = 2,500: the same fits read under a Student t interval at max(df, 5), without the skew widening | reported with no verdict; a falsifier may read it | lower leg 0.9355 to 0.9613, shown upper end 0.9613, tails 137 below and 66 above, 99% lower ends 0.0273 and 0.0118 | reported |
| `control` | `c1_ate__cf_shift_control` | diagnostic | held-out law C1, right-skewed gamma errors scaled by arm, effect in V = 2, n = 2,500: the same fits read under the two-sided Cornish-Fisher shift by the curve's acceleration | reported with no verdict; a falsifier may read it | lower leg 0.9283 to 0.9555, shown upper end 0.9555, tails 116 below and 109 above, 99% lower ends 0.0226 and 0.0211 | reported |
| `control` | `c1_ate__normal_reference_control` | diagnostic | held-out law C1, right-skewed gamma errors scaled by arm, effect in V = 2, n = 2,500: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9266 to 0.9542, shown upper end 0.9542, tails 151 below and 86 above, 99% lower ends 0.0304 and 0.0160 | reported |
| `control` | `c1_ey1__capped_t_control` | diagnostic | held-out law C1, right-skewed gamma errors scaled by arm, treated mean in V = 2, n = 2,500: the same fits read under a Student t interval at max(df, 5), without the skew widening | reported with no verdict; a falsifier may read it | lower leg 0.9344 to 0.9604, shown upper end 0.9604, tails 144 below and 73 above, 99% lower ends 0.0288 and 0.0132 | reported |
| `control` | `c1_ey1__cf_shift_control` | diagnostic | held-out law C1, right-skewed gamma errors scaled by arm, treated mean in V = 2, n = 2,500: the same fits read under the two-sided Cornish-Fisher shift by the curve's acceleration | reported with no verdict; a falsifier may read it | lower leg 0.9305 to 0.9573, shown upper end 0.9573, tails 110 below and 117 above, 99% lower ends 0.0213 and 0.0228 | reported |
| `control` | `c1_ey1__normal_reference_control` | diagnostic | held-out law C1, right-skewed gamma errors scaled by arm, treated mean in V = 2, n = 2,500: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9244 to 0.9524, shown upper end 0.9524, tails 165 below and 93 above, 99% lower ends 0.0336 and 0.0176 | reported |
| `control` | `c2_ey1__capped_t_control` | diagnostic | held-out law C2, a left-skewed Beta outcome with a coarse covariate, treated mean in V = 2, n = 2,000: the same fits read under a Student t interval at max(df, 5), without the skew widening | reported with no verdict; a falsifier may read it | lower leg 0.9239 to 0.9520, shown upper end 0.9520, tails 51 below and 171 above, 99% lower ends 0.0086 and 0.0349 | reported |
| `control` | `c2_ey1__cf_shift_control` | diagnostic | held-out law C2, a left-skewed Beta outcome with a coarse covariate, treated mean in V = 2, n = 2,000: the same fits read under the two-sided Cornish-Fisher shift by the curve's acceleration | reported with no verdict; a falsifier may read it | lower leg 0.9283 to 0.9555, shown upper end 0.9555, tails 93 below and 125 above, 99% lower ends 0.0176 and 0.0246 | reported |
| `control` | `c2_ey1__normal_reference_control` | diagnostic | held-out law C2, a left-skewed Beta outcome with a coarse covariate, treated mean in V = 2, n = 2,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9129 to 0.9430, shown upper end 0.9430, tails 62 below and 199 above, 99% lower ends 0.0109 and 0.0413 | reported |
| `control` | `c3_ate__capped_t_control` | diagnostic | held-out law C3, symmetric Laplace errors (the no-harm control), effect in V = 2, n = 2,500: the same fits read under a Student t interval at max(df, 5), without the skew widening | reported with no verdict; a falsifier may read it | lower leg 0.9339 to 0.9600, shown upper end 0.9600, tails 110 below and 99 above, 99% lower ends 0.0213 and 0.0189 | reported |
| `control` | `c3_ate__cf_shift_control` | diagnostic | held-out law C3, symmetric Laplace errors (the no-harm control), effect in V = 2, n = 2,500: the same fits read under the two-sided Cornish-Fisher shift by the curve's acceleration | reported with no verdict; a falsifier may read it | lower leg 0.9228 to 0.9511, shown upper end 0.9511, tails 131 below and 123 above, 99% lower ends 0.0259 and 0.0242 | reported |
| `control` | `c3_ate__normal_reference_control` | diagnostic | held-out law C3, symmetric Laplace errors (the no-harm control), effect in V = 2, n = 2,500: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9261 to 0.9538, shown upper end 0.9538, tails 129 below and 115 above, 99% lower ends 0.0255 and 0.0224 | reported |
| `control` | `c3_ey1__capped_t_control` | diagnostic | held-out law C3, symmetric Laplace errors (the no-harm control), treated mean in V = 2, n = 2,500: the same fits read under a Student t interval at max(df, 5), without the skew widening | reported with no verdict; a falsifier may read it | lower leg 0.9278 to 0.9551, shown upper end 0.9551, tails 118 below and 102 above, 99% lower ends 0.0230 and 0.0195 | reported |
| `control` | `c3_ey1__cf_shift_control` | diagnostic | held-out law C3, symmetric Laplace errors (the no-harm control), treated mean in V = 2, n = 2,500: the same fits read under the two-sided Cornish-Fisher shift by the curve's acceleration | reported with no verdict; a falsifier may read it | lower leg 0.9167 to 0.9462, shown upper end 0.9462, tails 150 below and 129 above, 99% lower ends 0.0302 and 0.0255 | reported |
| `control` | `c3_ey1__normal_reference_control` | diagnostic | held-out law C3, symmetric Laplace errors (the no-harm control), treated mean in V = 2, n = 2,500: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9178 to 0.9471, shown upper end 0.9471, tails 145 below and 123 above, 99% lower ends 0.0291 and 0.0242 | reported |
| `control` | `d1_ate__cf_shift_control` | diagnostic | held-out law D1, a sparse treated arm near 0 in a small stratum, effect in V = 2, n = 3,000: the same fits read under the two-sided Cornish-Fisher shift by the curve's acceleration | reported with no verdict; a falsifier may read it | lower leg 0.9129 to 0.9430, shown upper end 0.9443, tails 150 below and 122 above, 99% lower ends 0.0302 and 0.0240 | reported |
| `control` | `d1_ate__logit_t_control` | diagnostic | held-out law D1, a sparse treated arm near 0 in a small stratum, effect in V = 2, n = 3,000: the same fits read under a Student t interval on the range-logit scale at max(df, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9228 to 0.9511, shown upper end 0.9523, tails 191 below and 51 above, 99% lower ends 0.0396 and 0.0087 | reported |
| `control` | `d1_ate__normal_reference_control` | diagnostic | held-out law D1, a sparse treated arm near 0 in a small stratum, effect in V = 2, n = 3,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9031 to 0.9349, shown upper end 0.9361, tails 258 below and 56 above, 99% lower ends 0.0550 and 0.0097 | reported |
| `control` | `d1_ate__range_logit_control` | diagnostic | held-out law D1, a sparse treated arm near 0 in a small stratum, effect in V = 2, n = 3,000: the same fits read under the normal interval on the difference's own range-logit scale | reported with no verdict; a falsifier may read it | lower leg 0.9031 to 0.9349, shown upper end 0.9361, tails 247 below and 64 above, 99% lower ends 0.0524 and 0.0114 | reported |
| `control` | `d1_ey1__logit_t_control` | diagnostic | held-out law D1, a sparse treated arm near 0 in a small stratum, treated mean in V = 2, n = 3,000: the same fits read under a Student t interval on the range-logit scale at max(df, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9622 to 0.9815, shown upper end 0.9827, tails 0 below and 102 above, 99% lower ends 0 and 0.0196 | reported |
| `control` | `d1_ey1__normal_reference_control` | diagnostic | held-out law D1, a sparse treated arm near 0 in a small stratum, treated mean in V = 2, n = 3,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.8912 to 0.9248, shown upper end 0.9261, tails 322 below and 46 above, 99% lower ends 0.0699 and 0.0076 | reported |
| `control` | `d2_ate__cf_shift_control` | diagnostic | held-out law D2, a sparse untreated arm near 1 with a continuous covariate, effect, n = 2,000: the same fits read under the two-sided Cornish-Fisher shift by the curve's acceleration | reported with no verdict; a falsifier may read it | lower leg 0.9222 to 0.9506, shown upper end 0.9506, tails 145 below and 113 above, 99% lower ends 0.0291 and 0.0219 | reported |
| `control` | `d2_ate__logit_t_control` | diagnostic | held-out law D2, a sparse untreated arm near 1 with a continuous covariate, effect, n = 2,000: the same fits read under a Student t interval on the range-logit scale at max(df, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9333 to 0.9595, shown upper end 0.9595, tails 189 below and 33 above, 99% lower ends 0.0390 and 0.0050 | reported |
| `control` | `d2_ate__normal_reference_control` | diagnostic | held-out law D2, a sparse untreated arm near 1 with a continuous covariate, effect, n = 2,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9184 to 0.9475, shown upper end 0.9475, tails 225 below and 38 above, 99% lower ends 0.0473 and 0.0060 | reported |
| `control` | `d2_ate__range_logit_control` | diagnostic | held-out law D2, a sparse untreated arm near 1 with a continuous covariate, effect, n = 2,000: the same fits read under the normal interval on the difference's own range-logit scale | reported with no verdict; a falsifier may read it | lower leg 0.9206 to 0.9493, shown upper end 0.9493, tails 219 below and 42 above, 99% lower ends 0.0459 and 0.0068 | reported |
| `control` | `d2_ey0__logit_t_control` | diagnostic | held-out law D2, a sparse untreated arm near 1 with a continuous covariate, untreated mean, n = 2,000: the same fits read under a Student t interval on the range-logit scale at max(df, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9616 to 0.9810, shown upper end 0.9810, tails 101 below and 19 above, 99% lower ends 0.0193 and 0.0024 | reported |
| `control` | `d2_ey0__normal_reference_control` | diagnostic | held-out law D2, a sparse untreated arm near 1 with a continuous covariate, untreated mean, n = 2,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9020 to 0.9339, shown upper end 0.9339, tails 32 below and 297 above, 99% lower ends 0.0048 and 0.0640 | reported |
| `control` | `ho_hurdle_ey1__capped_t_control` | diagnostic | study 1's held-out law H2, a point mass at 0 and a gamma part, saturated cell-mean outcome model, treated mean in V = 2, n = 2,000: the same fits read under a Student t interval at max(df, 5), without the skew widening | reported with no verdict; a falsifier may read it | lower leg 0.9140 to 0.9439, shown upper end 0.9452, tails 221 below and 56 above, 99% lower ends 0.0464 and 0.0097 | reported |
| `control` | `ho_hurdle_ey1__cf_shift_control` | diagnostic | study 1's held-out law H2, a point mass at 0 and a gamma part, saturated cell-mean outcome model, treated mean in V = 2, n = 2,000: the same fits read under the two-sided Cornish-Fisher shift by the curve's acceleration | reported with no verdict; a falsifier may read it | lower leg 0.9200 to 0.9488, shown upper end 0.9501, tails 146 below and 103 above, 99% lower ends 0.0293 and 0.0198 | reported |
| `control` | `ho_hurdle_ey1__normal_reference_control` | diagnostic | study 1's held-out law H2, a point mass at 0 and a gamma part, saturated cell-mean outcome model, treated mean in V = 2, n = 2,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9015 to 0.9335, shown upper end 0.9348, tails 253 below and 70 above, 99% lower ends 0.0538 and 0.0126 | reported |
| `control` | `ho_sparse_treat_ate__cf_shift_control` | diagnostic | study 1's held-out law H1, a sparse treated arm near 1 with no strata, effect, n = 2,000: the same fits read under the two-sided Cornish-Fisher shift by the curve's acceleration | reported with no verdict; a falsifier may read it | lower leg 0.9372 to 0.9626, shown upper end 0.9626, tails 97 below and 110 above, 99% lower ends 0.0184 and 0.0213 | reported |
| `control` | `ho_sparse_treat_ate__logit_t_control` | diagnostic | study 1's held-out law H1, a sparse treated arm near 1 with no strata, effect, n = 2,000: the same fits read under a Student t interval on the range-logit scale at max(df, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9428 to 0.9670, shown upper end 0.9670, tails 46 below and 152 above, 99% lower ends 0.0076 and 0.0306 | reported |
| `control` | `ho_sparse_treat_ate__normal_reference_control` | diagnostic | study 1's held-out law H1, a sparse treated arm near 1 with no strata, effect, n = 2,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9195 to 0.9484, shown upper end 0.9484, tails 42 below and 239 above, 99% lower ends 0.0068 and 0.0505 | reported |
| `control` | `ho_sparse_treat_ate__range_logit_control` | diagnostic | study 1's held-out law H1, a sparse treated arm near 1 with no strata, effect, n = 2,000: the same fits read under the normal interval on the difference's own range-logit scale | reported with no verdict; a falsifier may read it | lower leg 0.9211 to 0.9497, shown upper end 0.9497, tails 54 below and 218 above, 99% lower ends 0.0093 and 0.0457 | reported |
| `control` | `lognormal06_ey1__capped_t_control` | diagnostic | a continuous outcome with a lognormal error of skewness 2.0, treated mean in V = 2, n = 2,000: the same fits read under a Student t interval at max(df, 5), without the skew widening | reported with no verdict; a falsifier may read it | lower leg 0.9233 to 0.9515, shown upper end 0.9515, tails 216 below and 37 above, 99% lower ends 0.0452 and 0.0058 | reported |
| `control` | `lognormal06_ey1__cf_shift_control` | diagnostic | a continuous outcome with a lognormal error of skewness 2.0, treated mean in V = 2, n = 2,000: the same fits read under the two-sided Cornish-Fisher shift by the curve's acceleration | reported with no verdict; a falsifier may read it | lower leg 0.9167 to 0.9462, shown upper end 0.9462, tails 175 below and 87 above, 99% lower ends 0.0358 and 0.0163 | reported |
| `control` | `lognormal06_ey1__normal_reference_control` | diagnostic | a continuous outcome with a lognormal error of skewness 2.0, treated mean in V = 2, n = 2,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9151 to 0.9448, shown upper end 0.9448, tails 233 below and 53 above, 99% lower ends 0.0491 and 0.0090 | reported |
| `heldout_coverage` | `c1_ate` | positive | held-out law C1, right-skewed gamma errors scaled by arm, effect in V = 2, n = 2,500: the skew-widened t | on the first 2,000 fits, the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98; on all 4,000, each side's one-sided miss rate with a 99% lower end at most 0.040 | lower leg 0.9411 to 0.9657, shown upper end 0.9657, tails 116 below and 66 above, 99% lower ends 0.0226 and 0.0118 | pass |
| `heldout_coverage` | `c1_ey1` | positive | held-out law C1, right-skewed gamma errors scaled by arm, treated mean in V = 2, n = 2,500: the skew-widened t | on the first 2,000 fits, the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98; on all 4,000, each side's one-sided miss rate with a 99% lower end at most 0.040 | lower leg 0.9417 to 0.9661, shown upper end 0.9661, tails 110 below and 73 above, 99% lower ends 0.0213 and 0.0132 | pass |
| `heldout_coverage` | `c2_ey1` | positive | held-out law C2, a left-skewed Beta outcome with a coarse covariate, treated mean in V = 2, n = 2,000: the skew-widened t | on the first 2,000 fits, the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98; on all 4,000, each side's one-sided miss rate with a 99% lower end at most 0.040 | lower leg 0.9400 to 0.9648, shown upper end 0.9648, tails 51 below and 125 above, 99% lower ends 0.0086 and 0.0246 | pass |
| `heldout_coverage` | `c3_ate` | positive | held-out law C3, symmetric Laplace errors (the no-harm control), effect in V = 2, n = 2,500: the skew-widened t | on the first 2,000 fits, the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98; on all 4,000, each side's one-sided miss rate with a 99% lower end at most 0.040 | lower leg 0.9350 to 0.9608, shown upper end 0.9608, tails 108 below and 98 above, 99% lower ends 0.0208 and 0.0186 | pass |
| `heldout_coverage` | `c3_ey1` | positive | held-out law C3, symmetric Laplace errors (the no-harm control), treated mean in V = 2, n = 2,500: the skew-widened t | on the first 2,000 fits, the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98; on all 4,000, each side's one-sided miss rate with a 99% lower end at most 0.040 | lower leg 0.9294 to 0.9564, shown upper end 0.9564, tails 115 below and 102 above, 99% lower ends 0.0224 and 0.0195 | pass |
| `heldout_coverage` | `d1_ate` | positive | held-out law D1, a sparse treated arm near 0 in a small stratum, effect in V = 2, n = 3,000: MOVER's interval | on the first 2,000 fits, the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98; on all 4,000, each side's one-sided miss rate with a 99% lower end at most 0.040 | lower leg 0.9406 to 0.9652, shown upper end 0.9665, tails 50 below and 135 above, 99% lower ends 0.0084 and 0.0269 | pass |
| `heldout_coverage` | `d1_ey1` | positive | held-out law D1, a sparse treated arm near 0 in a small stratum, treated mean in V = 2, n = 3,000: the range-logit normal interval | on the first 2,000 fits, the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98; on all 4,000, each side's one-sided miss rate with a 99% lower end at most 0.040 | lower leg 0.9344 to 0.9604, shown upper end 0.9617, tails 61 below and 123 above, 99% lower ends 0.0107 and 0.0242 | pass |
| `heldout_coverage` | `d2_ate` | positive | held-out law D2, a sparse untreated arm near 1 with a continuous covariate, effect, n = 2,000: MOVER's interval | on the first 2,000 fits, the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98; on all 4,000, each side's one-sided miss rate with a 99% lower end at most 0.040 | lower leg 0.9383 to 0.9635, shown upper end 0.9635, tails 77 below and 119 above, 99% lower ends 0.0141 and 0.0233 | pass |
| `heldout_coverage` | `d2_ey0` | positive | held-out law D2, a sparse untreated arm near 1 with a continuous covariate, untreated mean, n = 2,000: the range-logit normal interval | on the first 2,000 fits, the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98; on all 4,000, each side's one-sided miss rate with a 99% lower end at most 0.040 | lower leg 0.9411 to 0.9657, shown upper end 0.9657, tails 115 below and 78 above, 99% lower ends 0.0224 and 0.0143 | pass |
| `owner_coverage` | `ho_hurdle_ey1` | positive | study 1's held-out law H2, a point mass at 0 and a gamma part, saturated cell-mean outcome model, treated mean in V = 2, n = 2,000: the skew-widened t | on the first 2,000 fits, the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98; on all 4,000, each side's one-sided miss rate with a 99% lower end at most 0.040, and the exact 99.5% lower bound of the paired gain over the t interval at max(df, 5) above 1/2 | lower leg 0.9339 to 0.9600, shown upper end 0.9612, tails 146 below and 56 above, 99% lower ends 0.0293 and 0.0097, paired gain 75 against 0, lower bound 0.9318 | pass |
| `owner_coverage` | `ho_sparse_treat_ate` | positive | study 1's held-out law H1, a sparse treated arm near 1 with no strata, effect, n = 2,000: MOVER's interval | on the first 2,000 fits, the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98; on all 4,000, each side's one-sided miss rate with a 99% lower end at most 0.040, and the exact 99.5% lower bound of the paired gain over the identity normal interval above 1/2 | lower leg 0.9439 to 0.9678, shown upper end 0.9678, tails 121 below and 45 above, 99% lower ends 0.0237 and 0.0074, paired gain 194 against 79, lower bound 0.6348 | pass |
| `owner_coverage` | `lognormal06_ey1` | positive | a continuous outcome with a lognormal error of skewness 2.0, treated mean in V = 2, n = 2,000: the skew-widened t | on the first 2,000 fits, the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98; on all 4,000, each side's one-sided miss rate with a 99% lower end at most 0.040, and the exact 99.5% lower bound of the paired gain over the t interval at max(df, 5) above 1/2 | lower leg 0.9311 to 0.9578, shown upper end 0.9578, tails 175 below and 37 above, 99% lower ends 0.0358 and 0.0058, paired gain 41 against 0, lower bound 0.8788 | pass |
| `reported` | `boundary_1000_ate` | diagnostic | the stratified law of `canonical-stratified-tmle`, effect in V = 2, n = 1,000: MOVER's interval | reported with no verdict: both legs and both tails | lower leg 0.9389 to 0.9639, shown upper end 0.9723, tails 120 below and 30 above, 99% lower ends 0.0237 and 0.0045 | reported |
| `reported` | `boundary_2000_ate` | diagnostic | the stratified law of `canonical-stratified-tmle`, effect in V = 2, n = 2,000: MOVER's interval | reported with no verdict: both legs and both tails | lower leg 0.9451 to 0.9687, shown upper end 0.9687, tails 102 below and 82 above, 99% lower ends 0.0195 and 0.0152 | reported |
| `reported` | `c1_ey0` | diagnostic | held-out law C1, untreated mean in V = 2, n = 2,500: the skew-widened t | reported with no verdict: both legs and both tails | lower leg 0.9439 to 0.9678, shown upper end 0.9678, tails 94 below and 79 above, 99% lower ends 0.0178 and 0.0145 | reported |
| `reported` | `c2_ate` | diagnostic | held-out law C2, effect in V = 2, n = 2,000: the skew-widened t | reported with no verdict: both legs and both tails | lower leg 0.9305 to 0.9573, shown upper end 0.9573, tails 83 below and 111 above, 99% lower ends 0.0154 and 0.0215 | reported |
| `reported` | `c2_ey0` | diagnostic | held-out law C2, untreated mean in V = 2, n = 2,000: the skew-widened t | reported with no verdict: both legs and both tails | lower leg 0.9327 to 0.9591, shown upper end 0.9591, tails 92 below and 116 above, 99% lower ends 0.0173 and 0.0226 | reported |
| `reported` | `c3_ey0` | diagnostic | held-out law C3, untreated mean in V = 2, n = 2,500: the skew-widened t | reported with no verdict: both legs and both tails | lower leg 0.9311 to 0.9578, shown upper end 0.9578, tails 96 below and 112 above, 99% lower ends 0.0182 and 0.0217 | reported |
| `reported` | `d1_ate_v0` | diagnostic | held-out law D1, effect in the large stratum V = 0, n = 3,000: MOVER's interval | reported with no verdict: both legs and both tails | lower leg 0.9355 to 0.9613, shown upper end 0.9613, tails 78 below and 110 above, 99% lower ends 0.0143 and 0.0213 | reported |
| `reported` | `d1_ey0` | diagnostic | held-out law D1, untreated mean in V = 2, n = 3,000: the range-logit normal interval | reported with no verdict: both legs and both tails | lower leg 0.9361 to 0.9617, shown upper end 0.9617, tails 89 below and 104 above, 99% lower ends 0.0167 and 0.0200 | reported |
| `reported` | `d2_ey1` | diagnostic | held-out law D2, treated mean, n = 2,000: the range-logit normal interval | reported with no verdict: both legs and both tails | lower leg 0.9473 to 0.9704, shown upper end 0.9704, tails 89 below and 85 above, 99% lower ends 0.0167 and 0.0158 | reported |
| `reported` | `ho_hurdle_ate` | diagnostic | study 1's held-out law H2, effect in V = 2, n = 2,000: the skew-widened t | reported with no verdict: both legs and both tails | lower leg 0.9272 to 0.9546, shown upper end 0.9559, tails 143 below and 75 above, 99% lower ends 0.0287 and 0.0137 | reported |
| `reported` | `ho_hurdle_ey0` | diagnostic | study 1's held-out law H2, untreated mean in V = 2, n = 2,000: the skew-widened t | reported with no verdict: both legs and both tails | lower leg 0.9400 to 0.9648, shown upper end 0.9660, tails 142 below and 47 above, 99% lower ends 0.0284 and 0.0078 | reported |
| `reported` | `identity_msm_v2_w` | diagnostic | the alternating Beta-outcome law, identity working MSM, W coefficient in V = 2, n = 2,000: the skew-widened t | reported with no verdict: both legs and both tails | lower leg 0.9417 to 0.9661, shown upper end 0.9661, tails 108 below and 89 above, 99% lower ends 0.0208 and 0.0167 | reported |
| `reported` | `lognormal_ey1` | diagnostic | a continuous outcome with a lognormal error of skewness 6.2 (the stated limit), treated mean in V = 2, n = 2,000: the skew-widened t | reported with no verdict: both legs and both tails | lower leg 0.9009 to 0.9330, shown upper end 0.9330, tails 302 below and 10 above, 99% lower ends 0.0651 and 0.000930 | reported |
| `reported` | `rare_ate` | diagnostic | a rare binary outcome, three strata, effect in V = 2, n = 2,000: MOVER's interval | reported with no verdict: both legs and both tails | lower leg 0.9344 to 0.9604, shown upper end 0.9730, tails 56 below and 91 above, 99% lower ends 0.0098 and 0.0174 | reported |
| `reported` | `student_ate` | diagnostic | a continuous outcome with a Student t error on 3 df, effect in V = 2, n = 2,000: the skew-widened t | reported with no verdict: both legs and both tails | lower leg 0.9451 to 0.9687, shown upper end 0.9687, tails 81 below and 83 above, 99% lower ends 0.0150 and 0.0154 | reported |
| `reported` | `student_ey1` | diagnostic | a continuous outcome with a Student t error on 3 df, treated mean in V = 2, n = 2,000: the skew-widened t | reported with no verdict: both legs and both tails | lower leg 0.9479 to 0.9709, shown upper end 0.9709, tails 83 below and 91 above, 99% lower ends 0.0154 and 0.0171 | reported |
| `witness` | `mover_swap_witness` | diagnostic | the fits of ho_sparse_treat_ate read under MOVER with each arm's two sides swapped, a deliberate mutation | reported: it must fail its heavy-side tail clause, which shows the clause sees a wrong pairing | tails 12 below and 474 above, 99% lower ends 0.0012 and 0.1057 | reported |
| `witness` | `sign_flip_witness` | diagnostic | the fits of ho_hurdle_ey1 read under the skew-widened t with the acceleration's sign flipped, a deliberate mutation | reported: it must fail its heavy-side tail clause, which shows the clause sees a wrong sign | tails 221 below and 37 above, 99% lower ends 0.0464 and 0.0058 | reported |
<!-- /generated -->

## Result

The primary row passes: MOVER covers `ate[V=2]` in 0.951 of fits with an SE ratio of 1.001. All
12 gated cells pass, the three paired gains clear 1/2, and no falsifier fired. Both witnesses
fail their tail clauses, so the clause can tell a wrong sign or a wrong pairing. A post-run review
recomputed every verdict from the committed replicates through the fast-tier code and found the
run as declared.

| reading | value | source |
| --- | --- | --- |
| primary tests passed | 1 of 1 | `performance-tests.csv` |
| gated property cells passed | 12 of 12 | `properties.csv` |
| failed fits | 5 of 4,000 in group `ho_hurdle` (an empty design cell; 6.2 expected) | `property-failures.csv` |

**Bounded differences.** On the held-out laws MOVER covers 0.952 (D1) and 0.951 (D2), and the
identity normal interval on the same fits 0.920 and 0.934. Pooled over both, MOVER covers 356
fits that the identity interval misses and misses 160 that it covers. The identity interval's
misses fall on one side, 258 below against 56 above on D1. MOVER moves them to the other side and
shrinks them, 50 below against 135 above, the light-side over-correction that the logit scale of
a binary mean predicts. On H1, the owner cell, MOVER covers 0.958 with 121 misses below and 45
above, against 0.930 for the identity interval.

**Continuous targets.** The skew-widened t covers 0.954 on both C1 cells and 0.956 on C2's
treated mean, and on the owner cells 0.949 (H2) and 0.947 (lognormal 0.6), against 0.931 and
0.937 for the t without widening. Its heavy side keeps a residue: 175 of 4,000 lognormal fits
miss below, inside the clause's 193 but above the nominal 100. The construction removes the
skewness term of the studentized statistic and leaves its median-bias term, so no per-tail rate is
claimed. On the symmetric law C3 the interval is 2.7% and 2.2% wider than the t interval, with
99% upper bounds of 1.028 and 1.023 against the 1.08 the width clause allows.

No falsifier fired, and each owner cell cleared its paired gain over the interval it replaces, so
the owners `F3-difference-near-bound` and `F3-skewed-continuous` closed. Study 1's committed rows
of those cells moved to `F3-normal-reference`, the standing record of replaced intervals in
[red cells](red-cells.md).

Each failed fit is an H2 sample whose treated or untreated cell of stratum 2 is empty, so the
saturated outcome model has no row to fit there. The accounting counts each as a miss in the
lower leg and as a one-sided miss on the side read by each tail clause, and every leg passes at
that worst case (`property-failure-accounting.csv`).

**What the study does not cover.** The lognormal law of skewness 6.2 stays a limit: the
skew-widened t covers 0.922, with 302 misses below. H2's SE shortfall from its saturated
learner's leverage, an SE ratio of 0.93 to 0.96, is not an interval question and is reported, not
corrected.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | primary replications |
| `n` | 2000 | observations per replication |
| `independent_tests_total` | 1 | parameter tests against the truth |
| `independent_tests_passed` | 1 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 12 | gated repeated-sampling property cells |
| `property_cells_passed` | 12 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0237 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9510 | lowest measured primary coverage, MOVER's interval |
| `min_coverage_ci_lower` | 0.9307 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `properties[heldout_coverage/d1_ate]:coverage` | 0.9520 | held-out difference, arm near 0 |
| `properties[heldout_coverage/d2_ate]:coverage` | 0.9510 | held-out difference, arm near 1 |
| `properties[heldout_coverage/c1_ey1]:coverage` | 0.9543 | held-out right-skewed mean |
| `properties[heldout_coverage/c2_ey1]:coverage` | 0.9560 | held-out left-skewed mean |
| `properties[heldout_coverage/c3_ey1]:width_ratio` | 1.0271 | symmetric control, width over the t interval |
| `properties[heldout_coverage/c3_ey1]:width_ratio_upper` | 1.0282 | its 99% upper bound, against 1.08 |
| `properties[owner_coverage/ho_sparse_treat_ate]:coverage` | 0.9583 | owner difference near a bound |
| `properties[owner_coverage/ho_sparse_treat_ate]:gain_lower` | 0.6348 | its paired gain's lower bound, against 1/2 |
| `properties[owner_coverage/ho_hurdle_ey1]:coverage` | 0.9494 | owner skewed mean, over the completed fits |
| `properties[owner_coverage/lognormal06_ey1]:coverage` | 0.9470 | owner lognormal mean, skewness 2.0 |
| `properties[owner_coverage/lognormal06_ey1]:below_ci_lower` | 0.0358 | its heavy side's 99% lower end, against 0.040 |
| `properties[reported/lognormal_ey1]:coverage` | 0.9220 | lognormal mean, skewness 6.2, the stated limit |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in error SDs |
| `margin:coverage_floor` | 0.9000 | validity floor the primary coverage lower endpoint must clear |
| `margin:subject_failure_bound` | 0.0100 | the failed-fit bound of the committed run |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power control must clear |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical SDs |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:subject_failures_allowed` | 10 | failed primary fits that bound allows |
| `margin:type_i_ceiling` | 0.1000 | largest size a one-sided type-I bound may establish |

## Limitations

| limitation | what it means for use |
| --- | --- |
| A per-tail rate is not derived | The skew-widened t removes the curve's skewness term and leaves the median-bias term of the same order, so the claim is what the tail clause tests: no one-sided miss rate shown above 0.040 on the registered laws. Its light side keeps the t limit and is conservative on a skewed law |
| Heavy skew in a small stratum | The lognormal law of skewness 6.2 in a stratum of about 200 rows covers 0.922. The correction needs a finite sixth moment of the curve, which a heavy lognormal tail strains |
| Width on a symmetric law | The widening reads a noisy acceleration, so it costs width where the law is symmetric: 2.2% to 2.7% on C3 |
| MOVER's scope | Only a difference whose builder records its two arm means (`components`) reads MOVER: point-treatment, stratum and longitudinal differences of binary means, and a contrast of two such means. A difference built otherwise, such as the median over repeated cross-fitting splits or a clustered fit, keeps the identity normal interval and says so (`interval_scale`) |
| MOVER near a correlation of 1 | With unequal arms and a correlation above about 0.87 a MOVER limit can stop widening as the level rises; the p-value is the largest level at which the interval still holds the null. A TMLE's arm means share only their outcome-regression terms, so their correlation is usually near 0 (median 0.003 to 0.007 here) |
| Clustered estimates, bands and ratios | Keep their intervals: a clustered estimate its cluster reference, a band its own construction, a ratio of continuous means the log-scale normal interval |
| Laws now seen | D1, D2 and C1 to C3 have been read. A revised construction chosen because it wins on them is in-sample, so it needs its own declaration on new held-out laws |

## Reproduction

The declared run is `python -m tests.canonical.asymmetric_wald.regenerate --output <empty
directory>`, whose docstring gives the run form, the declaration and the red-cell reading. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/asymmetric_wald/manifest.json)
records the seeds, the margins, the configuration, the source hashes and the result hashes. The
[run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/asymmetric_wald/run.log)
records the commit, the clean tree and the wall time. The
[property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/asymmetric_wald/properties.csv),
the [falsifiers](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/asymmetric_wald/falsifiers.csv)
and the [shipped interval of the primary fits](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/asymmetric_wald/primary-shipped.csv)
carry every published row.
