# Smoothed dose-response curve

## What this solves

For a continuous treatment $A$, the dose-response curve $\theta(a) = E\{\bar Q(a, W)\}$, with
$\bar Q(a, W) = E(Y \mid A = a, W)$, is not pathwise differentiable at a fixed dose in a
nonparametric model, so no root-$n$ interval exists for it. `DoseResponseTMLE` estimates its
kernel-smoothed version at declared bandwidths and doses, which is pathwise differentiable. It
reports a Wald interval per declared pair and one simultaneous band over every declared pair. The
band also covers the smoothed curve at a bandwidth chosen after the fit, provided the choice is one
of the declared bandwidths.

| claim | status |
| --- | --- |
| point estimate and interval for $\theta_h(a_0)$ at a declared $(h, a_0)$ | ships |
| a simultaneous band over every declared dose and bandwidth | ships |
| the band read at a bandwidth chosen after seeing the data | ships, through the band only. A pointwise interval at a chosen bandwidth is not adjusted for the choice |
| a bandwidth selector | not shipped. The band makes the comparison across declared bandwidths |
| an interval for the unsmoothed curve $\theta(a_0)$, or a band over every dose in an interval | refused, [F39](../roadmap.md#f39-unsmoothed-dose-response-curve) |

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from cleverly.dose_response import DoseResponse
from cleverly.estimators import DoseResponseTMLE

rng = np.random.default_rng(0)
n = 800
w = rng.normal(size=n)
a = np.clip(2.5 + 0.8 * w + rng.normal(size=n), -1.0, 6.0)
y = rng.binomial(1, 1 / (1 + np.exp(-(0.6 * np.sin(a) + 0.5 * w))))
frame = pd.DataFrame({"W": w, "A": a, "Y": y})

curve = DoseResponse(doses=[1.0, 2.5, 4.0], bandwidths=[0.8, 1.2], support=(0.5, 4.5))
estimator = DoseResponseTMLE(
    curve,
    outcome_learner=HistGradientBoostingClassifier(max_depth=2, max_iter=50),
    treatment_learner=LogisticRegression(),
    density_bins=10,
    random_state=0,
)
result = estimator.fit(frame, outcome="Y", treatment="A", covariates=["W"])
print(result.to_frame()[["bandwidth", "dose", "estimate", "band_lower", "band_upper"]])
```

`cleverly.plotting.curve(result)` draws one series per bandwidth, with the band. The
[dose-response tutorial](../examples/dose-response.ipynb) walks through one analysis.

## Estimand

The user declares, without reading the data, a dose interval $D = [a_\mathrm{lo}, a_\mathrm{hi}]$,
evaluation doses $a_{0,1} < \dots < a_{0,J}$ in $D$, distinct bandwidths $h_1 < \dots < h_L$, and a
kernel $K$ (`"epanechnikov"`, the default, `"triweight"` or `"uniform"`). For each pair $(h, a_0)$
the weight is the local-linear equivalent kernel under Lebesgue measure on $D$,

$$
w_{h,a_0}(a) = e_1^\top M_{h,a_0}^{-1}\,\bigl(1, (a - a_0)/h\bigr)^\top K_h(a - a_0)\,1\{a \in D\},
\qquad
M_{h,a_0} = \int_D \bigl(1, \tfrac{a - a_0}{h}\bigr)^\top \bigl(1, \tfrac{a - a_0}{h}\bigr)
K_h(a - a_0)\,da,
$$

with $K_h(u) = K(u/h)/h$, and the target is

$$
\theta_h(a_0) = \int_D w_{h,a_0}(a)\,\theta(a)\,da .
$$

| property | consequence |
| --- | --- |
| $w$ depends on $(K, h, a_0, D)$ only | $\theta_h$ is a known linear functional of $\theta$ |
| $\int_D w = 1$ and $\int_D w(a)(a - a_0)\,da = 0$ | $\theta_h(a_0) = \theta(a_0)$ wherever $\theta$ is linear on the support of $w$, at interior and boundary doses |
| interior pair, symmetric $K$: $w = K_h(\cdot - a_0) \ge 0$ | the target is the kernel average of Bibaut and van der Laan (2017), Section 1.3. Under `fluctuation="logistic"` the estimate lies in the outcome's range; the linear submodel can leave it |
| boundary pair: $w$ is signed | for Epanechnikov at $a_0 = a_\mathrm{lo}$, $\min w \approx -0.92/h$, $\int \lvert w \rvert \approx 1.57$ and $\int w^2$ is about 7.5 times its interior value. The estimate can leave the outcome's range, and its variance is larger |
| $\theta \in C^2$ near $a_0$ | $\lvert \theta_h(a_0) - \theta(a_0) \rvert \le \tfrac12 \sup \lvert\theta''\rvert\, h^2 \int \lvert w \rvert$, so the smoothing bias is $O(h^2)$ uniformly over $D$ |

$\theta_h(a_0)$ is also the intercept of the least-squares projection of $\theta$ onto the working
model $\beta_0 + \beta_1 (a - a_0)/h$ with the known weight $K_h(a - a_0) 1\{a \in D\}$. That is the
projection parameter of Neugebauer and van der Laan (2003), Section 2.1, Equation (1), with the sum
over regimes replaced by an integral over doses, as the shipped [continuous MSM](msm-projections.md)
does. Lebesgue measure keeps the weight known. Kennedy, Ma, McHugh and Small (2017) weight by the
marginal dose density instead, which is a functional of $P$ with its own influence-function term.

$D$ is part of the estimand. `DoseResponse` never reads $A$, and an interval set from sample
quantiles of $A$ moves the target at boundary pairs by $O_P(n^{-1/2})$, a term the reported
interval omits. Identification needs consistency, conditional exchangeability, and positivity:
$g_0(a \mid W) \ge c > 0$ for every $a$ in the union of the declared weights' supports, almost
surely.

## Influence function and remainder

The efficient influence function of $\theta_h(a_0)$ in the nonparametric model is

$$
D_{h,a_0}(P)(O) = \frac{w_{h,a_0}(A)}{g(A \mid W)}\{Y - \bar Q(A, W)\}
+ \int_D w_{h,a_0}(a)\,\bar Q(a, W)\,da - \theta_h(a_0).
$$

Bibaut and van der Laan (2017), Section 1.3, page 5, give this form under a known $g$. Their PDF
prints $\bar Q(a, W)$ in the first term, a typo for $\bar Q(A, W)$. The step to an estimated $g$ is
this project's derivation: $\theta_h$ depends on $P$ only through $(P_W, \bar Q)$, so its pathwise
derivative along every score of $g(\cdot \mid W)$ is zero, and $D$ is then the gradient in the full
nonparametric model, where the gradient is unique, provided $w/g_0 \in L_2(P_0)$. Bibaut and van
der Laan state $\operatorname{Var} D \sim C h^{-1}$, so a pair's standard error is about
$(nh)^{-1/2}$.

The exact remainder is

$$
\theta_h(P) - \theta_h(P_0) + P_0 D_{h,a_0}(P)
= E_0 \int_D w_{h,a_0}(a)\,\frac{g(a \mid W) - g_0(a \mid W)}{g(a \mid W)}
\{\bar Q(a, W) - \bar Q_0(a, W)\}\,da .
$$

It needs $g > 0$ where $w \ne 0$. No regression or mechanism is fitted on an estimated index, so the
linear-remainder audit does not apply. The interval needs both nuisances consistent, in two forms.

- Asymptotically, at a fixed $h$, the product of the $L_2$ errors of $\hat g$ and $\hat{\bar Q}$ over
  the kernel's support is $o_P(n^{-1/2})$, and $1/\hat g$ is bounded there.
- In a finite sample, the remainder is small next to the pair's own standard error, about
  $(nh)^{-1/2}$. A narrower bandwidth asks for better nuisances.

With one nuisance inconsistent the remainder is first order. The estimate stays consistent, and the
interval is not claimed.

## The estimator

The route is the generic CV-TMLE theorem of Zheng and van der Laan (2011), Theorems 1 and 2, whose
inputs are the canonical gradient and the second-order remainder above.

1. Cross-fit $\bar Q$ and $g$ over `n_folds` folds. $g$ is the package's pooled-hazard binned
   conditional density on a core of `density_bins` bins, refined in the outer bins and read as a
   corrected log frequency polygon
   ([the grid and reading paragraphs](longitudinal-tmle.md#modified-treatment-policies-at-a-node)).
   $\bar Q$ is any outcome learner, with the dose as a feature.
2. Build one quadrature rule shared by every pair: a composite Gauss-Legendre rule with
   `nodes_per_piece` nodes on each piece of the partition of $D$ at the density's bin edges and
   bin centres, every $a_0 \pm h$ inside $D$, and $D$'s endpoints. The weight is a polynomial on
   each piece and $\log \hat g$ is linear there, so $1/\hat g$ is smooth on every piece and the
   rule integrates it with $\bar Q$ and the logistic map
   (`tests/unit/test_density_reconstruction.py::test_the_dose_response_rule_breaks_at_the_bin_centres`).
3. For each pair, fit one pooled one-dimensional fluctuation over all rows, with the out-of-fold
   clever covariate $H(A, W) = w_{h,a_0}(A)/\hat g(A \mid W)$:
   $\operatorname{logit} \bar Q^*(a, W) = \operatorname{logit} \bar Q(a, W) + \varepsilon H(a, W)$
   on the scaled outcome (`fluctuation="logistic"`, the default), or
   $\bar Q^* = \bar Q + \varepsilon H$ (`fluctuation="linear"`). No fluctuation is shared between
   pairs.
4. Report $\hat\theta_h(a_0) = n^{-1}\sum_i \int_D w_{h,a_0}(a)\,\bar Q^*(a, W_i)\,da$ by the rule
   of step 2. The submodel is one function of $(a, W)$ read at the nodes, so the score equation
   $P_n D(\hat P^*) = 0$ holds once the fluctuation is solved. `result.scores` records each pair's
   score at its solution.
5. Form each pair's Wald interval from its cross-fitted influence curve, and the band from the
   stacked curves by the studentized max-$t$ multiplier bootstrap (`simultaneous_bands`).

The fit repeats steps 4 and 5 with twice the nodes and records the largest change relative to a
pair's standard error as `result.quadrature_shift`. It adds a note to `result.warnings` when that
exceeds 0.05, and warns with `DataWarning`. A tree-based $\bar Q$ puts its own jumps inside pieces, where the rule converges more
slowly.

The fit refuses before any learner runs when $D$ leaves the observed dose range, since the binned
density is zero outside it, and when the quadrature stage is predicted to need more than 4 GiB
(`max_memory_bytes=` sets another budget; the
[refusal table](scope-and-refusals.md) gives the count). It refuses after the
nuisance fit, naming the pair, when the fitted density is zero where a weight is nonzero, and when
a pair's fluctuation does not solve its score. Both raise `DataError`. Declare
$D$ with a margin inside the expected dose range, and choose it without reading the sample: an
endpoint set from the sample changes the target.

A declared conditional dose support, `PointTreatment(treatment_support=Support(lower, upper))`,
asserts $g_0(a \mid W) = 0$ outside $[l(W), u(W)]$. The fit refuses an observed dose outside its
row's support (`DataError`, the policy route's wording), and then refuses, with
`invalid_declaration`, every pair whose kernel support $[\max(a_0 - h, \ell_D),
\min(a_0 + h, u_D)]$ some row's support does not cover, naming the pairs and the row count:
positivity fails there by declaration. A support that covers every pair's kernel on every row
changes nothing. `dose_response_preflight` runs these checks and the refusals of the
[compositions table](#compositions), for the direct fit and for `CausalStudy.identify`.

## Study route

`CausalStudy.identify(DoseResponseCurve(curve))` binds the declaration to a `PointTreatment`
design with one continuous dose, and `estimate()` fits `DoseResponseTMLE` on the study's data.
The route adds no estimand and no estimation step: `tests/unit/test_dose_response_study.py`
checks that it equals the direct fit bit for bit in every estimate, variance, influence curve,
band limit, support record and label, on a binary outcome with and without strata, a continuous
outcome with the linear fluctuation, an in-sample fit and a weighted fit. The registered study
therefore covers the route exactly where it covers the direct fit.

The identification record states consistency and no unmeasured confounding for every dose in
$D$, no interference, positivity over every declared pair's kernel, that the declaration is made
without reading the data, and that each target is the smoothed curve, not $\theta(a_0)$. With
strata it adds positivity within each stratum, and with `treatment_adjustment=` the mechanism
restriction. Its remainder condition is the product form above.

`estimate()` takes `method="tmle"` and reads these `TMLEMethod` fields, passing each to the
engine keyword of the same meaning: `outcome_learner`, `treatment_learner`, `density_bins`,
`cross_fit`, `n_folds`, `learner_folds`, `fluctuation`, `q_bounds`, `alpha`, `n_multiplier`
(the engine's `n_replicates`), `multiplier_kind`, `band_reference`, `random_state`, `run_id` and
`n_jobs`. `TMLEMethod` defaults to 10 folds, and the registered study ran 5; the label does not
read the fold count. Every other method and field is refused before any learner:

| request | code |
| --- | --- |
| `LongitudinalTreatment`, `TimeToEvent`, `LongPanel` | `implementation_gap`, [X95](../roadmap.md#x95-dose-response-intervals-below-the-support-floor) part (e) |
| `TrialWithExternalData`, a discrete or vector treatment | `invalid_declaration` |
| `collaborative_tmle`, `drtmle` | `implementation_gap`, F5 |
| `adaptive_iptw` | `invalid_declaration`: a continuous dose has no arms |
| `repeats > 1`, `split_plan`, `n_bootstrap`, `bootstrap_resampling`, treatment screening | `implementation_gap`, X95 part (e) |
| `stratify_by` other than `"none"` | refused for every fit when `CrossFitting` is built ([fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules)) |
| `targeting_scheme="fold"`, `fold_evaluation=True` | `inference_gap`: no reviewed interval covers a fold-specific fluctuation of a smoothed pair |
| `simultaneous=False` | `invalid_declaration`: the band is always reported |
| any other `ModelSpec`, `Targeting` or `Inference` field | `invalid_declaration`: the curve does not read it |

The result is the same `DoseResponseResult`, with `identified_effect`, `method`, one
`ParameterKey` per pair (axis `"dose"`, `value` the dose, `bandwidth` the bandwidth) and a
`provenance` whose `protocol_fingerprint` is the study protocol's. `result.summary()` prints the
pairs and the band; its full form adds the identification assumptions, the declaration and the
study description.

## Support

`result.pair_support` holds one entry per reported estimate, and `result.to_frame()` repeats its main
columns.

| field | meaning |
| --- | --- |
| `rows` | rows whose observed dose gets a nonzero weight |
| `kish` | the Kish effective size of the clever covariate, $(\sum_i \lvert H_i \rvert)^2 / \sum_i H_i^2$, without observation weights |
| `min_density` | the smallest fitted density over the loaded observed doses and the pair's nodes |
| `max_covariate` | the largest $\lvert H_i \rvert$ |
| `below_floor` | whether `kish` is below `DoseResponse.support_floor` (default 50) |

A pair below the floor keeps its interval and its place in the band, and its estimate carries the
inference status `low_dose_support_plugin`, and the fit warns with `DataWarning`. The label says
that the normal approximation behind the interval rests on few effective rows. The default floor
is provisional, and the label is not a validated screen.

In the
[dose-response pilot](method-evidence/dose-response.md#sizing) the Kish size did not separate the
intervals that under-cover: an unlabelled stratified pair with a mean Kish size of 69 covered
0.908 at n = 4,000, and at the tail pair a larger Kish size came with a smaller reported SE and no
smaller error. The registered run repeats both readings. The stratified pair at n = 8,000 (mean
Kish size 136, all but 3 of 4,000 fits unlabelled) covered 0.921 with an SE ratio of 0.96. At the
`tail` pair the unlabelled fits covered 0.922 and the labelled ones 0.948. The `tail` scenario
measures one pair near the floor, so its passing verdict does not validate the floor at other
doses, bandwidths or laws.
[X95](../roadmap.md#x95-dose-response-intervals-below-the-support-floor) records what would remove
the label.

## Compositions

| composition | status |
| --- | --- |
| cross-fitted fit, with or without baseline strata | accepted. A continuous $Y$ needs `q_bounds=`, its declared range, as every cross-fitted fit does ([fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules)) |
| in-sample fit (`cross_fit=False`), observation weights | accepted, with an `experimental:` note in `result.warnings` and the `experimental` support label. No registered study measures them |
| clustered rows (`id=`) | refused, `implementation_gap`, [X95](../roadmap.md#x95-dose-response-intervals-below-the-support-floor) part (b). The intervals would need the few-cluster rule |
| a dose with fewer than 20 distinct values | refused, `invalid_declaration`. A discrete dose has no density between its values. Fit its levels as arms with `TMLE` |
| `one_step` targeting | not offered. Each pair takes its own one-dimensional fluctuation |
| baseline strata | each stratum $v$ gets its own curve, $E\{\int_D w(a) \bar Q(a, W)\,da \mid V = v\}$, from one nuisance fit, named with a `[V=v]` suffix |
| a binary, categorical or vector treatment | refused, `invalid_declaration` |
| missing outcome, missing treatment, intermediate variable | refused, `implementation_gap`, roadmap X10. The second mechanism would have to be evaluated at every node |
| a treatment with declared dose atoms, or a declared known treatment mechanism | refused, `implementation_gap`; [X95](../roadmap.md#x95-dose-response-intervals-below-the-support-floor) part (c) |
| a declared conditional dose support (`treatment_support=`) | accepted when it covers every pair's kernel on every row; otherwise refused, `invalid_declaration` ([The estimator](#the-estimator)) |
| a declared treatment-mechanism subset (`treatment_adjustment=`) | accepted, `experimental`. The influence function lies in the tangent space of a model that restricts only $g$, so it is unchanged; no registered study fits a restricted density |
| `CausalStudy` | `CausalStudy.identify(DoseResponseCurve(curve))` ([Study route](#study-route)) |

## Support label

Each pair's estimate carries `support.label`, and the band carries `result.band.support`. The
label reads `supported` when the [registered study](method-evidence/dose-response.md) measured the
fit's configuration, which
`cleverly.estimators.dose_response.DOSE_RESPONSE_MEASURED` lists. Every measured configuration is
cross-fitted, on iid unweighted rows with complete outcomes.

| outcome | fluctuation | strata | pointwise intervals | band | study scenario |
| --- | --- | --- | --- | --- | --- |
| binary | `"logistic"` (default) | none | `supported` | `supported` | `binary`, `tail` |
| binary | `"logistic"` (default) | baseline strata | `supported` | `experimental` | `strata` |
| continuous, with `q_bounds=` | `"linear"` | none | `supported` | `experimental` | `gaussian` |
| any other combination | | | `experimental` | `experimental` | none |

A declared treatment-mechanism subset reads `experimental` in every row of this table.

The band reads `supported` only where the study's band cell measured it, on the 15 pairs of the
binary declaration, with at least 10,000 Mammen draws and the normal reference; another
`multiplier_kind` or `band_reference` reads `experimental`. The pairs' estimates carry no range, so the
band stays on the identity scale and a limit can leave [0, 1]. A pair below the support floor reads `experimental` with
[X95](../roadmap.md#x95-dose-response-intervals-below-the-support-floor) as its evidence, and a
band over such a pair reads `experimental` too. The label does not read the learners, the sample
size, the number or spacing of the pairs, or the law. The study measured declared sieve learners,
one law family and sizes from 1,000 to 8,000, and each label's `limits` repeat that.

## Assessing the curve

`result.diagnostics` declares the `dose_response` assessment family, and `result.validate()` and
`result.assess()` read its rows. Every supported row restates numbers the fit stores. No row
refits, so a saved fit answers them too.

- `support()` returns a `DoseResponseSupportReport`: the `pair_support` records, the floor, the
  number of pairs below it and the quadrature shift. Its row reads `completed`, or `warning`
  when some pair is below the floor, naming the pairs that carry the experimental
  `low_dose_support_plugin` interval. It never reads `passed`, because the floor is provisional
  ([X95](../roadmap.md#x95-dose-response-intervals-below-the-support-floor)).
- `score_equations()` returns a `DoseResponseScoreReport`: each pair's mean fluctuation score at
  its solution, on the scaled outcome, against the package's score threshold, read with each
  pair's plug-in standard error divided by the outcome scaler's range. The fit already refuses a
  pair whose fluctuation does not solve its score to within about $10^{-6}\,\widehat{sd}/\sqrt{n}$,
  while the package threshold is about $10^{-3}\,\widehat{sd}/n$. Below about a million rows
  the solver's bound is the tighter one (by a factor of about 40 at n = 600 and 10 at n = 8,000),
  so on such a fit this row restates convergence; near a million rows and above it can fail.
- `nuisance_models()` reports the outcome regression at the observed dose, from a record of
  $n$ floats the fit keeps. It is out of fold on a cross-fitted fit and in sample with
  `cross_fit=False`, and its evaluation label says which.
- `influence()` reports each pair's influence curve, as on a point fit.

`corrections` and `truncation_curve` do not apply: each pair is targeted by its own
one-dimensional fluctuation, and no bound enters the clever covariate, which divides by the
fitted density without a floor. `refute` and the omitted-variable and marginal bounds are
`implementation_gap` refusals that name
[X82](../roadmap.md#x82-refutation-and-sensitivity-checks-for-every-estimator) part K, and the
simulated common cause names
[X66](../roadmap.md#x66-simulated-confounding-beyond-x82-part-f) part (l). The E-value does not
apply, because a pair is a mean at one dose and not a contrast. The missingness tilt and its
tipping point do not apply, because the fit takes no missing outcome. `replayability` reads
summaries only: the result keeps no estimator or data. `tests/unit/test_dose_response_assessment.py`
checks each row, the score scale and the out-of-fold source of the outcome record.

## Validation

| check | test |
| --- | --- |
| the pathwise derivative along outcome and covariate scores equals $E[D s]$ on an exact law with three covariate levels and a piecewise-linear dose density, at interior and boundary pairs | `tests/unit/test_dose_response.py::test_outcome_score_derivative_equals_the_influence_curve`, `::test_covariate_score_derivative_equals_the_influence_curve` |
| the derivative along a score of the dose law is zero, and so is its inner product with $D$. This is a structural identity: it holds for any $D$ of the form $f(A, W)\{Y - \bar Q_0\} + c(W)$, so it cannot fail for a wrong clever covariate, which the rows above and below catch | `::test_a_score_of_the_dose_law_has_zero_derivative_and_zero_inner_product` |
| deliberate mutations (drop $1/g$, read $g$ at $a_0$, omit the integral term) each fail a derivative check | `::test_each_mutation_breaks_a_derivative_check` |
| the exact remainder equals the product form, is nonzero with both nuisances wrong, and vanishes with either one right | `::test_the_exact_remainder_is_the_product_and_vanishes_with_one_correct_nuisance` |
| the equivalent kernel's moments and sign, interior and boundary | `::test_equivalent_kernel_moments_and_sign` |
| the composite rule is exact on a piecewise polynomial and fails without its breaks | `::test_the_composite_rule_is_exact_on_a_piecewise_polynomial_and_needs_its_breaks` |
| the estimator's own targeting (`DoseResponseTMLE._target` and `target_pair`), given the true density and a wrong outcome regression on the law's support as weighted rows, returns the truth exactly at every pair, where the untargeted plug-in misses | `::test_the_estimators_targeting_recovers_the_truth_from_a_wrong_outcome_regression` |
| dropping $1/g$, leaving the plug-in untargeted, or reading the regression at the wrong node each miss the truth at every pair | `::test_each_targeting_mutation_misses_the_truth` |
| every pair solves its score, including a 200-pair declaration; round trip; strata; support report; refusals before any learner fits | the fit tests in the same file |
| a pair whose fluctuation does not solve its score raises `DataError`, which a repeated-sampling study records as one lost replicate | `::test_an_unsolved_fluctuation_is_a_data_error_a_study_records` |
| the study route equals the direct fit bit for bit, in five configurations | `tests/unit/test_dose_response_study.py::test_the_route_equals_the_direct_fit` |
| each read method field reaches the engine and moves the fit away from the default; each unread field is refused before any learner | `::test_each_read_field_reaches_the_engine`, `::test_an_unread_field_is_refused_before_any_learner` |
| a treatment-mechanism subset reads `experimental`, and the full set and no declaration read `supported` | `::test_a_treatment_mechanism_subset_reads_experimental` |
| a declared support that cuts one or two pairs' kernels names exactly those pairs; a covering support changes nothing | `::test_a_support_that_cuts_a_kernel_is_refused_before_any_learner`, `::test_a_covering_support_changes_nothing` |

The [registered study](method-evidence/dose-response.md) passes every gate: 27 primary rows
(lowest coverage 0.9207, worst standardized bias 0.030) and 7 property cells, including the band's
joint coverage, exact-efficiency calibration and the targeting gain. It has no comparator.
A [separate study](method-evidence/dose-response-npcausal.md) pairs `DoseResponseTMLE` with
npcausal `ctseff` on a law where the curve is a line, so both smoothed curves equal it. Its
[validation grid row](method-evidence/validation-grid.md) and
[results](method-evidence/dose-response-npcausal.md#results-of-the-declared-run) give the paired
verdicts. One pair is red under `reporting`: at bandwidth 2 and dose 3.5 it is equivalent on its
complete cases and unaccounted under the worst case of the failure accounting, owned by
`X93-failure-worst-case` ([red cells](method-evidence/red-cells.md)).
