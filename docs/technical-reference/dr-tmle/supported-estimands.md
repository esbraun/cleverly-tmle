# Supported estimands


A **discrete point treatment** and the `mean` group: every treatment-specific mean and the
reference-arm contrasts requested through `ey` and `ate`. For multiple levels, the
univariate implementation follows R `drtmle`'s armwise construction (`R/fluctuate.R`, `fluctuateG`):
reduced regressions and the two extra equations are fitted once per arm, and equation (9)
independently fluctuates each one-vs-rest mechanism margin with response `1(A = a)`, offset
`logit(g_a)`, covariate `Qr_a / g_a`, one scalar per arm.

The targeted margins are **not renormalised**, and the reason is that what this estimator
owes is a set of solved score equations rather than a likelihood: projecting the `K` tilted
margins back onto the simplex would move every one of them off the root just found. They
are not inert, and the estimate does see them: the next round's equation (8) divides by `g_a*`,
so the targeted `Qbar*` and hence `psi` do depend on margins that sum to something other than
one (measured: `0.9975` on the three-armed fixture). That is what `drtmle` does too, and it is
licensed by the score equations rather than by the mechanism still being a conditional
distribution. The initial categorical mechanism *is* compatible and sums to one, using
cleverly's existing multiclass learner path; `diagnostics.support()` reports how far the
targeted rows depart from it.

**Two arms keep their own route, which is not the armwise one.** `drtmle` fluctuates both
margins independently even at `K = 2`; cleverly instead tilts `g_1` alone along a
two-column covariate, so `g_0* = 1 - g_1*` holds exactly. Both solve the *same two* score
equations. Column 0's is `P_n[Qr_0/g_0* {1(A=a_0) - g_0*}] = 0` once the sign convention in
`reduced_mechanism_covariate` is unwound. They differ only in the submodel. No read source
states that this difference is second order, so that statement is an unsourced derivation.
Because the two-arm route differs from the armwise one, the estimator is not continuous in `K`,
and a reader comparing a two-arm
cleverly fit against a two-arm `drtmle` fit should expect agreement in the equations solved
rather than in the iterates.

RM19 measured the two routes on 2,000 fresh draws of the binary `treatment_correct` law at
`n = 3000`. There, the treatment mechanism is correct and the outcome regression is not. The
transcription of the R loop took each of four named choices from R or from this package.
Averaged over the other three choices, the two-arm tilt raised the mean `ate` by `0.000833` to
`0.001718`, at the level 0.998. The whole difference from R reads `no increment at the declared
resolution`.

`sqrt(n)` times that difference falls from `0.0789` at `n = 1500` to `0.0053` at
`n = 6000`. Three sizes do not prove a rate, and that measurement does not isolate the tilt.
[What the localization design found](https://github.com/esbraun/cleverly-tmle/blob/4ce96cda2bda93ba9233026977e3ff63ea3e0003/docs/roadmap.md#what-the-localization-design-found) gives each
interval.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from cleverly import ATE, CausalStudy, DRTMLEMethod, PointTreatment
from cleverly.datasets import make_nonlinear_ate

frame, truth = make_nonlinear_ate(n=1000, seed=0)
study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2", "W3", "W4"),
    ),
)
res = study.estimate(
    ATE(),
    method=DRTMLEMethod(guard=("Q", "g")),
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    cross_fit=False,
    random_state=0,
)
```

`make_nonlinear_ate` has a Gaussian outcome, whose support is the whole real line, so this fit
turns cross-fitting off. A cross-fitted fit of a continuous outcome is refused unless `q_bounds`
declares the support. `DRTMLE` inherits that refusal from `TMLE`. The
[fold and outcome-scale rules](../cv-tmle.md#fold-and-outcome-scale-rules) give the message.

`guard=` says which extra equations to solve, in `drtmle`'s vocabulary and crossed the way that
package crosses it. Both apply by default. It also names the corrections the reported curve
subtracts: one per equation solved, so `guard=("g",)` reports `D = D* − D*_Q` and the score
check's verdict names *that* curve. The other equation's correction is still recomputed and
reported, held to no threshold, because it is what says what the guard did not buy. An empty
guard is a plain `TMLE`, bit for bit.

## Supported, with conditions

| keyword | status |
| --- | --- |
| `delta=`, randomized | `randomized=True` or `treatment_probabilities=` selects Díaz & van der Laan (2017)'s missing-outcome construction, at any number of arms. Above two arms, the fit applies the construction to each arm indicator and stacks the arm estimators. The conditions are below. |
| `delta=`, observational | without `randomized=` or `treatment_probabilities=`, a non-empty `guard` selects the [composite indicator](theorem.md#observational-missing-data-the-composite-indicator), at any number of arms |
| `treatment_delta=` | a declared missing treatment, with or without `delta=`, selects the composite indicator at every `guard`. `guard=()` is the composite TMLE. The arm means and their contrasts (`ey`, `ate`, `rr`, `or`) at every `guard`, and regime means and arm-indexed `msm=` coefficients on `TMLE`; the table below gives the compositions |
| `weights=` | **fixed analysis weights only.** The estimand is the parameter of the tilted law `dP_w = w dP / E[w]`. The transport argument is below. |
| `repeats=` | supported; varies exactly one thing, the **primary split**. Each draw fits its own reductions and runs its own alternation; the report uses the median point and split-adjusted median variance. `result.extra["drtmle"]` describes **draw 0 only**. |
| `reduction="bivariate"` | supported for complete outcomes and discrete treatment. It fits one reduced probability on the two-column `(Qbar-hat(a,W), g-hat(a|W))` design and uses van der Laan's distinct `D_Y`, once per arm as the pinned R implementation does; univariate remains the default because its reduced regressions can converge faster. The cited theorem is binary, so the multi-arm case is an implementation-backed armwise extension rather than a claim about that theorem's literal scope. |
| a random-forest reduction learner | computes, but steps **outside** the cross-fitting argument of [section 3](targeting.md#reduced-regression-cross-fitting), because its fitted class grows with `n`. Not refused; scoped. |
| `g_bounds=` a fixed value | permitted, and it puts the fit outside the asymptotic half of the [bound-inactive scope](targeting.md#the-bound-inactive-scope): the argument needs a bound *sequence* going to zero, which `"auto"` supplies and a fixed bound above `ess inf g_0` does not. |
| `q_bounds=` | **required** with `cross_fit=True` on a continuous outcome, and equal to the outcome's known support. Refused on a binary outcome, which already sits on the unit interval. |
| `stratify_folds=` | `"none"` only, which is the default. The other two policies are refused whenever the fit draws a split. |

**What the composite indicator accepts.** The composite construction takes every composition
that both of its parents take. Each row has evidence in `tests/unit/test_composite_missing_data.py`.

| composition | composite TMLE (`TMLE`, `guard=()`) | composite DR-TMLE (non-empty `guard`) |
| --- | --- | --- |
| `guard=` | `()` | `("Q", "g")`, `("Q",)`, `("g",)` |
| `reduction=` | not applicable | `"univariate"`, `"bivariate"` |
| `weights=` (fixed) | yes | yes; `weights_estimated=True` reports `"estimated_weight_plugin"`, as on complete data |
| `id=` | yes | yes |
| `strata=` | yes, in one pooled outcome step | yes; see [baseline strata](#baseline-strata) |
| `n_bootstrap=` | yes; a replicate resamples rows with their `Delta_A` | yes |
| `screen_treatment=True` | yes, screened on the recorded rows | yes |
| `evaluation=`, `reduced_crossfit="nested"` | not applicable | refused: both constructions carry one fold-free treatment mechanism, and the composite has up to three factors |
| `interventions=` (static and known `W`-dependent rules) | yes; the covariate is the arm covariate at the arm the rule assigns | refused, as on complete data |
| `msm=` over the arms | yes; each coefficient projects the identified arm means | refused, as on complete data |
| `cross_fit=True` | refused ([F21](../../roadmap.md#f21-other-missing-outcome-cv-tmle-variants)) | refused |

The fit is in sample, `cross_fit=False`. Each nuisance factor is fitted on the rows it conditions on.
The `missingness_learner` fits both observation factors. R `drtmle` 1.1.2 fits the treatment
observation factor with its one treatment learner, `SL_g`.

**What `delta=` accepts on the randomized route.** Set `randomized=True` to estimate the treatment
probabilities. You can
instead pass row-aligned known probabilities as `treatment_probabilities=` to `fit`. The
probabilities can depend on `W`, as in a stratified randomization. The table gives the accepted
shapes.

| shape | arms | read as |
| --- | --- | --- |
| a mapping keyed by treatment level, such as `{"placebo": p0, "active": p1}` | any | one `(n,)` column per level; every level is named |
| an `(n, K)` array | any | one column per arm, in encoded arm order (the sorted levels) |
| an `(n,)` array | two only | the probability of the arm whose code is `1`; refused above two arms |

The randomized surface requires `cross_fit=False`, `repeats=1`, pooled reductions, and no analysis
weights or evaluation companion.

With `guard=()`, the same array configures a **plain TMLE** at the design mechanism. The result is
the ordinary estimator bit for bit, which is what a pure randomization-probability analysis wants.
The other conditions above then do not apply, because no extra equation is solved and no theorem is
claimed. `cross_fit=False` still applies. `DRTMLE` refuses `delta=` with `cross_fit=True` at every
`guard`, including `guard=()`, before any learner is fitted
(`tests/unit/test_drtmle_missing.py::test_an_unguarded_cross_fitted_missing_outcome_fit_is_refused`).
Ordinary `TMLE` admits a cross-fitted missing-outcome fit only under its
[stacked contract](../point-treatment-tmle.md#stacked-cv-tmle-for-arm-indexed-targets).

**Why `weights=` transports.** The derivation was read at an unweighted law. Transporting it needs
two things at once. The reduced regressions must be `P_w`-conditional expectations, which weighted
loss gives. The mechanism they condition on and divide by must be the `P_w` mechanism, which holds
because they are built from `nuisance.propensity`. `tests/unit/test_remainder_drtmle.py` runs the
whole expansion at two tilted laws, and keeps the wrong transport as a control that fails.

`tests/unit/test_simulated_confounding.py` adds the applied evidence. It refits nonuniform-weight
complete-outcome fits for each binary parameter DR-TMLE can replay: the arm means, the ATE, the
risk ratio, and the odds ratio. The identified effect's method catalog refuses PAR and PAF under
DR-TMLE, so that test replays neither. The test also removes the weight from the reduced
regressions alone, and requires the cell estimate to move. Neither file establishes interval
validity or weighted parity with the canonical implementation.

### Baseline strata

A fit with `strata=` reports each stratum's DR-TMLE beside the marginal one, at every `guard` and
on every route: complete data, the randomized missing-outcome construction, and the composite
indicator. The table gives what changes inside the alternation.

| piece | stratified form |
| --- | --- |
| equations (8), (9) and (10), and the missing-outcome tilts | one block $I(S=s) H / P_n(S=s)$ per stratum for each covariate |
| reduced regressions | fitted inside each stratum, on the stratum's rows of each fold's training complement |
| evaluation companion (`evaluation=`) | each companion row moves by its own stratum's coefficients |

Each stratum's equations then share no row with another stratum's, so a stratum's estimate is
the DR-TMLE of the law given $S=s$. `tests/unit/test_stratified_drtmle_exact.py` checks this
against separate fits on the stratum subsets, on every route. The
[stratified DR-TMLE study](../method-evidence/stratified-dr-tmle.md) measures the marginal and
stratum ATEs against exact truths and pairs the stratum targets with R `drtmle`.

The marginal estimate is the $P_n(S=s)$-weighted mixture of the stratum estimates. Its reductions
condition on $(g_n(W), S)$, which is finer than the $g_n(W)$ of Theorem 1 of Benkeser et al.
(2017).

The theorem's iterated-expectation step needs only that the weights in (9) and (10) are
measurable in the conditioning sigma-field. Its rate conditions hold inside each stratum for a
fixed number of strata. The marginal estimate is therefore valid, and it differs from the
estimate of an unstratified fit. The arm targets of ordinary TMLE follow the same convention.

A stratum with no trainable row of some arm in some fold's training complement refuses after the
fold draw and before any learner.

## Refused by name

Each row below is refused because the derivation read here does not cover it, or because no code
path accepts the input. A refused row raises at construction or at `fit`. Its message names what a
derivation would need. An undeclared missing treatment raises `DataError` from the data container,
and its message names `treatment_delta=`. Every refusal of a declared missing treatment comes from
one gate before any learner, and `tests/unit/test_refusals_before_the_nuisance_fit.py` pins each
one.

The `weights_estimated=` row does not raise. The fit runs and reports its point estimate under the
`"estimated_weight_plugin"` status when its weights vary. Constant weights fit the unweighted
estimator, so a constant column declared estimated keeps its interval. Its `ci`, `pvalue`, and
`std_error` raise `CapabilityError`, and `plugin_std_error` and `plugin_interval` report the
retained diagnostic.

| refused | why |
| --- | --- |
| continuous treatment | the reductions and corrections are indexed by treatment mass at a discrete arm; a continuous dose requires density-based equations |
| `reduction="bivariate"` with `delta=` | the supported missing-outcome estimator is Díaz & van der Laan's distinct five-reduction construction, not the complete-outcome one- versus two-dimensional choice. `reduction="univariate"` selects that published missing-data cycle; a bivariate analogue of its five reductions and three correction blocks has not been derived here. |
| `att` / `atc` | a different score equation with no reduced-dimension derivation |
| `interventions=`, `policies=`, `incremental=`, `msm=` | as above |
| an undeclared missing treatment value | the package does not infer a missing treatment from a missing value. The data container raises `DataError` before any learner and names `treatment_delta=` for a numeric column. A label column meets the null-label check of `src/cleverly/data/causal_data.py` first |
| a missing treatment with `att`, `atc`, `ey_obs`, `par`, `paf`, `incremental=` or `policies=` | when the recording may depend on the treatment, `P(A = a \| W)` is not identified, and each of these reads the treatment law of every row. See [what the data do not identify](theorem.md#observational-missing-data-the-composite-indicator) |
| a missing treatment with `learned_rule=` | the composite conditions identify the value of a fixed rule. No composite derivation for a rule learned from the fit is written, and the value is cross-fitted. See [F27](../../roadmap.md#f27-learned-policy-value-outside-the-published-conditions) |
| a missing treatment with `intermediate=` | no derivation of the composite indicator for a controlled direct effect is written, and the condition that ties `Z` to the recording of the treatment is not stated |
| a missing treatment with `randomized=True` or `treatment_probabilities=` | those select Díaz & van der Laan's construction, which observes the treatment on every row |
| a missing treatment with `CTMLE` | no collaborative score is derived for the composite indicator ([F5](../../roadmap.md#f5-other-refused-c-tmle-and-dr-tmle-compositions)) |
| `treatment_probabilities=` with `n_bootstrap=`, **whatever `guard=` is** | the array is row-aligned to the data as passed, and a replicate refits on resampled rows it cannot be reindexed to. An n-out-of-n resample passes the length check, so the misalignment would be silent; `randomized=True` estimates the mechanism inside each replicate instead. Unconditional on the guard, because the array is row-aligned however few equations are being solved |
| `treatment_probabilities=` without `delta=` | it replaces the treatment learner outright, and nothing read here states a complete-data construction that reads a known design mechanism differently from a fitted one |
| `intermediate=` | the reduced equations carry no controlled-intermediate factor |
| `delta=` or a missing treatment with `cross_fit=True`, **whatever `guard=` is** | the published missing-outcome theorems use Donsker conditions and do not establish a cross-validated extension. With `guard=()` and `delta=` alone, the fit is a plain TMLE on a surface that only ordinary TMLE's audited stacked contract admits |
| `targeting_scheme="fold"` | each fold would need its own reduced regressions and alternation |
| `cv_evaluation=True` | the common-update construction would need the corrected parameter and influence curve derived under fold-wise evaluation |
| composition with `CTMLE` | a reduced regression conditions on `ĝ` *as a covariate*, and C-TMLE's `ĝ` is deliberately not an estimate of `g_0`. C-TMLE also scores its path by the loss of the targeted `Q̄`, so the criterion choosing `ĝ` presupposes that `Q̄` is informative. That is precisely the case this variant insures against. |
| estimated weights (`weights_estimated=`) with a non-empty `guard` | **fits, and reports the `"estimated_weight_plugin"` status.** The ordinary answer is that the interval conditions on the weights. That answer is an argument about `D*`, and not about `Q_r`, `g_{r,1}` and `g_{r,2}`. `guard=()` fits the ordinary TMLE and keeps its interval. [F5](../../roadmap.md#f5-other-refused-c-tmle-and-dr-tmle-compositions) holds the result that would reopen it. `simulated_confounding` refuses the composition before it draws. Constant weights declared estimated fit the unweighted estimator, keep the interval, and `simulated_confounding` answers on them. `CausalData.declares_estimated_weights` is the one reading of the declaration |
| `evaluation=` with `repeats>1`, `targeting="one_step"`, or `target_weights=True` | each by name; the middle one on cost, up to 20,000 adaptive steps |
| `targeting="one_step"` with `reduced_crossfit="nested"` at a non-empty `guard` | on cost, as the `evaluation=` row. The nested designs move by each of up to 20,000 adaptive steps. The fit refuses before any learner, also on an estimator whose settings change after construction |
| `reduced_crossfit="nested"` with `cross_fit=False` or `n_folds < 3` | there is no complement to leave a fold out of; nested leaves two folds out at a time. `cross_fit=True` with fewer than two folds is refused earlier, when the declaration is constructed, so this row's fold clause reaches only `n_folds=2` |
| a continuous outcome with `cross_fit=True` and `q_bounds=None` | the scale would come from every observed outcome, held-out rows included, and no shipped result covers it. See the [fold and outcome-scale rules](../cv-tmle.md#fold-and-outcome-scale-rules) |
| `stratify_folds="treatment"` or `"treatment+outcome"` with `cross_fit=True` | the partition would be a function of the data the fit then conditions on. Refused when the declaration is constructed |
