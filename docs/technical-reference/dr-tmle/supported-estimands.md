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
`reduced_mechanism_covariate` is unwound. They differ only in the submodel, hence at second
order. So the estimator is not continuous in `K`, and a reader comparing a two-arm
cleverly fit against a two-arm `drtmle` fit should expect agreement in the equations solved
rather than in the iterates.

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
    random_state=0,
)
```

`guard=` says which extra equations to solve, in `drtmle`'s vocabulary and crossed the way that
package crosses it. Both apply by default. It also names the corrections the reported curve
subtracts: one per equation solved, so `guard=("g",)` reports `D = D* − D*_Q` and the score
check's verdict names *that* curve. The other equation's correction is still recomputed and
reported, held to no threshold, because it is what says what the guard did not buy. An empty
guard is a plain `TMLE`, bit for bit.

## Supported, with conditions

| keyword | status |
| --- | --- |
| `delta=` | **randomized binary trials only**, using Díaz & van der Laan (2017)'s missing-outcome construction. The conditions are below. |
| `weights=` | **fixed analysis weights only.** The estimand is the parameter of the tilted law `dP_w = w dP / E[w]`. The transport argument is below. |
| `repeats=` | supported; varies exactly one thing, the **primary split**. Each draw fits its own reductions and runs its own alternation; the report uses the median point and split-adjusted median variance. `result.extra["drtmle"]` describes **draw 0 only**. |
| `reduction="bivariate"` | supported for complete outcomes and discrete treatment. It fits one reduced probability on the two-column `(Qbar-hat(a,W), g-hat(a|W))` design and uses van der Laan's distinct `D_Y`, once per arm as the pinned R implementation does; univariate remains the default because its reduced regressions can converge faster. The cited theorem is binary, so the multi-arm case is an implementation-backed armwise extension rather than a claim about that theorem's literal scope. |
| a random-forest reduction learner | computes, but steps **outside** the cross-fitting argument of [section 3](targeting.md#reduced-regression-cross-fitting), because its fitted class grows with `n`. Not refused; scoped. |
| `g_bounds=` a fixed value | permitted, and it puts the fit outside the asymptotic half of the [bound-inactive scope](targeting.md#the-bound-inactive-scope): the argument needs a bound *sequence* going to zero, which `"auto"` supplies and a fixed bound above `ess inf g_0` does not. |

**What `delta=` accepts.** Set `randomized=True` to estimate the treatment probabilities. You can
instead pass row-aligned known probabilities as `treatment_probabilities=` to `fit`. Three shapes
are accepted: a mapping keyed by treatment level, such as `{"placebo": p0, "active": p1}`; an
`(n, 2)` array in encoded arm order; or an `(n,)` array read as the probability of the arm whose
code is `1`. This surface requires `cross_fit=False`, `repeats=1`, pooled reductions, and no
analysis weights or evaluation companion.

With `guard=()`, the same array configures a **plain TMLE** at the design mechanism. The result is
the ordinary estimator bit for bit, which is what a pure randomization-probability analysis wants.
None of the conditions above then holds, because no extra equation is solved and no theorem is
claimed.

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

## Refused by name

Each because the derivation read here does not cover it, not because the loop would not run. Every
row raises at construction or at `fit`, with a message naming what a derivation would need. The
`weights_estimated=` row is the one exception. The fit raises nowhere. It records that the
caller estimated the weights, and no interval claim here covers that estimation.

| refused | why |
| --- | --- |
| continuous treatment | the reductions and corrections are indexed by treatment mass at a discrete arm; a continuous dose requires density-based equations |
| `reduction="bivariate"` with `delta=` | the supported missing-outcome estimator is Díaz & van der Laan's distinct five-reduction construction, not the complete-outcome one- versus two-dimensional choice. `reduction="univariate"` selects that published missing-data cycle; a bivariate analogue of its five reductions and three correction blocks has not been derived here. |
| `att` / `atc` | a different score equation with no reduced-dimension derivation |
| `interventions=`, `shifts=`, `incremental=`, `msm=` | as above |
| observational treatment with `delta=` | Díaz & van der Laan (2017) derives the construction for randomized trials; the canonical package accepting observational treatment is implementation provenance, not a theorem for that composition |
| missing treatment (`treatment_delta=`) | reserved for a future published construction. Canonical missing-`A` smoke tests do not supply this package's required identification, corrected curve, remainder, and rate conditions |
| `treatment_probabilities=` with `n_bootstrap=`, **whatever `guard=` is** | the array is row-aligned to the data as passed, and a replicate refits on resampled rows it cannot be reindexed to. An n-out-of-n resample passes the length check, so the misalignment would be silent; `randomized=True` estimates the mechanism inside each replicate instead. Unconditional on the guard, because the array is row-aligned however few equations are being solved |
| `treatment_probabilities=` without `delta=` | it replaces the treatment learner outright, and nothing read here states a complete-data construction that reads a known design mechanism differently from a fitted one |
| `intermediate=` | the reduced equations carry no controlled-intermediate factor |
| `targeting_scheme="fold"` | each fold would need its own reduced regressions and alternation |
| `cv_evaluation=True` | the common-update construction would need the corrected parameter and influence curve derived under fold-wise evaluation |
| composition with `CTMLE` | a reduced regression conditions on `ĝ` *as a covariate*, and C-TMLE's `ĝ` is deliberately not an estimate of `g_0`. C-TMLE also scores its path by the loss of the targeted `Q̄`, so the criterion choosing `ĝ` presupposes that `Q̄` is informative. That is precisely the case this variant insures against. |
| estimated weights (`weights_estimated=`) | **not refused at `fit`; no interval claim covers it.** The ordinary answer is that the interval conditions on the weights, and that answer is an argument about `D*` rather than about `Q_r`, `g_{r,1}` and `g_{r,2}`. `simulated_confounding` refuses the composition before it draws |
| `evaluation=` with `repeats>1`, `targeting="one_step"`, or `target_weights=True` | each by name; the middle one on cost, up to 20,000 adaptive steps |
| `reduced_crossfit="nested"` with `cross_fit=False` or `n_folds < 3` | there is no complement to leave a fold out of; nested leaves two folds out at a time |
