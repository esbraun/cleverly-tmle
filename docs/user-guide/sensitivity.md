# Sensitivity analysis

An estimate is causal only if the adjustment set holds every common cause of the treatment and the
outcome, and no data can check that. A sensitivity analysis asks how strong a confounder that the
data do not record would have to be to change the conclusion. `result.sensitivity` offers four
analyses:

- `omitted_confounding()` gives the range of estimates that a missing confounder of declared
  strength allows. One strength is its link to the outcome (`cf_y`), the other its link to the
  treatment (`cf_d`), each a share between 0 and 1. `benchmark()` measures the strengths of a
  recorded covariate, and `robustness_value()` finds the strength that would move the estimate to 0.
- `evalue()` gives how strongly a missing confounder would need to be linked to both the treatment
  and the outcome, on the risk-ratio scale, to explain the effect away.
- `marginal_bounds()` gives the range of effects when a missing confounder can multiply each unit's
  odds of treatment by at most a factor `gamma`, with no limit on its link to the outcome.
- `simulated_confounding()` plants a confounder of chosen strengths in the data and refits.

```python
all_sensitivity = result.sensitivity.run_all()
print(all_sensitivity.summary())
```

`run_all()` runs the cheap analyses. A simulated surface needs an explicit strength grid, so
`run_all()` never starts it. The other point-treatment methods are `elements()`, `contour()`,
`missingness()` and `tipping_gamma()`. Support and truncation stability live under
`result.diagnostics`.

## How strong would a missing confounder have to be?

`omitted_confounding()` reads one more number from the fit, a term that grows as the weights get
more extreme (set by `nu2_estimator`). The default estimator of it is doubly robust: it stays on
target in large samples if either the outcome model or the treatment model is right, and its
one-sided limits hold for any treatment learner. The plug-in estimator gives the limits for a
declared known mechanism and for a scikit-learn `LogisticRegression`, with no penalty or a fixed L2
penalty, fitted in sample or cross-fitted. The example fits one, with the default `C=1`, and reads
the limits.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_linear_ate

linear_frame, _ = make_linear_ate(1000, seed=1)
linear_study = CausalStudy(
    linear_frame,
    design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")),
)
plugin_fit = linear_study.estimate(
    ATE(),
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    cross_fit=False,
    random_state=1,
)
plugin_bounds = plugin_fit.sensitivity.omitted_confounding(
    cf_y=0.05, cf_d=0.05, nu2_estimator="plugin"
)
print(plugin_bounds.summary())
print(plugin_bounds.ci_lower, plugin_bounds.robustness_value_ci)
```

`plugin_bounds.summary(detail="full")` names, under Settings, the mechanism whose score the standard
error carries. A treatment learner outside the admitted set, such as a random forest, makes
`ci_lower`, `ci_upper`, `robustness_value_ci` raise `CapabilityError` with the roadmap item F26, and
`robustness_value` omits `rva`. The bounds and the rest of `robustness_value` stay available.

:::{dropdown} Details

The plug-in estimate of that term, $\nu^2$, has a bias of order $1/n$, and its constant grows with
$g^{-2}$. The bound subtracts a second-order estimate of that bias. A fixed L2 penalty adds to the
bias, and the same correction covers it. The correction needs a smooth mechanism. The standard error
of the limits is the influence variance of the uncorrected curve, because the subtraction does not
change the curve. :::

The bound withholds its limits when a cap on extreme weights binds, when its bias correction is
unstable, or when the corrected term is not positive. More rows, or a `g_bounds` that the fitted
treatment model does not approach, can clear the refusal. The
[learner rule](../technical-reference/validation-methods.md#plug-in-limits-and-the-learner-rule)
lists every key.

To ask what a missing confounder as strong as a recorded covariate could do, hand the benchmark to
the bound. `benchmark()` refits without the named covariates, and `benchmark=` reads its two
strengths:

```python
calibration = plugin_fit.sensitivity.benchmark(covariates=("W1",), random_state=1)
as_strong_as_w1 = plugin_fit.sensitivity.omitted_confounding(
    benchmark=calibration, nu2_estimator="plugin"
)
print(as_strong_as_w1.summary())
```

The range then says how far a confounder as strong as `W1` could move the estimate, under the
bound's assumptions. A record from another fit or another estimand is refused, and so are explicit
`cf_y=` or `cf_d=` beside it. A treatment-side strength of 1 or more calibrates no finite bound and
raises `ValueError`. [The benchmark](../technical-reference/validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours)
defines the two strengths.

## E-value

`result.sensitivity.evalue()` gives how strongly a missing confounder would need to be linked to
both the treatment and the outcome, on the risk-ratio scale, to explain the effect away, and how
strongly to move the interval limit nearer to no effect there.

On a `TimeToEvent` fit, name two reported risks at one horizon, numerator first:
`result.sensitivity.evalue(("risk_regimen[a1 @ t=2]", "risk_regimen[a0 @ t=2]"))`. Some
configurations have no coverage study for their confidence limit yet. The [E-value
reference](../technical-reference/validation-methods.md#e-value) lists them. Each prints its
confidence-limit E-value as descriptive, and `EValue.limit_descriptive` is true.

`result.sensitivity.evalue().plot()` draws the trade-off behind the E-value. Each curve holds the
weakest pairs of links, one between a hidden confounder and the treatment and one between it and
the outcome, that could explain the estimate, or its interval limit, away. Each link is a risk
ratio. The E-value is the point where the two links are equal. The
[plotting reference](../api/plotting.md) lists the figure's options.

## Marginal sensitivity bounds

The marginal sensitivity bounds read a grid of `gamma` values. At `gamma=1`, no missing
confounder, the range shrinks to one value close to the fit's own estimate.

```python
odds_bounds = plugin_fit.sensitivity.marginal_bounds(gamma=(1.0, 1.5, 2.0))
print(odds_bounds.summary())
```

Read the `lower` and `upper` columns at each `gamma`, and the interval for the range beside them.
`odds_bounds.breakdown` is the smallest grid `gamma` whose range contains 0, or `None` when no grid
value reaches it. The bounds cover two-arm and multi-arm `TMLE`, `DRTMLE` and `CTMLE` fits with a
continuous or binary outcome, and longitudinal fits refuse them. Without `estimand=` they bound the
parameter the fit reported, so a multi-arm fit needs `estimand=` to name its contrast. A [simulation
study](../technical-reference/method-evidence/marginal-bound-limit-coverage.md) backs the interval
for the ATE of a two-arm, five-fold cross-fitted `TMLE` with the study's `GLM` models and at least
8,000 rows. Every other fit, such as this in-sample one, prints the interval as descriptive.
[Marginal sensitivity model
bounds](../technical-reference/validation-methods.md#marginal-sensitivity-model-bounds) gives the
estimator, its conditions and each refusal.

The bounds fit extra quantile and tail regressions on the fit's folds, without refitting the
estimator, so a combined report runs them only with `include_refits=True`.

## Plant a confounder and refit

`simulated_confounding()` adds a hidden common cause of chosen strengths to the data, refits the
whole analysis at each pair of strengths, and reports how far the estimate moves. It draws one
hidden value per row (the latent variable). On a yes/no treatment of strength `k`, it switches the
treatment of the share `k` of rows with the highest hidden values, and on a continuous outcome it
subtracts the outcome strength times the hidden value ([the stress
surface](../technical-reference/validation-methods.md#simulated-common-cause-stress-surface)).

```python
from cleverly.sensitivity import ConfounderStrengthGrid

grid = ConfounderStrengthGrid(
    treatment=(0.0, 0.05, 0.10),
    outcome=(0.0, 0.25, 0.50),
)
surface = plugin_fit.sensitivity.simulated_confounding(
    grid=grid,
    benchmark_covariates=("W1", "W2"),
    random_state=1,
)
movements = surface.to_frame()
calibration = surface.calibration_frame()
```

Pass the seed of the original fit, or pass no `random_state` at all. The surface reproduces the
folds of that fit under the seed of that fit alone. A different seed gives every non-anchor cell new
folds, and a fit that declared no seed has the same effect. The anchor keeps the original folds, so
movement near the anchor can carry a fold artifact.

On a binary fit, the treatment flip is non-differential misclassification. The association it
induces between the latent variable and the treatment depends on the treated fraction. That
association is zero on a balanced design. The technical reference
[measures it](../technical-reference/validation-methods.md#simulated-common-cause-stress-surface).

Each cell reports its own realised association in `induced_treatment_association`. The frame
carries the same value in a column of that name, and `summary()` prints it. On a binary fit, check
the column before you read a treatment movement as confounding. A value near the anchor can
reflect misclassification alone.

On a continuous fit, the latent variable changes the dose by construction, so the association
grows with the treatment strength. A confounding path also needs the latent variable to enter the
outcome, and only a nonzero outcome strength puts it there. A cell in the zero outcome-strength
column therefore carries no confounding path, whatever its association. Its movement reports the
dose perturbation alone. The technical reference states the same
[reading rule](../technical-reference/validation-methods.md#simulated-common-cause-stress-surface).

The zero treatment-strength column carries no confounding path either. Its movement reports the
outcome perturbation alone. The Gaussian outcome law subtracts a level from every row, and an
`ate_policy[...]` contrast removes most of that level. An `ey_policy[...]` policy mean keeps it, so
read the zero treatment-strength column of a policy-mean surface as an artifact of the outcome
law.

The operation refits the complete estimator at each nonzero strength pair. The zero cell equals the
original estimate exactly. Additive surfaces record `movement_scale="estimate_difference"`.

Read the output as a qualitative stress surface. It is not a corrected estimate, bound, p-value,
confidence interval, robustness value, or pass/fail result. Calibration reports model-dependent
source conventions for numeric covariates. It does not change the declared grid.

Inspect `result.sensitivity.capability("simulated_confounding")` before a combined run. Its
`available`, `status`, and `reason` fields read the same fit-wide check as direct execution. The
check stops at the first boundary your fit meets and names it in the same words a direct call
raises, and each refusal comes before calibration, the hidden-value draw, or a refit. Take a fit
whose backdoor provenance is not a registered explicit adjustment set. Its row reports
`available=False` and the reason `"simulated_confounding needs registered explicit-adjustment
backdoor provenance"`. The [stress
surface](../technical-reference/validation-methods.md#simulated-common-cause-stress-surface) lists
every boundary.

## Go deeper

### The combined report's sensitivity rows

The omitted-confounding row says "at the default strengths" when you omit both strength arguments.
It names the omitted strength when you supply only one. The effective arguments retain both numeric
values.

The E-value row records its source alias, such as `estimand="ate"`, for replay. A derived ratio can
have a different output alias, such as `rr`. For a default odds-ratio source, the row keeps
`estimand=None` because an explicit alias chooses the approximation. The returned E-value records
that source in `source_estimand`.

### Missing outcomes: tilt and tipping point

The `missingness()` tilt and the `tipping_gamma()` search perturb the arm means and their
contrasts and ratios, `ey_obs`, PAR and PAF, the regime parameters, and the MSM coefficients. Away
from $\gamma = 0$ on an ordinary `TMLE` fit, the arm means, their contrasts and ratios, `ey_obs`,
PAR and PAF are re-estimated at each $\gamma$, each with its own confidence interval, and
`tipping_gamma(use_ci=True)` finds where an interval limit reaches the null. A
[simulation study](../technical-reference/method-evidence/missingness-tilt.md) backs those
intervals for a binary outcome with two arms and at least 1,000 rows, fitted with the
correctly specified logistic regressions that the study declares. The summary and the figure call
the intervals of every other fit descriptive, a continuous outcome included.

A regime parameter, an MSM coefficient, and every parameter of a `DRTMLE` or `CTMLE` fit get no
interval off $\gamma = 0$. Their spread there is the main analysis's standard error placed around
the moved estimate, reported under the `plugin_` column names, and it is a diagnostic: the moved
estimate has its own sampling error, which can be much larger. For those,
`tipping_gamma(use_ci=True)` refuses and the default point search answers. [The targeted
tilt](../technical-reference/validation-methods.md#targeted-missingness-tilt) gives the reason for
each.

Two missing-outcome fits get neither method. A shift fit gets neither because whether the tilted
parameter is still the shift parameter has not been derived. An incremental fit gets neither
because its targeting also solves a score in the treatment mechanism, and no derivation covers
the tilt of that fixed point.

A per-arm outcome-adaptive `CTMLE` fit regresses its observation factor on the outcome regression
only. Both methods read the full-covariate model `nuisance.tilt_missingness` on that fit, and no
targeting step reads it.

Both capability rows read `unavailable` before any call, with the sentence that the call raises. A
fit with no missing outcome reads `not_applicable`. The combined assessment reports the same row
instead of substituting an armwise analysis.

A completed `tipping_gamma` search can return `None` when its interval contains no tipping point.

### A continuous dose

A continuous-dose fit uses signed treatment strengths. Pass one exact modified-policy mean or
contrast alias because a continuous result has no bare `ate` parameter.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, ModifiedTreatmentPolicyEffect, PointTreatment
from cleverly.datasets import make_shift_dose
from cleverly.interventions import Shift
from cleverly.sensitivity import ConfounderStrengthGrid

shift_frame, _ = make_shift_dose(n=200, seed=21)
shift_study = CausalStudy(
    shift_frame,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2", "W3"),
        treatment_kind="continuous",
    ),
)
shift_result = shift_study.estimate(
    ModifiedTreatmentPolicyEffect(
        policies=(
            Shift(0.0, cap=3.0, name="natural course"),
            Shift(0.5, cap=3.0, name="up half"),
        )
    ),
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    cross_fit=False,
    learner_folds=2,
    random_state=21,
    simultaneous=False,
)
shift_alias = "ate_policy[up half vs natural course]"
shift_surface = shift_result.sensitivity.simulated_confounding(
    estimand=shift_alias,
    grid=ConfounderStrengthGrid(
        treatment=(-0.25, 0.0, 0.25),
        outcome=(0.0, 0.25, 0.50),
    ),
    benchmark_covariates=("W1", "W2"),
    random_state=21,
)
```

A `ModifiedTreatmentPolicy` fit reports policy means instead. Select one mean with an alias such
as `ey_policy[up half]`. The surface reports that mean and its signed displacement in each cell.

Select a policy whose `delta` is nonzero. A zero-delta shift assigns every unit its own dose, so
its mean is $E[Y]$ and no common cause can move it through the treatment. The surface refuses that
mean by name. It still accepts an `ate_policy[...]` contrast that uses the zero-delta policy as its
reference, because the contrast keeps its treatment dependence.

The continuous treatment law is $A'=A+k_AU$. It keeps the declared modified treatment policies
fixed during each ordinary-TMLE refit. The outcome laws and common-randomness contract stay the
same as the binary surface.

### Other fits

The table gives what to pass on other kinds of fit and how to read a cell.
[The stress surface](../technical-reference/validation-methods.md#simulated-common-cause-stress-surface)
states the law each fit uses and every boundary.

| fit | what to pass | how to read a cell |
| --- | --- | --- |
| `CounterfactualMean()`, which reports both arms | one exact alias through `estimand=`, such as `"ey[1]"`. A `CounterfactualMean(treatment=1)` fit needs only the grid | the arm the question sets stays fixed while the observed treatment and outcome move |
| a regime mean or contrast, or a working-model (MSM) coefficient, of a yes/no treatment, or an MSM coefficient of a continuous dose, fitted with ordinary `TMLE` | the alias, when the fit reports more than one | the declared policy at each row, or the link, grid and coefficient names, stay fixed. A log or logit MSM reports coefficient differences, not ratios |
| an incremental target of a yes/no treatment, which multiplies each row's odds of treatment, fitted with ordinary `TMLE` | the alias | each cell rebuilds the intervention from its refitted treatment model, so movement includes the change in intervention probabilities. A multiplier-one mean is refused; a contrast against it runs |
| `ATT()` or `ATC()` without baseline strata | the grid only | the treated or untreated group changes with the switched treatments. Read `surface.conditioning_arm` and each cell's `target_population_fraction`; the battery warns when the group keeps under half its share |
| `RiskRatio` or `OddsRatio` of a yes/no treatment, without baseline strata | the grid only | the displacement is the refitted log ratio minus the original log ratio (`movement_scale="log_ratio"`); exponentiate it for the ratio of the two ratios |
| `PopulationAttributableRisk` or `PopulationAttributableFraction` of a yes/no treatment, fitted with ordinary `TMLE` | the grid only | each cell recomputes both means with the reference arm fixed. A PAF displacement is a difference of two fractions |
| a fit with baseline `strata=` | the complete alias from `result.estimates`, which `key.stratum` in `result.parameter_keys` selects | the stratum stays fixed and every row is refitted. Calibration describes the full fitted population |
| three or more treatment arms | a contrast alias such as `ate[high vs low]`, or `arm_pair=("high", "low")` | the two arms of the pair swap rows; other arms keep theirs |
| a clustered fit | `latent_level="row"`, `"cluster"`, or `"mixed"` with `latent_rho` | a treatment constant within every cluster accepts only `"cluster"` when a cell moves the treatment |
| a missing outcome or a missing treatment | `observation_reads=`, the covariates every missingness model reads, for a cell that moves the treatment | only recorded values move, and every missingness indicator stays. The report states the declaration, which the surface cannot verify |
| a controlled direct effect | `ConfounderStrengthGrid(intermediate=(0.0, 0.2))` | the treatment is held and the intermediate switches; a nonzero treatment strength is refused |
| a continuous outcome fitted with `q_bounds` | `bounded_outcome_law="logit"` (the default) or `"normal_score"` | moved outcomes stay inside the bounds, so the outcome strength is not in outcome standard deviations; compare `induced_outcome_association` across fits |
| observation weights | nothing | each weight stays on its row, and the replay does not regenerate estimated weights. `surface.target_measure` and `surface.weight_report` describe the weights |
| `LTMLE`, `SDR`, `SequentialGComputation`, `SequentialIPW` or `ParametricGFormula` on an uncensored end-of-study panel with no covariate measured after the first node | `estimand=`, such as `"ate_regimen[never vs always]"`. `treatment={"A2": (0.0, 0.1)}` moves one node | one hidden value per person is shared by every node and the outcome. Each cell reports one association per node, and a later node's includes the path through the earlier ones |
