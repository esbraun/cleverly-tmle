# Sensitivity analysis

A confounder is a variable that affects both the treatment and the outcome. An estimate is causal
only if the covariates you adjusted for (the adjustment set) include every confounder. No data can
check that: the fit sees only the columns you have, so a confounder you did not record cannot show
up in any check. A sensitivity analysis asks the next question instead: how strong would a missing
confounder have to be to change the conclusion? A robust result is one that only a strong missing
confounder could overturn. It does not show that no such confounder exists; whether one that strong
is plausible is your judgement about the subject.

`result.sensitivity` offers four analyses, and each measures strength in its own way:

- `omitted_confounding()` gives the range of estimates that a missing confounder of declared
  strength allows, and `robustness_value()` the strength that would move the estimate to 0.
  `benchmark()` measures the strength of a covariate you did record, as a yardstick.
- `evalue()` gives how strongly a missing confounder would need to be linked to both the treatment
  and the outcome, as risk ratios, to explain the effect away.
- `marginal_bounds()` gives the range of effects when a missing confounder can multiply each row's
  odds of treatment by at most a factor `gamma`, with no limit on its link to the outcome.
- `simulated_confounding()` plants a confounder of chosen strengths in the data and refits.

The first two answer "how strong would a missing confounder have to be to explain the effect away?"
directly: the robustness value and the E-value are each one such strength.

## Run the quick analyses

The examples on this page use one small simulated fit of an average treatment effect, the change in
the mean outcome `Y` if every row got the treatment `A` compared with none. If you have your own
fitted result, skip this block and use your result in place of `linear_fit`.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_linear_ate

linear_frame, _ = make_linear_ate(1000, seed=1)
linear_study = CausalStudy(
    linear_frame,
    design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")),
)
linear_fit = linear_study.estimate(
    ATE(),
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    cross_fit=False,
    random_state=1,
)
quick = linear_fit.sensitivity.run_all()
print(quick.summary())
```

The fit estimates the effect at 1.46. `run_all()` runs every analysis that reads what the fit
stored, and lists the rest under `on request`. Shortened, the report reads:

```text
check                        result           finding
------------------------------------------------------------------------------
unmeasured confounding       done             Left-out variables with both
                                              strengths at 0.509 could, at
                                              worst, move the estimate to 0;
                                              weaker ones, with both strengths
                                              below 0.509, could not.
covariate benchmark          on request       needs an explicit covariates
                                              argument, which a combined
                                              report has no basis to choose
simulated confounder         on request       needs an explicit grid argument,
                                              which a combined report has no
                                              basis to choose
E-value                      done             An unmeasured confounder would
                                              need a risk ratio of 3.41 with
                                              both the treatment and the
                                              outcome, beyond the measured
                                              covariates, to explain away the
                                              estimate; 3.17 to move the
                                              interval to the null.
worst-case bounds            on request       not run by default because it
                                              refits the models
```

The two `done` rows answer the main question; the sections below read each one. The benchmark and
the planted confounder need an argument you choose, and the worst-case bounds (`marginal_bounds()`)
fit extra models, so `run_all()` runs them only with `include_refits=True` or when you call them.
The [combined report](#sensitivity-rows-in-the-combined-report) and the other methods are under Go
deeper.

## How strong would a missing confounder have to be?

`omitted_confounding()` describes a missing confounder by two strengths, each a share between 0 and
1. The outcome-side strength (`cf_y`) is the share of the outcome's variation, left after adjusting
for your covariates, that the confounder would explain. The treatment-side strength (`cf_d`) is how
much the confounder would add to predicting who got the treatment. It is measured as the share of
the average squared weight that only it would add, where each row's weight is one over its predicted
chance of its own treatment. The bound reports the range of estimates such a confounder allows, when
its two links line up in the way that moves the estimate most (the worst alignment).

```python
bounds = linear_fit.sensitivity.omitted_confounding(cf_y=0.05, cf_d=0.05)
print(bounds.summary())
```

```text
      estimate            range   95% one-sided limits
------------------------------------------------------------------------------
ate      1.463   [1.359, 1.566]         [1.254, 1.671]
==============================================================================
Robustness value
            robustness value
------------------------------------------------------------------------------
estimate               0.509
95% limit              0.480
```

A confounder with both strengths at 0.05 could move the estimate anywhere in the `range`, 1.36 to
1.57, and the `95% one-sided limits` add sampling uncertainty to each end. The range excludes 0, so
such a confounder cannot erase the effect. The robustness value is the strength, equal on both
sides, that would move the estimate to 0: here 0.51, or 0.48 for the 95% limit nearest 0
(`robustness_value()` returns it alone). A value near 0 means a weak confounder could erase the
effect; a value near 1 means only one that explains almost all the remaining variation on both sides
could.

The bound needs one more number from the fit, a term that grows as the weights get more extreme (set
by `nu2_estimator`). The default estimator of it is doubly robust in a narrow sense: small errors in
the fitted treatment model move it only slightly, so its one-sided limits hold for any treatment
learner. It still needs a treatment model that is right in large samples; a wrong one makes the
bounds too narrow ([omitted-variable
bounds](../technical-reference/validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours)).
[The plug-in estimator](#the-plug-in-estimator-of-the-weight-term) is the alternative.

### Compare with a covariate you recorded

A strength of 0.05 or 0.5 means little until you compare it with something you know. The benchmark
refits without a recorded covariate, measures how much leaving it out hurts both models, and hands
those two strengths to the bound:

```python
calibration = linear_fit.sensitivity.benchmark(covariates=("W1",), random_state=1)
as_strong_as_w1 = linear_fit.sensitivity.omitted_confounding(benchmark=calibration)
print(as_strong_as_w1.summary())
```

```text
Question:   how far left-out            Outcome-side strength:    0.995 (cf_y)
            confounders could move ate  Treatment-side strength:  0.016 (cf_d)
...
      estimate            range   95% one-sided limits
------------------------------------------------------------------------------
ate      1.463   [1.205, 1.721]         [1.099, 1.827]
```

`W1` explains almost all the outcome variation the others leave (0.995) but adds little to
predicting the treatment (0.016). A confounder as strong as `W1` could move the estimate at most to
between 1.21 and 1.72, or 1.10 to 1.83 with sampling uncertainty, under the bound's assumptions.
Those are the declared strengths, the worst alignment, and a treatment model that is right in large
samples. That range excludes 0, which is the sentence to give a manager. Name your own covariates in
`covariates=`; the covariate that predicts the outcome best is the usual yardstick.

A benchmark record from another fit or another estimand is refused, and so are explicit `cf_y=` or
`cf_d=` beside it. A treatment-side strength of 1 or more calibrates no finite bound and raises
`ValueError`. [The
benchmark](../technical-reference/validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours)
defines the two strengths.

## E-value

`evalue()` gives how strongly a missing confounder would need to be linked to both the treatment and
the outcome, as risk ratios beyond the measured covariates, to explain the effect away. It also
gives a second value for the interval: how strong it would need to be to move the interval limit
nearest no effect all the way to no effect.

```python
evalue = linear_fit.sensitivity.evalue()
print(evalue.summary())
```

```text
                             risk ratio    E-value
------------------------------------------------------------------------------
estimate                          1.998       3.41
95% limit nearest the null        1.884       3.17
```

Here a confounder would need a risk ratio of 3.4 with both the treatment and the outcome, beyond the
covariates, to explain the effect away, and 3.2 to pull the interval to no effect. An E-value of 1
means no confounding at all is needed, so larger means more robust; there is no fixed cutoff. The
two links can trade off, one below the E-value if the other is above it. On a continuous outcome, as
here, the summary converts the mean difference to a risk ratio, and each step of that conversion is
an approximation.

Some setups have no simulation study for their interval-limit E-value yet. Each prints that E-value
as descriptive: no study has checked it for this kind of analysis, so report it as a description of
this sample. `EValue.limit_descriptive` is then true, and the [E-value
reference](../technical-reference/validation-methods.md#e-value) lists those setups. `evalue.plot()`
draws the trade-off: each curve holds the weakest pairs of links that could explain the estimate, or
its interval limit, away, and the E-value is the point where the two links are equal.

## Worst-case bounds on treatment odds

`marginal_bounds()` asks a different question. Suppose a missing confounder could change each row's
odds of treatment by at most a factor `gamma`, in either direction, with no limit on its link to the
outcome: what range of effects is then possible?

```python
odds_bounds = linear_fit.sensitivity.marginal_bounds(gamma=(1.0, 1.5, 2.0))
print(odds_bounds.summary())
```

```text
gamma      lower      upper   95% interval for the range
------------------------------------------------------------------------------
1          1.463      1.463               [1.338, 1.588]
1.5        1.133      1.790               [1.005, 1.916]
2          0.907      2.013               [0.776, 2.143]
...
Warnings:
[1] No validation study licenses the interval for the range for this kind of
    fit, so read it as descriptive.
```

Read the `lower` and `upper` columns at each `gamma`, and the interval for the range beside them. At
`gamma=1`, no missing confounder, the range shrinks to one value: the bound's own doubly robust
estimate, which is close to the fit's. Here even a confounder that doubles or halves the odds of
treatment leaves the range above 0. `odds_bounds.breakdown` is the smallest grid `gamma` whose range
contains 0, or `None` when no grid value reaches it.

The warning says that no simulation study backs this interval for this fit, so read it as a
description of this sample. A [simulation
study](../technical-reference/method-evidence/marginal-bound-limit-coverage.md) backs the interval
for the average effect of a two-arm `TMLE` fit cross-fitted on five folds, with the study's
regression models and at least 8,000 rows. The bounds cover fits with `TMLE` (the default method),
`DRTMLE` and `CTMLE` of one treatment given once, with two or more arms and a continuous or yes/no
outcome, and fits of a treatment over time refuse them. Without `estimand=` they bound the estimate
the fit reported, so a fit with more than two arms needs `estimand=` to name its contrast. [Marginal
sensitivity model
bounds](../technical-reference/validation-methods.md#marginal-sensitivity-model-bounds) gives the
estimator, its conditions and each refusal.

## Plant a confounder and refit

`simulated_confounding()` adds a made-up confounder of chosen strengths to the data, refits the
whole analysis at each pair of strengths, and reports how far the estimate moves. It draws one
hidden value per row. On a yes/no treatment of strength `k`, it switches the treatment of the share
`k` of rows with the highest hidden values. It subtracts the outcome strength times the hidden value
from a continuous outcome, or switches a yes/no outcome in the same tail ([the stress
surface](../technical-reference/validation-methods.md#simulated-common-cause-stress-surface)).

```python
from cleverly.sensitivity import ConfounderStrengthGrid

grid = ConfounderStrengthGrid(
    treatment=(0.0, 0.05, 0.10),
    outcome=(0.0, 0.25, 0.50),
)
surface = linear_fit.sensitivity.simulated_confounding(
    grid=grid,
    benchmark_covariates=("W1", "W2"),
    random_state=1,
)
print(surface.summary())
movements = surface.to_frame()
calibration = surface.calibration_frame()
```

```text
cell   treatment strength   outcome strength   estimate   movement
------------------------------------------------------------------------------
1                       0                  0      1.463     +0.000
5                    0.05               0.25      1.402     -0.061
9                     0.1                0.5      1.297     -0.166
...
cell   treatment
------------------------------------------------------------------------------
9         -0.055
...
covariate   role        strength
------------------------------------------------------------------------------
W1          treatment      0.261
W1          outcome        1.097
```

Each cell's `movement` is its estimate minus the original one; in `surface.to_frame()` it is the
`displacement` column. The last table puts the covariates you named in `benchmark_covariates` on the
same scale, so choose grid values around those strengths. Here `W1` sits at 0.26 on the treatment
axis and 1.10 on the outcome axis, beyond the largest values of this grid. Read the surface as a
stress test. It shows how far the estimate moves, and gives no corrected estimate, bound, interval
or pass or fail.

Check the association table before you read a movement along the treatment axis as confounding. On a
yes/no treatment, switching treatments also mislabels them, and the association the switch induces
between the hidden value and the treatment depends on the treated share; it is zero when half the
rows are treated. An association near zero, as here at about -0.05, means the treatment axis moved
the estimate by mislabelling rather than by confounding. Each cell reports its own association in
`induced_treatment_association`, and the [technical reference measures
it](../technical-reference/validation-methods.md#simulated-common-cause-stress-surface).

Pass the seed of the original fit, or pass no `random_state` at all: the surface reproduces the
folds of that fit only under its seed. [More on the planted
confounder](#more-on-the-planted-confounder) gives the reading rules for a continuous treatment and
the checks the call runs first.

## Go deeper

### The plug-in estimator of the weight term

The plug-in estimator (`nu2_estimator="plugin"`) gives the bound's one-sided limits for a declared
known treatment model and for a scikit-learn `LogisticRegression`, with no penalty or a fixed L2
penalty, fitted in sample or cross-fitted. The example fit uses one, with the default `C=1`:

```python
plugin_bounds = linear_fit.sensitivity.omitted_confounding(
    cf_y=0.05, cf_d=0.05, nu2_estimator="plugin"
)
print(plugin_bounds.summary())
print(plugin_bounds.ci_lower, plugin_bounds.robustness_value_ci)
```

`ci_lower` is the 95% one-sided limit of the lower end of the range, and `robustness_value_ci` the
robustness value of that limit. `plugin_bounds.summary(detail="full")` names, under Settings, the
treatment model whose score the standard error carries. A treatment learner outside that list, such
as a random forest, makes `ci_lower`, `ci_upper` and `robustness_value_ci` raise `CapabilityError`
with the roadmap item F26, and `robustness_value` omits `rva`, the robustness value of the limit.
The bounds and the rest of `robustness_value` stay available.

:::{dropdown} Details

The plug-in estimate of that term, $\nu^2$, has a bias of order $1/n$, and its constant grows with
$g^{-2}$. The bound subtracts a second-order estimate of that bias. A fixed L2 penalty adds to the
bias, and the same correction covers it. The correction needs a smooth mechanism. The standard error
of the limits is the influence variance of the uncorrected curve, because the subtraction does not
change the curve.

:::

The bound withholds its limits when a cap on extreme weights binds or nearly binds, when its bias
correction is unstable, or when the corrected term is not positive. More rows, or a cap on the
predicted chances (`g_bounds`) that the fitted treatment model does not approach, can clear the
refusal. The [learner
rule](../technical-reference/validation-methods.md#plug-in-limits-and-the-learner-rule) lists every
key.

### Sensitivity rows in the combined report

`run_all()` here and `result.assess()` on [Check a fitted result](assessment.md) report the same
sensitivity rows. The other point-treatment methods are `elements()`, the inputs of the bound,
`contour()`, the bound over a grid of strengths, and `missingness()` and `tipping_gamma()`, below.
The overlap check and the weight-cap curve live under `result.diagnostics`.

The unmeasured-confounding row says "at the default strengths" when you omit both strength
arguments. It names the omitted strength when you supply only one. The arguments the row keeps hold
both numbers.

The E-value row records the estimate it read, such as `estimand="ate"`, so the call can be run
again. A derived ratio can carry another name, such as `rr`. For a default odds-ratio source, the
row keeps `estimand=None`, because an explicit name chooses the approximation. The returned E-value
records that source in `source_estimand`. On a `TimeToEvent` fit, name two reported risks at one
time point, numerator first: `result.sensitivity.evalue(("risk_regimen[a1 @ t=2]", "risk_regimen[a0
@ t=2]"))`.

### When outcomes are missing

When some outcomes are missing, `missingness()` asks how far the result moves if the missing
outcomes differ from the observed ones, by shifting their predicted log-odds by `gamma` (a tilt),
and `tipping_gamma()` searches for the `gamma` at which the conclusion changes. They perturb the arm
means and their contrasts and ratios, `ey_obs`, the population attributable risk and fraction (PAR
and PAF), the regime parameters, and the coefficients of a working model across treatment levels
(MSM). Away from $\gamma = 0$ on an ordinary `TMLE` fit, the arm means, their contrasts and ratios,
`ey_obs`, PAR and PAF are re-estimated at each $\gamma$, each with its own confidence interval, and
`tipping_gamma(use_ci=True)` finds where an interval limit reaches the null.

A [simulation study](../technical-reference/method-evidence/missingness-tilt.md) backs those
intervals for a yes/no outcome with two arms and at least 1,000 rows, fitted with the correctly
specified logistic regressions that the study declares. The summary and the figure call the
intervals of every other fit descriptive, a continuous outcome included.

A regime parameter, an MSM coefficient, and every parameter of a `DRTMLE` or `CTMLE` fit get no
interval off $\gamma = 0$. Their spread there is the main analysis's standard error placed around
the moved estimate, reported under the `plugin_` column names, and it is a diagnostic: the moved
estimate has its own sampling error, which can be much larger. For those,
`tipping_gamma(use_ci=True)` refuses and the default point search answers. [The targeted
tilt](../technical-reference/validation-methods.md#targeted-missingness-tilt) gives the reason for
each.

Two missing-outcome fits get neither method. A shift fit gets neither because whether the tilted
parameter is still the shift parameter has not been derived. An incremental fit gets neither because
its correction step also solves a score in the treatment model, and no derivation covers the tilt of
that fixed point.

A per-arm outcome-adaptive `CTMLE` fit regresses its model of who has an outcome recorded on the
outcome regression only. Both methods read the full-covariate model `nuisance.tilt_missingness` on
that fit, and no correction step reads it.

Both capability rows read `unavailable` before any call, with the sentence that the call raises. A
fit with no missing outcome reads `not_applicable`. The combined report shows the same row instead
of substituting an armwise analysis. A completed `tipping_gamma` search can return `None` when its
interval contains no tipping point.

### More on the planted confounder

A different seed gives every cell but the zero cell (the anchor) new folds, and a fit that declared
no seed has the same effect. The anchor keeps the original folds, so movement near the anchor can
carry an artifact of the new folds. The call refits the complete estimator at each nonzero strength
pair, and the zero cell equals the original estimate exactly. Additive surfaces record
`movement_scale="estimate_difference"`.

On a continuous treatment, the hidden value changes the dose by construction, so the association
grows with the treatment strength. A confounding path also needs the hidden value to enter the
outcome, and only a nonzero outcome strength puts it there. A cell in the zero outcome-strength
column therefore carries no confounding path, whatever its association, and its movement reports the
dose change alone. The technical reference states the same [reading
rule](../technical-reference/validation-methods.md#simulated-common-cause-stress-surface).

The zero treatment-strength column carries no confounding path either; its movement reports the
outcome change alone. On a continuous outcome the change subtracts a level from every row, and a
policy contrast (`ate_policy[...]`) removes most of that level. A policy mean (`ey_policy[...]`)
keeps it, so read the zero treatment-strength column of a policy-mean surface as an artifact of the
simulated outcome.

The calibration table describes each covariate on the surface's scale, by a rule that depends on the
model behind it for a numeric covariate. It does not change the grid you declared.

Inspect `result.sensitivity.capability("simulated_confounding")` before a combined run. Its
`available`, `status`, and `reason` fields read the same fit-wide check as a direct call. The check
stops at the first limit your fit meets and names it in the same words a direct call raises, and
each refusal comes before the hidden-value draw or any refit. A fit whose adjustment set was not
declared as an explicit list of covariates is one example. Its row reports `available=False` and the
reason `"simulated_confounding needs registered explicit-adjustment backdoor provenance"`. The
[stress surface](../technical-reference/validation-methods.md#simulated-common-cause-stress-surface)
lists every limit.

### A continuous dose

A continuous-dose fit uses signed treatment strengths. Pass one exact name of a modified-policy mean
or contrast, because a continuous result has no bare `ate` estimate.

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

A `ModifiedTreatmentPolicy` fit reports policy means instead. Select one mean with a name such as
`ey_policy[up half]`. The surface reports that mean and its signed displacement in each cell.

Select a policy whose `delta` is nonzero. A zero-delta shift assigns every unit its own dose, so its
mean is $E[Y]$ and no common cause can move it through the treatment. The surface refuses that mean
by name. It still accepts an `ate_policy[...]` contrast that uses the zero-delta policy as its
reference, because the contrast keeps its treatment dependence.

The continuous treatment law is $A'=A+k_AU$. It keeps the declared modified treatment policies fixed
during each ordinary-TMLE refit. The outcome laws and the shared random draws stay the same as on
the yes/no surface.

### The planted confounder on other fits

[The stress
surface](../technical-reference/validation-methods.md#simulated-common-cause-stress-surface) states
the law each fit uses and every limit.

| fit | what to pass | how to read a cell |
| --- | --- | --- |
| `CounterfactualMean()`, which reports both arms | one exact name through `estimand=`, such as `"ey[1]"`. A `CounterfactualMean(treatment=1)` fit needs only the grid | the arm the question sets stays fixed while the observed treatment and outcome move |
| a regime mean or contrast, or a working-model (MSM) coefficient, of a yes/no treatment, or an MSM coefficient of a continuous dose, fitted with ordinary `TMLE` | the name, when the fit reports more than one | the declared policy at each row, or the link, grid and coefficient names, stay fixed. A log or logit MSM reports coefficient differences, not ratios |
| an incremental target of a yes/no treatment, which multiplies each row's odds of treatment, fitted with ordinary `TMLE` | the name | each cell rebuilds the intervention from its refitted treatment model, so movement includes the change in intervention probabilities. A multiplier-one mean is refused; a contrast against it runs |
| the effect among the treated or the untreated (`ATT()` or `ATC()`), without baseline strata | the grid only | the treated or untreated group changes with the switched treatments. Read `surface.conditioning_arm` and each cell's `target_population_fraction`; the battery warns when the group keeps under half its share. A cell moves through confounding, mislabelling and the change of group; an association near zero rules out only the first, as `summary()` notes |
| `RiskRatio` or `OddsRatio` of a yes/no treatment, without baseline strata | the grid only | the displacement is the refitted log ratio minus the original log ratio (`movement_scale="log_ratio"`); exponentiate it for the ratio of the two ratios |
| `PopulationAttributableRisk` or `PopulationAttributableFraction` of a yes/no treatment, fitted with ordinary `TMLE` | the grid only | each cell recomputes both means with the reference arm fixed. A PAF displacement is a difference of two fractions |
| a fit with baseline `strata=` | the complete name from `result.estimates`, which `key.stratum` in `result.parameter_keys` selects | the stratum stays fixed and every row is refitted. Calibration describes the full fitted population |
| three or more treatment arms | a contrast name such as `ate[high vs low]`, or `arm_pair=("high", "low")` | the two arms of the pair swap rows; other arms keep theirs |
| a clustered fit | `latent_level="row"`, `"cluster"`, or `"mixed"` with `latent_rho` | a treatment constant within every cluster accepts only `"cluster"` when a cell moves the treatment |
| a missing outcome or a missing treatment | `observation_reads=`, the covariates every missingness model reads, for a cell that moves the treatment | only recorded values move, and every missingness indicator stays. The report states the declaration, which the surface cannot verify |
| a controlled direct effect | `ConfounderStrengthGrid(intermediate=(0.0, 0.2))` | the treatment is held and the intermediate switches; a nonzero treatment strength is refused |
| a continuous outcome fitted with a declared range (`q_bounds`) | `bounded_outcome_law="logit"` (the default) or `"normal_score"` | moved outcomes stay inside the bounds, so the outcome strength is not in outcome standard deviations; compare `induced_outcome_association` across fits |
| observation weights | nothing | each weight stays on its row, and the replay does not regenerate estimated weights. `surface.target_measure` and `surface.weight_report` describe the weights |
| a treatment over time, fitted with `LTMLE`, `SDR`, `SequentialGComputation`, `SequentialIPW` or `ParametricGFormula`, on a panel with no covariate measured after the first node: an uncensored end-of-study outcome, or a survival outcome with the treatment held from baseline ([survival](survival.md#check-sensitivity-to-a-missing-confounder)) | `estimand=`, such as `"ate_regimen[never vs always]"`. `treatment={"A2": (0.0, 0.1)}` moves one node | one hidden value per person is shared by every node and the outcome. Each cell reports one association per node, and a later node's includes the path through the earlier ones |
