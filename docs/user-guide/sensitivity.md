# Sensitivity analysis

A confounder is a variable that affects both the treatment and the outcome. An estimate is causal
only if the covariates you adjusted for (the adjustment set) include every confounder. No data can
check that: the fit sees only the columns you have, so a confounder you did not record cannot show
up in any check. A sensitivity analysis asks the next question instead: how strong would a missing
confounder have to be to change the conclusion? A robust result is one that only a strong missing
confounder could overturn. It does not show that no such confounder exists; whether one that strong
is plausible is your judgement about the subject.

A sensitivity analysis differs from a [refutation](refutation.md) test, which checks that the analysis recovers a known answer on changed
data and says nothing about a variable the data do not hold.

`result.sensitivity` offers four analyses, and each measures strength in its own way:

- `omitted_confounding()` gives the range of estimates that a missing confounder of declared
  strength allows, and `robustness_value()` the strength that would move the estimate to 0.
  `benchmark()` measures the strength of a covariate you did record, as a yardstick.
- `evalue()` gives how strongly a missing confounder would need to be linked to both the treatment
  and the outcome to explain the effect away, as risk ratios: the chance of something in one group
  divided by the chance in another.
- `marginal_bounds()` gives the range of effects when a missing confounder can multiply each row's
  odds of treatment by at most a factor `gamma`, with no limit on its link to the outcome.
- `simulated_confounding()` plants a confounder of chosen strengths in the data and refits.

The robustness value and the E-value each answer "how strong would a missing confounder have to be
to explain the effect away?" with one number, on different scales: shares of explained variation
and risk ratios. `gamma` is a third scale, and the package converts none into another. For a
manager, the clearest sentence compares a missing confounder with a covariate you recorded ([Compare
with a covariate you recorded](#compare-with-a-covariate-you-recorded)); report the robustness value
and the E-value beside it.

## Run the quick analyses

The examples on this page use one small simulated fit of an average treatment effect, the change in
the mean outcome `Y` if every row got the treatment `A` compared with none. The fit uses two models:
the outcome model predicts `Y` from `A` and the covariates, and the treatment model predicts each
row's chance of getting `A`. It fits both on every row (`cross_fit=False`). The package's default,
cross-fitting, predicts each row from models trained without it, and on a continuous outcome it
needs a declared range of values, which this simulated `Y` lacks. If you have your own fitted
result, skip the fit and use your result, with its own settings, in place of `linear_fit`.

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
effect = linear_fit["ate"]
print(f"effect {effect.psi:.3f}, 95% CI [{effect.ci[0]:.3f}, {effect.ci[1]:.3f}]")
quick = linear_fit.sensitivity.run_all()
print(quick.summary())
```

```text
effect 1.463, 95% CI [1.338, 1.588]
```

`run_all()` runs every analysis that needs no new model fit, and lists the rest under `on request`.
Shortened, the report reads:

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

The `unmeasured confounding` and `E-value` rows answer the main question. A missing confounder would
need strengths of about half (0.509) on both sides, or a risk ratio of 3.4 with both the treatment
and the outcome, to erase an effect of 1.46. The benchmark and the planted confounder need an
argument you choose, and the worst-case bounds (`marginal_bounds()`) fit extra models, so
`run_all()` runs them only with `include_refits=True` or when you call them.

## How far could a missing confounder move the estimate?

`omitted_confounding()` describes a missing confounder by two strengths, each a share between 0
and 1. The outcome-side strength (`cf_y`) is the share of the outcome's variation, left after
adjusting for your covariates, that the confounder would explain. The treatment-side strength
(`cf_d`) says how much knowing the confounder would sharpen the prediction of who got the treatment.

The estimate weights each row by one over its predicted chance of its own treatment, so that the
treated and the untreated rows each stand in for everyone. Sharper predictions push chances toward 0
or 1 and make the weights larger, and larger weights leave more room for bias. At a `cf_d` of 0.05
the average squared weight would be about 5% larger (1 / (1 - 0.05) times). The bound reports the
range of estimates such a confounder allows, when its two links line up in the way that moves the
estimate most (the worst alignment).

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
1.57. The `95% one-sided limits` add sampling uncertainty: a 95% lower limit for the low end and a
95% upper limit for the high end. The range excludes 0, so such a confounder cannot erase the
effect. There is no standard strength to declare: try the strengths of a covariate you recorded, as
the next section does.

The robustness value needs no choice. It is the strength, equal on both sides, that would move the
estimate to 0: here 0.51, or 0.48 for the 95% limit nearest 0 (`robustness_value()` returns it
alone). A value near 0 means a weak confounder could erase the
effect; a value near 1 means only one that explains almost all the remaining variation on both sides
could.

The bound also assumes that the treatment model gets each row's chance of treatment right, at least
approximately, and more closely as the data grow. If it does not, the average squared weight comes
out too small and the bounds too narrow. No check proves this, but the treatment-model and balance
checks on [Diagnostics](assessment.md) look for signs of a poor treatment model.

### Compare with a covariate you recorded

A strength of 0.05 or 0.5 means little until you compare it with something you know. The benchmark
refits without a recorded covariate, measures how much leaving it out hurts both models, and hands
those two strengths to the bound:

```python
w1_strength = linear_fit.sensitivity.benchmark(covariates=("W1",), random_state=1)
as_strong_as_w1 = linear_fit.sensitivity.omitted_confounding(benchmark=w1_strength)
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
predicting the treatment (0.016). Under the bound's assumptions, the sentence for a manager is: "A
missing factor as strong as `W1` could at worst lower the estimated effect from 1.46 to 1.21, or to
1.10 allowing for sampling uncertainty, so it would not erase the effect." Name your own covariates
in `covariates=`: ones you understand well and believe matter for both the treatment and the
outcome, so that you can judge whether a missing confounder could be as strong.

## The E-value: link strength that explains the effect away

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

The `risk ratio` column is the effect itself on the risk-ratio scale. On a continuous outcome, as
here, the summary divides the mean difference of 1.46 by the outcome's standard deviation and turns
that into a risk ratio of about 2.0 with a standard formula for a roughly normal outcome. Read it as
roughly how much more likely the treated would be to land in the upper part of the outcome; it is an
approximation.

An E-value of 1 means no confounding at all is needed, so larger means more robust. There is no
fixed cutoff, and the package does not compute your covariates' risk ratios. Ask
whether any factor you know of could make both the treatment and the outcome about 3.4 times as
likely. The two links can trade off, one below the E-value if the other is above it.

Some setups have no simulation study for their interval-limit E-value yet; a note then calls it
descriptive, and `EValue.limit_descriptive` is true. The [E-value
reference](../technical-reference/validation-methods.md#e-value) lists those setups.
`evalue.plot()` draws the trade-off: each curve holds the weakest pairs of links that could explain
the estimate, or its interval limit, away, and the E-value is the point where the two links are
equal.

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

Read the `lower` and `upper` columns at each `gamma`. At `gamma=1`, no missing confounder, the range
shrinks to one value, the bound's own estimate, which is close to the fit's. Here even a confounder
that doubles or halves the odds of treatment leaves the range above 0. `odds_bounds.breakdown` is
the smallest grid `gamma` whose range contains 0, or `None` when no grid value reaches it; try
larger values to find it. The warning means no simulation study has checked how often this interval
holds the true range for this kind of fit. You may report it, but say so, for example "an interval
of 0.78 to 2.14, which no simulation study has checked for this kind of analysis". [Marginal
sensitivity model
bounds](../technical-reference/validation-methods.md#marginal-sensitivity-model-bounds) gives the
fits it covers, its conditions and each refusal.

## Plant a confounder and refit

`simulated_confounding()` adds a made-up confounder of chosen strengths to the data, refits the
whole analysis at each pair of strengths, and reports how far the estimate moves. It draws one
hidden value per row from a standard normal distribution. The treatment strength is a share of rows:
on a yes/no treatment of strength 0.05, it switches the treatment of the 5% of rows with the highest
hidden values. The outcome strength is in the outcome's own units: on a continuous outcome it
subtracts the strength times the hidden value, so 0.25 lowers the outcome by 0.25 for each standard
deviation of the hidden value. A yes/no outcome is switched in the same tail instead ([how the
confounder is
planted](../technical-reference/validation-methods.md#simulated-common-cause-stress-surface)).

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
covariate_strengths = surface.calibration_frame()
```

```text
Movement
cell   treatment strength   outcome strength   estimate   movement
------------------------------------------------------------------------------
1                       0                  0      1.463     +0.000
...
4                    0.05                  0      1.374     -0.089
5                    0.05               0.25      1.402     -0.061
...
7                     0.1                  0      1.244     -0.218
...
9                     0.1                0.5      1.297     -0.166
==============================================================================
Associations
cell   treatment
------------------------------------------------------------------------------
...
9         -0.055
==============================================================================
Observed covariates on the same scale
covariate   role        strength
------------------------------------------------------------------------------
W1          treatment      0.261
W1          outcome        1.097
W2          treatment      0.155
W2          outcome        0.456
```

Each cell's `movement` is its estimate minus the original one; in `surface.to_frame()` it is the
`displacement` column. The last table puts the covariates you named in `benchmark_covariates` on the
grid's scales: on the treatment axis, the share of rows whose predicted treatment changes when that
covariate is set to its average, and on the outcome axis, how much the outcome changes per standard
deviation of it. Here `W1` sits at 0.26 on the treatment axis and 1.10 on the outcome axis, beyond
the largest values of this grid, so to ask whether a confounder as strong as `W1` would matter,
widen the grid to reach it, such as `treatment=(0.0, 0.15, 0.3)` and `outcome=(0.0, 0.6, 1.2)`; each
cell is one more refit. Read the surface as a stress test: it shows how far the estimate moves, and
gives no corrected estimate, bound, interval or pass or fail.

Check the `Associations` table before you read a movement along the treatment axis as confounding.
It gives, for each cell, the correlation between the hidden value and the switched treatment. On a
yes/no treatment that association depends on the treated share, and it is zero when half the rows
are treated. Here it is about -0.05, near zero, so switching treatments mostly mislabelled rows
instead of tying the treatment to the hidden value.

Cells 4 and 7 show this: with no outcome strength they carry no confounding path at all, yet move
the estimate by up to -0.22. On this fit, then, the treatment-axis movements measure the damage of
mislabelled treatments rather than the bias a real confounder would cause. Each cell reports its own
association in `induced_treatment_association`. Run the planted confounder when you want to watch
the whole analysis rerun under a confounder as strong as a recorded covariate. Pass the seed of the
original fit as `random_state`, and see [More on the planted
confounder](#more-on-the-planted-confounder) for why and for the reading rules on a continuous
treatment.

## Go deeper

### The weight term of the bound

The bound needs one more number from the fit, a term that grows as the weights get more extreme (the
average squared weight, $\nu^2$, set by `nu2_estimator`). The default estimator of it is doubly
robust in a narrow sense: its error is second order in the treatment model's error, so small errors
in the fitted treatment model move it only slightly, and it gives one-sided limits for any treatment
learner (the model you pass as `treatment_learner`). It reads only the treatment model, so the
condition under the bound still applies: a treatment model that is wrong even in large samples makes
the term too small and the bounds too narrow ([omitted-variable
bounds](../technical-reference/validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours)).
The plug-in estimator below is the alternative.

A benchmark record from another fit or another estimand is refused, and so are explicit `cf_y=` or
`cf_d=` beside it. A treatment-side strength of 1 or more calibrates no finite bound and raises
`ValueError`. [The
benchmark](../technical-reference/validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours)
defines the two strengths.

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

### Which fits the worst-case bounds cover

A [simulation study](../technical-reference/method-evidence/marginal-bound-limit-coverage.md) backs
the interval for the range for the average effect of a two-arm `TMLE` fit cross-fitted on five
folds, with the study's regression models and at least 8,000 rows; an arm is one treatment group.
The bounds cover fits of one treatment given once with `TMLE` (the default method), `DRTMLE` (a
variant whose interval aims to stay valid when one of the two models is wrong) and `CTMLE`
(collaborative TMLE, which builds the treatment model step by step), with two or more arms and a
continuous or yes/no outcome. Fits of a treatment over time refuse them. Without `estimand=` they
bound the estimate the fit reported, so a fit with more than two arms needs `estimand=` to name its
contrast. At `gamma=1` the range is the bound's own doubly robust estimate, one that stays on target
if either model is right.

### Other methods and the combined report

`run_all()` here and `result.assess()` on [Diagnostics](assessment.md) report the same
sensitivity rows. The other point-treatment methods are `elements()`, the inputs of the bound,
`contour()`, the bound over a grid of strengths, and `missingness()` and `tipping_gamma()`, below.
The overlap check and the weight-cap curve live under `result.diagnostics`.

The unmeasured-confounding row says "at the default strengths" when you omit both strength
arguments. It names the omitted strength when you supply only one. The arguments the row keeps hold
both numbers.

The E-value row records the estimate it read, such as `estimand="ate"`, so the call can be run
again. When the E-value comes from a ratio the package derived, the row may record that ratio's
name, such as `rr`. When it comes from an odds ratio by default, the row keeps `estimand=None`,
because naming the odds ratio would switch the E-value to an approximate conversion. Either way, the
returned E-value names the estimate it read in `source_estimand`. On a `TimeToEvent` fit, name two
reported risks at one time point, numerator first: `result.sensitivity.evalue(("risk_regimen[a1 @
t=2]", "risk_regimen[a0 @ t=2]"))`.

### When outcomes are missing

When some outcomes are missing, `missingness()` asks how far the result moves if the missing
outcomes differ from the observed ones, by shifting their predicted log-odds by `gamma` (a tilt),
and `tipping_gamma()` searches for the `gamma` at which the conclusion changes. They perturb the
mean outcome of each treatment group (arm) and their contrasts and ratios, `ey_obs` (the mean
outcome under the treatments people actually got), the population attributable risk and fraction
(PAR and PAF), the means under treatment rules (regime parameters), and the coefficients of a
working model across treatment levels (MSM, a trend line across the levels). Away from $\gamma = 0$
on an ordinary `TMLE` fit, the arm means, their contrasts and ratios, `ey_obs`, PAR and PAF are
re-estimated at each $\gamma$, each with its own confidence interval, and
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
its correction step also adjusts the treatment model, and no derivation covers the tilt of that
joint adjustment.

One collaborative fit needs care. With `CTMLE(strategy="oat")`, its model of who has an outcome
recorded reads only the outcome model's predictions, not the covariates. On that fit both methods
read a separate model of who has an outcome recorded, one that uses every covariate
(`nuisance.tilt_missingness`) and that no correction step reads.

Where a fit gets neither method, its rows in `result.sensitivity.capability(...)` read `unavailable`
before any call, with the sentence that the call raises. A fit with no missing outcome reads
`not_applicable`. The combined report shows the same row instead of substituting an armwise
analysis. A completed `tipping_gamma` search can return `None` when its interval contains no tipping
point.

### More on the planted confounder

Pass the seed of the original fit, or pass no `random_state` at all: on a cross-fitted fit, the
surface reproduces the fit's split of the rows into folds only under its seed. A different seed
gives every cell but the zero cell (the anchor) new folds, and a fit that declared no seed has the
same effect. The anchor keeps the original folds, so movement near the anchor can carry an artifact
of the new folds. The call refits the complete estimator at each nonzero strength pair, and the zero
cell equals the original estimate exactly. Additive surfaces record
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
declared as an explicit list of covariates is one example. Its row reports `available=False`, with a
reason that states this requirement in the package's own terms (`"simulated_confounding needs
registered explicit-adjustment backdoor provenance"`). The [stress
surface](../technical-reference/validation-methods.md#simulated-common-cause-stress-surface) lists
every limit.

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

The planted confounder moves a continuous treatment by $A'=A+k_AU$. It keeps the declared modified
treatment policies fixed during each ordinary-TMLE refit. The outcome changes and the shared random
draws stay the same as on the yes/no surface.

### The planted confounder on other fits

[The stress
surface](../technical-reference/validation-methods.md#simulated-common-cause-stress-surface) states
how the planted confounder moves each kind of fit, and every limit.

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
| a continuous outcome fitted with a declared range (`q_bounds`) | `bounded_outcome_law="logit"` (the default) or `"normal_score"` | moved outcomes stay inside the bounds, so the outcome strength is not in the outcome's own units; compare `induced_outcome_association` across fits |
| observation weights | nothing | each weight stays on its row, and the replay does not regenerate estimated weights. `surface.target_measure` and `surface.weight_report` describe the weights |
| a treatment over time, fitted with `LTMLE`, `SDR`, `SequentialGComputation`, `SequentialIPW` or `ParametricGFormula`, on a panel with no covariate measured after the first node: an uncensored end-of-study outcome, or a survival outcome with the treatment held from baseline ([survival](#sensitivity-analysis-of-a-time-to-event-fit)) | `estimand=`, such as `"ate_regimen[never vs always]"`. `treatment={"A2": (0.0, 0.1)}` moves one node | one hidden value per person is shared by every node and the outcome. Each cell reports one association per node, and a later node's includes the path through the earlier ones |

### Sensitivity analysis of a fit over time

`result.sensitivity.simulated_confounding()` covers panels with no covariate measured after
the first node, as [the planted confounder on other fits](#the-planted-confounder-on-other-fits) shows. An
end-of-study outcome must be uncensored. A censored `TimeToEvent` fit is covered too, as the
[next section](#sensitivity-analysis-of-a-time-to-event-fit) describes.

`result.sensitivity.evalue()` reads the risk ratio of two static plans at one horizon of a
`TimeToEvent` fit. The other sensitivity analyses, such as the omitted-variable bound, have no
longitudinal estimator and report `unavailable`. Tan (2025) derives population bounds for binary,
static longitudinal strategies, but no sample estimator for them. The
[roadmap](../roadmap.md#f16-longitudinal-sensitivity-bound-estimation) records that boundary.

### Sensitivity analysis of a time-to-event fit

Two sensitivity analyses run on a `TimeToEvent` fit. Each asks how a confounder that the data do
not record could change the answer.

| call | what it does | where it runs |
| --- | --- | --- |
| `result.sensitivity.evalue((numerator, denominator))` | reads the risk ratio of two reported risks at one time and says how strongly a missing confounder would need to be tied to both treatment and event to explain it away | two static plans at one horizon. Name the risks, numerator first, such as `("risk_regimen[arm1 @ t=4]", "risk_regimen[arm0 @ t=4]")`. On a fit with exact times, name the two arms' risks of one event at one target time, such as `("cif_regimen[treated, relapse @ t=2]", "cif_regimen[control, relapse @ t=2]")` |
| `result.sensitivity.simulated_confounding(...)` | plants a made-up confounder of chosen strengths and refits. It is a stress test, not a bound | a survival or competing-risk fit whose treatment is one baseline decision held over follow-up, censored or not, and a fit with exact times |

`simulated_confounding` returns a surface: one estimate for each pair of confounder strengths in
the grid you pass. To read it, find the weakest confounder at which the estimate crosses zero, and judge
whether an unrecorded factor that strong is plausible in your setting. Each cell also reports
which share of the replayed rows came from the fitted models; a high share means the replay leans
on the models more than on the data.

:::{dropdown} Details: how the replay works
On a censored fit, the simulated confounder raises or lowers each person's chance of the event at
every time point, and the follow-up is replayed. Censoring is drawn again where the treatment
changed, and a person whose event the confounder removes stays in follow-up on fitted models. A
grid cell with both strengths at zero gives the original estimate.
[The stress surface](../technical-reference/validation-methods.md#simulated-common-cause-stress-surface)
gives the replay and its checks.

On a fit with exact times, the confounder scales each person's event rate of every event type
up or down, and new event and censoring times are drawn from models the analysis fits for this
purpose. Censoring is again drawn afresh only where the treatment changed, and both strengths at
zero give back the recorded times and the original estimate. Each grid cell other than that one
refits the whole analysis, so a 3 by 3 grid costs 8 refits: seconds at a few hundred people, and
minutes for a large cross-fitted fit. The surface prints numbers, not a pass or fail.
[Continuous-time survival replay](../technical-reference/validation-methods.md#simulated-common-cause-stress-surface)
gives the law and its checks.
:::

Both analyses, and the checks that generate new outcomes (`dummy_outcome`, `simulated_outcome`,
`plasmode`), refuse a fit with `eliminate=` or a separable effect; the E-value reads only two ordinary arms. The
placebo, subset and random-common-cause checks of [refutation](refutation.md#refutation-of-a-fit-over-time) run
on these fits.

### The analyses in the tutorials

| tutorial | what it runs |
| --- | --- |
| [One treatment, one outcome](../examples/point-treatment-tmle.ipynb) | the missing-confounder bound with a recorded covariate for comparison, the E-value, the worst-case bounds and the planted confounder |
| [Smoking and twins' birth weight](../examples/twins-causal-inference.ipynb) | the bound, the E-value and the planted confounder for the effect on the treated |
| [Missing survey answers](../examples/survey-nonresponse.ipynb), [Gain over current practice](../examples/population-impact.ipynb) | the bound and the E-value, and the analyses for missing outcomes |
| [An interview of a sample of patients](../examples/sampling-weights.ipynb) | the bound on a fit with sampling weights, with recorded covariates for comparison |
| [A strong predictor of treatment](../examples/collaborative-tmle.ipynb) | the bound and the planted confounder on a collaborative TMLE fit |
| [Three offer policies](../examples/interventions.ipynb) | the planted confounder on policy effects |
| [More contacts and more service time](../examples/schedules-and-intensity.ipynb) | the planted confounder on policy effects, moving only the recorded treatment, so it tests recording errors rather than confounding |
