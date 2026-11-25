# Try to break the result

A refutation test changes the problem so that the right answer is known in advance, refits the whole
analysis, and checks the estimate against that answer. Shuffle the treatment, for example, and the
effect should be about zero. A test can expose a bug, a standard error that is too small because
rows are not independent (repeated visits of one person treated as separate people, say), or a
setting that does not suit the data, which shows up as bias or poor coverage on simulated data like
yours.

A refutation cannot show that no confounder is missing. A confounder is a variable that affects both
the treatment and the outcome, and a missing one is a confounder the data do not record;
[sensitivity analysis](sensitivity.md) asks how strong one would have to be. Passing these tests
says that the analysis returns known answers on data like yours, not that your estimate is causal.

## Run the default checks

The examples on this page use one small simulated fit of an average treatment effect: the change in
the mean outcome `Y` if every row got the treatment `A`, compared with none. The fit names the
outcome, the treatment and the covariates to adjust for (`PointTreatment`). It uses two models: an
outcome model (`outcome_learner`), which predicts `Y` from `A` and the covariates, and a treatment
model (`treatment_learner`), which predicts each row's chance of `A`. With `cross_fit=False` both
are fitted on every row. If you have your own fitted result, skip this block and use your result in
place of `ate_result`.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_linear_ate

gaussian_frame, _ = make_linear_ate(n=200, seed=21)
gaussian_study = CausalStudy(
    gaussian_frame,
    design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")),
)
ate_result = gaussian_study.estimate(
    ATE(),
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    cross_fit=False,
    random_state=21,
)
```

With no arguments, `refute()` runs three tests on the `ate` estimate. (`estimand=` names another
estimate, by its row name in the summary.) `cleverly.validation.refute(result, ...)` takes the same
arguments.

```python
report = ate_result.diagnostics.refute(random_state=21)
print(report.summary())
```

The three default tests are `placebo`, `random_common_cause` and `subset`. Each refits the analysis
20 times, so the call costs about 60 fits: under a second on this small fit, and about 60 times as
long as one fit on a large one.

- `placebo` shuffles the treatment, so the refits should show no effect: 0 for a difference, or 1
  for a ratio, which the test reads as 0 on the log scale. The report tests the mean of the refits
  against that value with a t statistic. It checks the code path, not the confounding in your data.
- `random_common_cause` adds a column of pure noise to the covariates, so the estimate should
  barely move. It prints how far the estimate moved, in standard errors, and no fit gets a pass or
  fail for it today. `seed_spread=True` adds as many refits at fresh seeds without the noise
  column, to show how far a new random split of the rows into folds alone moves the estimate. (A
  cross-fitted fit splits the rows into folds and predicts each row from models trained without its
  fold.)
- `subset` refits on random 70% samples (`subset_fraction=0.7`), so the spread of the refits should
  match what the standard error predicts for samples of that size.

## Read the report

On this fit the summary reads, shortened:

```text
test                     mean         SD   expected           result
------------------------------------------------------------------------------
shuffled treatment    -0.0320     0.2372   about 0            no pass or fail
added noise column     1.2870     0.0094   about 1.286        no pass or fail
random subsamples      1.2713     0.0735   SD about 0.08448   no pass or fail
==============================================================================
Test statistics
test                 statistic                                value   p-value
------------------------------------------------------------------------------
shuffled treatment   t of the refits against 0               -0.603    0.5534
shuffled treatment   mean std. error / refit SD               0.660         -
added noise column   shift, in std. errors                    0.004         -
added noise column   SD of the shifts, in std. errors         0.073         -
random subsamples    spread ratio (refit SD / expected SD)    0.870    0.4760
```

Compare each `mean` (or `SD`, for the subsamples) with the `expected` column, and read the
statistics below it. The shuffled-treatment refits average -0.03, and their t statistic has a
p-value of 0.55, so they agree with no effect. The noise column moved the estimate by 0.004 standard
errors. The subsample refits spread 0.87 times as much as the standard error predicts, with a
p-value of 0.48. A placebo mean far from 0, or a spread ratio far from 1, each with a small p-value,
would point to a problem in the analysis.

The `result` column prints `passed` or `failed` only where a simulation study or an exact reference
backs that test for this kind of fit. Here none does, so each prints `no pass or fail` and you judge
the numbers. Such a test's `passed` is `None`, and so is `report.passed` unless another test failed.
`bool(report)` raises `TypeError` for a report without a pass or fail, so test `report.passed`
against `True` and `False`. The "Verdicts and evidence" part of [refutation
operations](../technical-reference/validation-methods.md#refutation-operations) says which fits get
a pass or fail. Without `random_state`, the call uses the fit's seed, or draws one for an unseeded
fit, and records it in the report under `random_state`; passing that value back repeats the report.

## Check against a known answer

Three tests replace the outcome with one where we know the answer. `dummy_outcome` draws an outcome
unrelated to the treatment and the covariates, so the true effect is zero. `simulated_outcome` draws
an outcome from a process you declare, with an effect you set. `plasmode` also resamples your
covariate rows and draws the treatment from a model fitted to your data, then draws the outcome from
the `simulated_outcome` process, so the test runs on simulated data built to resemble yours.

```python
from cleverly.validation import GaussianAdjustmentOutcome

generated = ate_result.diagnostics.refute(
    tests=("simulated_outcome",),
    simulated_outcome=GaussianAdjustmentOutcome(effect=0.5),
    random_state=21,
)
print(generated.summary())
draws = generated.draws_frame("simulated_outcome")
```

That call draws 400 outcomes, each with a true effect of 0.5, and refits the analysis on each; this
small fit takes a few seconds. Its summary reads, shortened:

```text
test               reading    estimate           95% CI  allowed band  inside
------------------------------------------------------------------------------
simulated outcome  std. bias     0.015  [-0.083, 0.113]   [-0.2, 0.2]     yes
simulated outcome  SE ratio      0.978   [0.911, 1.050]  [0.85, 1.15]     yes
simulated outcome  coverage      0.940   [0.912, 0.961]  [0.88, 0.99]     yes
```

Each test reads up to three numbers against the true value. The bias is the mean error of the 400
estimates, in units of their standard deviation (`std. bias`). The `SE ratio` divides the average
reported standard error by the actual spread of the estimates, and `coverage` is the share of 95%
intervals that held the truth. A number passes when its 95% interval lies inside its band: a bias
within 0.20 standard deviations of the estimates, a ratio between 0.85 and 1.15, and a coverage
between 0.88 and 0.99. Here all three are inside, so the test passed: on outcomes drawn from this
process, the analysis recovers the effect and its interval is honest.

`outcome_bands=` sets other bands, as an `EquivalenceBands`. Each test makes 400 draws by default;
fewer print the numbers with no pass or fail, and fewer than 10 are refused ([refutation
operations](../technical-reference/validation-methods.md#refutation-operations)). Each row in
`draws` records one draw: its seed, the estimate and standard error, the kind of process that drew
it (`family`), or the refit failure. The result holds only for the process you declared, which is
not your real outcome.

### Which outcome to declare

Declare the process that matches your outcome. The test refuses a process of another kind, such as
the default Gaussian process on a yes/no outcome. A declared range (`q_bounds`) is the known range
of a continuous outcome, which a cross-fitted fit of a continuous outcome with TMLE, the default
method, must have.

| your outcome | `simulated_outcome` and `plasmode` | `dummy_outcome` |
| --- | --- | --- |
| continuous, fitted in sample (`cross_fit=False`) with no declared range (`q_bounds`) | `GaussianAdjustmentOutcome` (the default) | `GaussianIndependentOutcome` (the default) |
| continuous with a declared range (`q_bounds`) | `BoundedAdjustmentOutcome` | `BoundedAdjustmentOutcome(effect=0, adjustment_scale=0)` |
| yes/no | `BernoulliAdjustmentOutcome` | `BernoulliAdjustmentOutcome(effect=0, adjustment_scale=0)` |
| an event over time | `LogisticHazardOutcome` | `LogisticHazardOutcome(effect=0, adjustment_scale=0)` |
| competing events over time | `MultinomialHazardOutcome` | the same with every effect 0 and `adjustment_scale=0` |

A fit that declares `q_bounds` takes only the bounded process, which draws inside those bounds with
its mean strictly inside them, and a fit without them refuses the bounded process.

These tests keep your covariate rows and redraw only the outcome. The standard-error ratio and the
coverage therefore print only when the quantity you estimate takes the same value at every row of
the generated data; otherwise the test reads the bias alone. A difference of two arms does so under
the Gaussian process and under `link="identity"`. Under the default logit link of
`BernoulliAdjustmentOutcome` or `BoundedAdjustmentOutcome`, a nonzero effect varies from row to row,
so a generated yes/no outcome with the default link reads the bias only. For all three readings on a
yes/no outcome, declare `link="identity"`, as [all three readings on a yes-or-no
outcome](#all-three-readings-on-a-yes-or-no-outcome) shows, or run `plasmode`, which reads all three
for every target on a fit that reports an interval. The "Generated point outcomes" part of
[refutation operations](../technical-reference/validation-methods.md#refutation-operations) gives
the rule for each target, and [Which fits take a generated
outcome](#which-fits-take-a-generated-outcome) the fits.

## Check for hidden bias with a control outcome

A negative control outcome is an outcome that the treatment cannot affect but that shares the
confounding of your real outcome, such as a health measure recorded before the program started. Pass
it through `negative_control_outcome=` as an array with one value per row, in the order of your
data; the test estimates the treatment's effect on it. The made-up control below depends on one
covariate and on noise, so the treatment cannot affect it.

```python
import numpy as np

rng = np.random.default_rng(0)
control = gaussian_frame["W1"].to_numpy() + rng.normal(size=len(gaussian_frame))
controlled = ate_result.diagnostics.refute(
    tests=("negative_control_outcome",),
    negative_control_outcome=control,
    random_state=21,
)
print(controlled.summary())
```

```text
test                           mean         SD   expected   result
------------------------------------------------------------------------------
negative-control outcome     0.1082          -   about 0    no pass or fail
...
negative-control outcome   z of the estimate against 0      0.784     0.4330
```

The estimate of 0.11 has a z statistic of 0.78 (p-value 0.43), so it agrees with no effect. A
nonzero estimate flags leftover bias or a bad control, and the test cannot say which; a zero
estimate does not prove that nothing is missing.

The test needs that array, so it is not one of the defaults, and a survival or competing-risk fit
refuses it. `negative_control_kind="pre_treatment"` declares a control measured before the
treatment. The call then refuses a control equal to a covariate you adjust for, or one the
covariates predict so well that less than `negative_control_floor` of its variance is left (by
default 0.01, so 1%). The "Negative controls" part of [refutation
operations](../technical-reference/validation-methods.md#refutation-operations) gives every refusal,
and [A later treatment as a control](#a-later-treatment-as-a-control) covers the control test for a
treatment over time.

## Check stability on bootstrap samples

`bootstrap_stability` refits on bootstrap samples, which draw people with replacement. Each copy of
a person keeps that person's fold and the fit's seeds, so the shift of the mean against the original
fit carries no noise from new folds or seeds. The report shows that shift and the spread of the
refits for you to judge. A shift that is large next to the standard error says the estimate depends
on which people happen to be in the sample. It needs at least 2 draws, and a fit given `split_plan=`
refuses it.

## Check stability under measurement error

If some covariates are measured with error, `bootstrap_measurement_error` asks how far the estimate
moves under an error you declare. Each draw resamples the rows, adds the declared error to the named
covariates, and refits the whole analysis.

```python
from cleverly.validation import BootstrapMeasurementError, RelativeGaussianNoise

measurement = ate_result.diagnostics.refute(
    tests=("bootstrap_measurement_error",),
    bootstrap_measurement_error=BootstrapMeasurementError(
        variables=("W1", "W2"),
        numeric_noise=RelativeGaussianNoise(standard_deviation=0.1),
    ),
    n_replicates=40,
    random_state=21,
)
measurement_draws = measurement.draws_frame("bootstrap_measurement_error")
```

Here noise with a standard deviation of 0.1 times each covariate's own moves the mean of the 40
refits from 1.29 to 1.27, with an SD of 0.11, for you to judge. Name your own covariates in
`variables=` and choose a noise size you find plausible for them; [More on measurement
error](#more-on-measurement-error) gives the rest.

## Count the refits first

To see what a refutation will cost before you run it, call
`ate_result.diagnostics.planned_refits(...)` with the same arguments you would pass to `refute()`.
It fits nothing. It returns one row per test with the number of refits and the model fits they make,
and it raises the same error `refute()` would raise before its first refit. Tests that refit drawn
data, such as `placebo`, `subset` and the generated outcomes, show an upper bound, because a drawn
fold where a yes/no target has one class needs no model. [The technical
reference](../technical-reference/validation-methods.md#refutation-operations) says which counts are
exact.

```python
plan = ate_result.diagnostics.planned_refits(tests=("placebo", "subset", "random_common_cause"))
print(plan.summary())
```

## Go deeper

### Estimates with no no-effect value

The placebo needs a known no-effect value: 0 for a difference, 1 for a ratio. A level, such as the
mean outcome of one arm (one treatment group) or the natural course (`NaturalCourseMean`, the mean
outcome under the treatments people actually got), has none. On such an estimate a call that
includes `placebo` or the negative control raises before any refit and names the tests to drop.
`tests=("random_common_cause", "subset")` runs the other two, and they keep their descriptive
readings for these targets. In a combined report the row defers on `tests` instead ([refutation
operations](../technical-reference/validation-methods.md#refutation-operations)).

### All three readings on a yes-or-no outcome

Under `link="identity"`, the chance of the outcome is the intercept plus the effect for treated rows
plus a covariate term scaled by `adjustment_scale`. Choose values that keep every row's chance
between 0 and 1; the test refuses a declaration that leaves that range on any row (the "Generated
point outcomes" part of [refutation
operations](../technical-reference/validation-methods.md#refutation-operations)).

```python
from cleverly.datasets import make_binary_outcome
from cleverly.validation import BernoulliAdjustmentOutcome

binary_frame, _ = make_binary_outcome(n=200, seed=21)
binary_result = CausalStudy(
    binary_frame,
    design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")),
).estimate(
    ATE(),
    outcome_learner=LogisticRegression(max_iter=1000),
    treatment_learner=LogisticRegression(max_iter=1000),
    cross_fit=False,
    random_state=21,
)
identity = binary_result.diagnostics.refute(
    tests=("simulated_outcome",),
    simulated_outcome=BernoulliAdjustmentOutcome(
        intercept=0.4, effect=0.2, adjustment_scale=0.2, link="identity"
    ),
    random_state=21,
)
print(identity.summary())
```

The summary then lists the bias, the SE ratio and the coverage, each against its band.

### Which fits take a generated outcome

| fit | what runs | what prints |
| --- | --- | --- |
| one treatment, fitted with `TMLE`, collaborative TMLE, `DRTMLE` or `AdaptiveIPTW` | all three tests on arm differences and levels, ratios, the natural course, regimes, policies and working-model coefficients; an incremental target through `plasmode` only | a pass or fail at 400 draws or more. Bias only where the quantity varies across rows, as with the default logit link, or on a fit that reports no interval |
| a continuous dose | `simulated_outcome` on a policy target, with the scalar `effect`; no `plasmode` | as above |
| a treatment over time, with an end-of-study or repeated outcome | `dummy_outcome`, and `simulated_outcome` on a fixed regimen, each with the Gaussian process on a fit with no `q_bounds` and one fold (`n_folds=1`); `plasmode` | the numbers, with no pass or fail. `simulated_outcome` reads the bias only on a level, and `dummy_outcome` alone runs on a policy that is not fixed |
| a treatment over time, with survival or competing events | `dummy_outcome` and `simulated_outcome` with the hazard declarations; `plasmode` | the numbers, with no pass or fail. A hazard that reads the covariates gives the bias only |
| a structural nested mean model; a continuous-time survival fit | refused | nothing |

Collaborative TMLE, `DRTMLE` and `AdaptiveIPTW` are variants of the method that build or guard the
treatment model differently; an incremental target changes each row's odds of treatment. Each case
of the refusal table in [refutation
operations](../technical-reference/validation-methods.md#refutation-operations) names what the test
needs instead, and `plasmode` refuses some fits that the other two take, such as a weighted fit.

### Placebo and noise options

`placebo_mode="draw"` draws the placebo treatment from its overall distribution instead of shuffling
it, and a panel over time always draws a whole treatment history (`"markov"`). On a fit of a
treatment that repeats over time, `noise_nodes=True` adds a noise column at every time point (node);
a fit of one treatment refuses it.

### A later treatment as a control

`future_treatment` runs on fits over time only, and a fit of one treatment refuses it. Pass
`future_treatment_control=`, one value per person, measured before the treatment at node
`future_treatment_node + 1` and outside the history the fit adjusts that treatment for (the earlier
treatments and covariates). If nothing unrecorded drives that later treatment, it should not depend
on the control once the history is taken into account, and the test checks that. The reading holds
only when the control was measured before that treatment and neither affects it nor responds to it;
the code cannot check this, so the report states the condition. The "Negative controls" part of
[refutation operations](../technical-reference/validation-methods.md#refutation-operations) gives
every refusal.

### More on measurement error

The numeric noise uses each variable's standard deviation in its bootstrap sample. A categorical
change acts on the original variable and rebuilds all of its indicator columns, and
`categorical_change_probability` on the declaration sets how often it happens. The test stays out of
the defaults (`DEFAULT_TESTS`) because every draw resamples, perturbs, and refits. Plain bootstrap
inference does not add measurement error. `measurement_error_rule=EmpiricalInclusionRule()` adds a
tail probability to the report, and it needs at least 40 draws.

### Save the draws as they finish

A 400-draw generated-outcome test refits 400 times, which takes tens of minutes on a large fit. Pass
`checkpoint=RefutationCheckpoint("refute-draws")` (from `cleverly.validation`) and each finished
draw is saved as a file in that folder. If the call stops, run the same call again with the same
folder: it loads the saved draws, refits only the rest, and returns the report an uninterrupted call
returns. This covers `dummy_outcome`, `simulated_outcome`, `plasmode`, `bootstrap_stability` and
`bootstrap_measurement_error`, the tests with long draw counts.

A call that changes the fit, the data, the seed or the test's settings refuses the folder and names
what changed. A fit with no seed draws one at the first call, and the refusal of a second call names
it, so pass it as `random_state=` to resume. Pass `progress=print`, or any function, to see each
draw as it finishes.

### Refute a sample of a large fit

On a very large fit, `sample_limit=` runs the refutation on a random part of the data. For example,
`refute(result, sample_limit=20_000)` draws 20,000 of the fit's units at random without replacement,
refits the same analysis on them, and runs every test on that refit, so each test reads the estimate
and standard error of the smaller analysis, never those of the full fit. A clustered fit draws whole
clusters, and `sample_limit` then counts clusters; a panel or survival fit draws subjects with all
their rows. The report prints the full-data estimate beside the sampled one for comparison.

A sampled result answers a narrower question: did the same analysis, run on these units, pass? It is
a test of the analysis at the smaller size, so it can miss a problem that only shows with more data,
and another seed draws other units. The pass or fail rules are the ones for a fit of the sampled
size, so a sample below the size a rule was checked at prints no verdict. A sample that loses a
treatment level or a subgroup (stratum) is refused, and one where a covariate becomes constant
prints its numbers without a verdict. A limit of at least the fit's unit count changes nothing.
`planned_refits(sample_limit=...)` counts the one extra refit of the sample. [The technical
reference](../technical-reference/validation-methods.md#refutation-operations) gives the exact
statement.

`sampling="units"` or `sampling="clusters"` sets what `sample_limit` counts. A rule that disagrees
with the fit is refused before any refit ([refutation
operations](../technical-reference/validation-methods.md#refutation-operations)).

### Every test and argument

The [API page](../api/sensitivity-validation.md) gives each argument's default from the signature,
and [refutation operations](../technical-reference/validation-methods.md#refutation-operations) says
which fits get a pass or fail.

| test | what it does | what you must pass | draws by default |
| --- | --- | --- | --- |
| `placebo` | shuffles the treatment | nothing | 20 |
| `random_common_cause` | adds a column of pure noise | nothing | 20 |
| `subset` | refits on random samples | nothing | 20 |
| `negative_control_outcome` | estimates the effect on an outcome the treatment cannot affect | `negative_control_outcome=` | one fit |
| `dummy_outcome` | replaces the outcome with noise unrelated to the treatment and covariates | a process, except on a continuous fit with no `q_bounds` | 400 |
| `simulated_outcome` | replaces the outcome with one whose effect you set | a process, except on a continuous fit with no `q_bounds` | 400 |
| `plasmode` | also resamples the covariates and redraws the treatment | as `simulated_outcome` | 400 |
| `bootstrap_measurement_error` | resamples people and adds declared noise to covariates | `bootstrap_measurement_error=` | 100 |
| `bootstrap_stability` | resamples people with no noise | nothing | 100 |
| `future_treatment` | estimates the effect of a later treatment on a control measured before it | `future_treatment_control=` | one fit |

| argument | what it sets |
| --- | --- |
| `estimand` | the estimate the tests read |
| `tests` | which tests run |
| `n_replicates` | the draws of each test that draws |
| `subset_fraction` | the share of people each `subset` refit keeps |
| `placebo_mode` | shuffle (`"permute"`), draw (`"draw"`), or draw a treatment history (`"markov"`) |
| `seed_spread` | extra `random_common_cause` refits at fresh seeds |
| `noise_nodes` | a noise column at every time point of a fit over time |
| `negative_control_outcome`, `negative_control_kind`, `negative_control_floor` | the control outcome, when it was measured, and the least share of its variance the covariates must leave |
| `future_treatment_control`, `future_treatment_node` | the control of `future_treatment` and the node before the later treatment |
| `dummy_outcome`, `simulated_outcome` | the processes the generated outcomes draw from |
| `outcome_bands` | the bands the generated-outcome readings must fall in |
| `bootstrap_measurement_error`, `measurement_error_rule` | the declared covariate noise, and the rule behind its tail probability |
| `sample_limit`, `sampling` | refute a random sample of people or clusters |
| `random_state` | the seed of every draw |
| `checkpoint`, `progress` | save each draw as it finishes, and report each one |
