# Try to break the result

A refutation test changes the problem so that the right answer is known in advance, refits the
whole analysis, and checks the estimate against that answer. It can expose a bug, information
leaking between rows, or a wrong setting. It cannot show that no confounder is missing, which is
the question [sensitivity analysis](sensitivity.md) asks. Run the tests with
`result.diagnostics.refute(...)`, or with `cleverly.validation.refute(result, ...)`, which takes
the same arguments.

## Run the default checks

The examples on this page use one small fit of an average treatment effect.

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

With no arguments, `refute()` runs three tests on the estimate named by `estimand`, `"ate"` by
default.

```python
report = ate_result.diagnostics.refute(random_state=21)
print(report.summary())
```

The three default tests are `placebo`, `random_common_cause` and `subset`. Each refits the
analysis 20 times, so the call costs about 60 fits
([refutation operations](../technical-reference/validation-methods.md#refutation-operations)).

`placebo` shuffles the treatment, so the refits should show no effect: 0 for a difference, or 1
for a ratio, read on the log scale. The report tests the mean of the refits against that value
with a t statistic. The test checks the code path, not the confounding in your data.
`placebo_mode="draw"` draws the placebo treatment from its overall distribution instead of
shuffling it, and a fit over time always draws a whole treatment history (`"markov"`).

`random_common_cause` adds a column of pure noise to the adjustment set, so the estimate should
barely move. It prints how far the estimate moved, in standard errors, and never a pass or fail.
`seed_spread=True` adds as many refits at fresh seeds without the noise column, to show how far a
new random split of the rows into folds alone moves the estimate. On a fit over time,
`noise_nodes=True` adds a noise column at every time point; a fit of one treatment refuses it.

`subset` refits on random 70% samples (`subset_fraction=0.7`), so the spread of the refits should
match what the standard error predicts for samples of that size. Without `random_state`, the call
uses the fit's own seed and records it in the report under `random_state`; passing that value back
repeats the report.

## Read the report

`print(report.summary())` shows one row per test: the mean and SD of the refits, what to expect of
them, and the result. A test prints `passed` or `failed` only where a simulation study or an exact
reference backs it for this kind of fit. Otherwise it prints `no pass or fail`, its `passed` is
`None`, and so is `report.passed` unless another test failed. The note under each test says what it
checks and what its numbers can show, and `summary(detail="full")` adds the studied fits each test
was compared with. `bool(report)` raises `TypeError` for a report without a pass or fail, so test
`report.passed` against `True` and `False`. The "Verdicts and evidence" part of [refutation
operations](../technical-reference/validation-methods.md#refutation-operations) says which fits get
a pass or fail.

The placebo needs a known no-effect value: 0 for a difference, 1 for a ratio. A level, such as the
mean outcome of one arm or the natural course (`NaturalCourseMean`), has none. The call refuses
`placebo` and the negative control below for such an estimate before any refit, and the other tests
still run: `random_common_cause` and `subset` keep their descriptive readings for these targets. In
a combined report the row then defers on `tests`, and `tests=("random_common_cause", "subset")` runs
it ([refutation operations](../technical-reference/validation-methods.md#refutation-operations)).

## Check against a known answer

Three tests replace the outcome with one where we know the answer. `dummy_outcome` draws an
outcome unrelated to the treatment and the covariates, so the true effect is zero.
`simulated_outcome` draws an outcome from a process you declare, with an effect you set.
`plasmode` also resamples your covariate rows and draws the treatment from a model fitted to your
data, then draws the outcome from the `simulated_outcome` process, so the test runs on simulated
data built to resemble yours.

Each test reads up to three numbers against the true value: the bias, the ratio of the reported
standard error to the actual spread of the refits, and the share of intervals that hold the truth.
A reading passes when its 95% interval lies inside its band: a bias within 0.20 standard
deviations of the draws, a ratio between 0.85 and 1.15, and a coverage between 0.88 and 0.99.
`outcome_bands=` sets other bands, as an `EquivalenceBands`. Each test makes 400 draws by default;
fewer print the numbers with no pass or fail, and fewer than 10 are refused
([refutation operations](../technical-reference/validation-methods.md#refutation-operations)).

```python
from cleverly.validation import GaussianAdjustmentOutcome

generated = ate_result.diagnostics.refute(
    tests=("simulated_outcome",),
    simulated_outcome=GaussianAdjustmentOutcome(effect=0.5),
    random_state=21,
)
draws = generated.draws_frame("simulated_outcome")
```

That call refits 400 times. Its summary lists the bias, the standard-error ratio, and the
coverage, each with its 95% interval and the band it must fall in, and marks `yes` where the
interval lies inside the band. Each row in `draws` records the child seed, estimate, standard
error, family, or refit failure.

### Which outcome to declare

Declare the process that matches your outcome. The test refuses a process of another kind, such as
the default Gaussian process on a yes/no outcome.

| your outcome | `simulated_outcome` and `plasmode` | `dummy_outcome` |
| --- | --- | --- |
| continuous, fitted in sample (`cross_fit=False`) with no declared range (`q_bounds`) | `GaussianAdjustmentOutcome` (the default) | `GaussianIndependentOutcome` (the default) |
| continuous with a declared range (`q_bounds`), which a cross-fitted `TMLE` of a continuous outcome must have | `BoundedAdjustmentOutcome` | `BoundedAdjustmentOutcome(effect=0, adjustment_scale=0)` |
| yes/no | `BernoulliAdjustmentOutcome` | `BernoulliAdjustmentOutcome(effect=0, adjustment_scale=0)` |
| an event over time | `LogisticHazardOutcome` | `LogisticHazardOutcome(effect=0, adjustment_scale=0)` |
| competing events over time | `MultinomialHazardOutcome` | the same with every effect 0 and `adjustment_scale=0` |

A fit that declares `q_bounds` takes only the bounded process, which draws inside those bounds
with its mean strictly inside them, and a fit without them refuses the bounded process.

The standard-error ratio and the coverage print only when the quantity you estimate takes the same
value at every row of the generated data; otherwise the test reads the bias alone. A difference of
two arms does so under the Gaussian process and under `link="identity"`. Under the default logit
link of `BernoulliAdjustmentOutcome` or `BoundedAdjustmentOutcome`, a nonzero effect varies from row
to row, so a generated yes/no outcome with the default link reads the bias only. For all three
readings on a yes/no outcome, declare `link="identity"` with an intercept and an effect that keep
every row's chance between 0 and 1, or run `plasmode`, which reads all three for every target on a
fit that reports an interval. The "Generated point outcomes" part of [refutation
operations](../technical-reference/validation-methods.md#refutation-operations) gives the rule for
each target.

### Which fits take a generated outcome

| fit | what runs | what prints |
| --- | --- | --- |
| one treatment, fitted with `TMLE`, collaborative TMLE, `DRTMLE` or `AdaptiveIPTW` | all three tests on arm differences and levels, ratios, the natural course, regimes, policies and working-model coefficients; an incremental target through `plasmode` only | a pass or fail at 400 draws or more. Bias only where the quantity varies across rows, as with the default logit link, or on a fit that reports no interval |
| a continuous dose | `simulated_outcome` on a policy target, with the scalar `effect`; no `plasmode` | as above |
| a treatment over time, with an end-of-study or repeated outcome | `dummy_outcome`, and `simulated_outcome` on a fixed regimen, each with the Gaussian process on a fit with no `q_bounds` and one fold (`n_folds=1`); `plasmode` | the numbers, with no pass or fail. `simulated_outcome` reads the bias only on a level, and `dummy_outcome` alone runs on a policy that is not fixed |
| a treatment over time, with survival or competing events | `dummy_outcome` and `simulated_outcome` with the hazard declarations; `plasmode` | the numbers, with no pass or fail. A hazard that reads the covariates gives the bias only |
| a structural nested mean model; a continuous-time survival fit | refused | nothing |

Each case of the refusal table in [refutation
operations](../technical-reference/validation-methods.md#refutation-operations) names what the test
needs instead, and `plasmode` refuses some fits that the other two take, such as a weighted fit.

## Check for hidden bias with a control

`negative_control_outcome=` takes an outcome that the treatment cannot affect but that shares the
confounding of your real outcome. A nonzero estimate flags leftover bias or a bad control, and the
test cannot say which; a zero estimate does not prove that nothing is missing. The test needs that
array, so it is not one of the defaults, and a survival or competing-risk fit refuses it.
`negative_control_kind="pre_treatment"` declares a control measured before the treatment. The call
then refuses a control equal to an adjustment column, or one the covariates predict so well that
less than `negative_control_floor` of its variance is left.

`future_treatment` runs on fits over time only, and a fit of one treatment refuses it. Pass
`future_treatment_control=`, one value per person, measured before the treatment at node
`future_treatment_node + 1` and outside the history the fit adjusts that treatment for. If nothing
unrecorded drives that later treatment, it should not depend on the control once the history is
taken into account, and the test checks that. The code cannot check when the control was measured,
so the report states the condition. The "Negative controls" part of [refutation
operations](../technical-reference/validation-methods.md#refutation-operations) gives every refusal.

## How stable is the estimate

`bootstrap_stability` refits on bootstrap samples, which draw people with replacement. Each copy
of a person keeps that person's fold and the fit's seeds, so the shift of the mean against the
original fit carries no noise from new folds or seeds. The report shows that shift and the spread
of the refits, with no pass or fail. It needs at
least 2 draws, and a fit given `split_plan=` refuses it.


Use bootstrap measurement error to assess estimate stability under declared covariate error.
The operation samples rows first. It then perturbs original adjustment variables and refits the
complete estimator.

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

The numeric multiplier uses each variable's standard deviation in its bootstrap sample.
Categorical changes act on the original variable and rebuild its complete indicator block.
Set `categorical_change_probability` on the declaration to control those changes.

The operation reports how far the estimate moves under the declared error and states no verdict.
It stays outside `DEFAULT_TESTS` because every draw resamples, perturbs, and refits.
Plain bootstrap inference does not add measurement error.
`measurement_error_rule=EmpiricalInclusionRule()` supplies the tail probability the report prints,
and it needs at least 40 draws.

## Count the refits first

To see what a refutation will cost before you run it, call
`ate_result.diagnostics.planned_refits(...)` with the same arguments you would pass to
`refute()`. It fits nothing. It returns one row per test with the number of refits and the model
fits they make, and it raises the same error `refute()` would raise before its first refit. Tests
that refit drawn data, such as `placebo`, `subset` and the generated outcomes, show an upper
bound, because a drawn fold where a binary target has one class needs no model.
[The technical reference](../technical-reference/validation-methods.md#refutation-operations)
says which counts are exact.

```python
plan = ate_result.diagnostics.planned_refits(tests=("placebo", "subset", "random_common_cause"))
print(plan.summary())
```

## Go deeper

### Save the draws as they finish

A 400-draw generated-outcome test refits 400 times, which takes tens of minutes on a large fit. Pass
`checkpoint=RefutationCheckpoint("refute-draws")` (from `cleverly.validation`) and each finished
draw is saved as a file in that folder. If the call stops, run the same call again with the same
folder: it loads the saved draws, refits only the rest, and returns the report an uninterrupted
call returns. This covers `dummy_outcome`, `simulated_outcome`, `plasmode`, `bootstrap_stability`
and `bootstrap_measurement_error`, the tests with long draw counts.

A call that changes the fit, the data, the seed or the test's settings refuses the folder and
names what changed. A fit with no seed draws one at the first call, and the refusal of a second
call names it, so pass it as `random_state=` to resume. Pass `progress=print`, or any function,
to see each draw as it finishes.

### Refute a sample of a large fit

On a very large fit, `sample_limit=` runs the refutation on a random part of the data. For
example, `refute(result, sample_limit=20_000)` draws 20,000 of the fit's units at random without
replacement, refits the same analysis on them, and runs every test on that refit, so each test
reads the estimate and standard error of the smaller analysis, never those of the full fit. A
clustered fit draws whole clusters, and `sample_limit` then counts clusters; a panel or survival
fit draws subjects with all their rows. The report prints the full-data estimate beside the
sampled one for comparison.

A sampled result answers a narrower question: did the same analysis, run on these units, pass?
It is a test of the analysis at the smaller size, so it can miss a problem that only shows with
more data, and another seed draws other units. The pass or fail rules are the ones for a fit of
the sampled size, so a sample below the size a rule was checked at prints no verdict. A sample
that loses a treatment level or a stratum is refused, and one where a covariate becomes constant
prints its numbers without a verdict. A limit of at least the fit's unit count changes nothing.
`planned_refits(sample_limit=...)` counts the one extra refit of the sample.
[The technical reference](../technical-reference/validation-methods.md#refutation-operations)
gives the exact statement.

`sampling="units"` or `sampling="clusters"` sets what `sample_limit` counts. A rule that disagrees
with the fit is refused before any refit
([refutation operations](../technical-reference/validation-methods.md#refutation-operations)).

### Every test and argument

The two tables list every test and every argument of `refute()`. The [API
page](../api/sensitivity-validation.md) gives each argument's default from the signature, and
[refutation operations](../technical-reference/validation-methods.md#refutation-operations) says
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
