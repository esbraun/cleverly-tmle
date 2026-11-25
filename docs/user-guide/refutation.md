# Try to break the result

A refutation test changes the problem so that the right answer is known in advance, refits the whole
analysis, and checks the estimate against that answer. Shuffle the treatment, for example, and the
effect should be about zero. A test can expose a bug, or a setting that does not suit the data,
which shows up as bias or poor coverage on simulated data like yours; coverage is the share of 95%
intervals that hold the true value, which should be near 95%. It cannot detect rows that depend on
each other when you have not declared it, such as repeated visits of one person entered as separate
people: declare those as clusters ([data and study design](data-design.md)).

A refutation cannot show that no confounder is missing. A confounder is a variable that affects both
the treatment and the outcome, and a missing one is a confounder the data do not record;
[sensitivity analysis](sensitivity.md) asks how strong one would have to be. Passing these tests
says that the analysis returns known answers on data like yours, not that your estimate is causal.

## The example fit

The examples on this page use one small simulated fit of an average treatment effect: the change in
the mean outcome `Y` if every row got the treatment `A`, compared with none. The fit names the
outcome, the treatment and the covariates to adjust for (`PointTreatment`). It uses two models: an
outcome model (`outcome_learner`), which predicts `Y` from `A` and the covariates, and a treatment
model (`treatment_learner`), which predicts each row's chance of `A`. TMLE, the default method,
combines them: it corrects the outcome model's predictions with the treatment model.

Both models are fitted on every row (`cross_fit=False`). The default, cross-fitting, predicts each
row from models trained without it, and on a continuous outcome it needs a declared range of values,
which this simulated `Y` lacks. The calls below are the same on either kind of fit. If you have your
own fitted result, skip this block and use your result, with its own settings, in place of
`ate_result`.

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

## Run the default checks

With no arguments, `refute()` runs three tests on the `ate` estimate; `estimand=` names another row
of the fit's summary.

```python
report = ate_result.diagnostics.refute(random_state=21)
print(report.summary())
```

The three default tests are `placebo`, `random_common_cause` and `subset`. Each refits the analysis
20 times, so the call costs about 60 fits: under a second on this small fit, and about 60 times as
long as one fit on a large one.

- `placebo` shuffles the treatment, so the refits should show no effect. The report tests the mean
  of the refits against no effect with a t statistic. It checks that the software returns no effect
  when there is none; it says nothing about confounding in your data.
- `random_common_cause` adds a column of pure noise to the covariates, so the estimate should
  barely move. It prints how far the estimate moved, in standard errors, and no fit gets a pass or
  fail for it.
- `subset` refits on random 70% samples (`subset_fraction=0.7`), so the spread of the refits should
  match what the standard error predicts for samples of that size.

## Read the report

On this fit the summary reads, without its notes:

```text
                               Refutation tests
==============================================================================
Estimate tested:  ate = 1.2864          Repeats per test:                   20
Tests run:        3                     Result:                no pass or fail
==============================================================================
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

All three tests look healthy. The header gives the estimate under test, 1.286, which is also what
the noise column should leave unchanged. Compare each `mean` (or `SD`, for the subsamples) with the
`expected` column, and read the statistics below it:

- The shuffled-treatment refits average -0.03, and their t statistic has a p-value of 0.55, so they
  agree with no effect. Beside it, `mean std. error / refit SD` compares the standard errors the
  shuffled refits reported with how much they actually varied; near 1 the two agree, and below 1
  the standard errors are smaller than the real spread. It has no p-value, and 20 refits measure a
  spread only roughly, so 0.66 is not yet a warning. Rerun the placebo with more refits first:
  `ate_result.diagnostics.refute(tests=("placebo",), n_replicates=200, random_state=21)` reads 0.93
  on this fit. A value that stays well below 1 says the standard error may be too small; check the
  `subset` ratio below, which tests it on your real treatment.
- The noise column moved the estimate by 0.004 standard errors, and `SD of the shifts` says how much
  the 20 refits' shifts varied. Both are in standard errors, and the interval spans about four
  standard errors, so shifts of a few hundredths are negligible.
- The subsample refits spread 0.87 times as much as the standard error predicts, with a p-value of
  0.48, so the standard error is about the right size.

A placebo mean far from 0, or a spread ratio far from 1, each with a small p-value (say below 0.05),
would point to a problem in the analysis. Then run the checks on [Check a fitted
result](assessment.md), try other learners (the models you pass for the outcome and the treatment)
or cross-fitting, and run the tests again. The `result` column prints `passed` or `failed` only
where a simulation study or an exact reference backs that test for this kind of fit. Here none does,
so each prints `no pass or fail` and you compare the numbers as above. The "Verdicts and evidence"
part of [refutation operations](../technical-reference/validation-methods.md#refutation-operations)
says which fits get a pass or fail, and [use the report in code](#use-the-report-in-code) covers
`report.passed`.

## Check against a known answer

Three tests replace the outcome with one where we know the answer, and each refits hundreds of
times, so [count the refits](#count-the-refits-before-a-long-test) on a large fit first.
`dummy_outcome` draws an outcome unrelated to the treatment and the covariates, so the true effect
is zero. `simulated_outcome` draws an outcome with an effect you set, plus a covariate term that
adds confounding the analysis must remove and random noise. `plasmode` also resamples your covariate
rows and draws the treatment from a model fitted to your data, then draws the outcome the same way,
so the test runs on simulated data built to resemble yours.

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

The example sets an effect of 0.5, half the default noise standard deviation of 1. To make the
simulated outcome resemble yours, set `effect` near your estimate and the noise near the spread your
outcome model leaves, with `noise=GaussianNoise(standard_deviation=...)` from `cleverly.validation`.
That spread is the square root of `result.sensitivity.elements().sigma2`, about 0.9 on this fit. The
example call draws 400 outcomes, each with a true effect of 0.5, and refits the analysis on each;
this small fit takes a few seconds. Its summary reads, shortened:

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
between 0.88 and 0.99. Here all three are inside, so the test passed: on outcomes simulated this
way, the analysis recovers the effect, its standard error matches the real spread, and its
intervals cover about 95% of the time.

`outcome_bands=` sets other bands (an `EquivalenceBands` object, one band per number). Each test
makes 400 draws by default; fewer print the numbers with no pass or fail, and fewer than 10 are
refused. Each row in `draws` records one draw: its seed, the estimate and standard error, the kind
of simulated outcome that drew it (`family`), or the refit failure. The result holds only for the
simulated outcome you declared, which is not your real outcome.

### Which simulated outcome fits your data

Declare the kind of simulated outcome that matches your outcome. The test refuses one of another
kind, such as the default Gaussian outcome on a yes/no outcome. A declared range is the range a
continuous outcome can take by how it is measured, passed to `estimate()` as, for example,
`q_bounds=(0, 100)`; a cross-fitted TMLE fit of a continuous outcome must have one. The fit's
settings record it in `result.method.targeting.q_bounds`, and `None` means you declared none.

| your outcome | `simulated_outcome` and `plasmode` | `dummy_outcome` |
| --- | --- | --- |
| continuous, fitted in sample (`cross_fit=False`) with no declared range (`q_bounds`) | `GaussianAdjustmentOutcome` (the default) | `GaussianIndependentOutcome` (the default) |
| continuous with a declared range (`q_bounds`) | `BoundedAdjustmentOutcome` | `BoundedAdjustmentOutcome(effect=0, adjustment_scale=0)` |
| yes/no | `BernoulliAdjustmentOutcome` | `BernoulliAdjustmentOutcome(effect=0, adjustment_scale=0)` |
| an event over time | `LogisticHazardOutcome` | `LogisticHazardOutcome(effect=0, adjustment_scale=0)` |
| competing events over time | `MultinomialHazardOutcome` | the same with every effect 0 and `adjustment_scale=0` |

A fit that declares `q_bounds` takes only the bounded outcome, which draws inside those bounds with
its mean strictly inside them, and a fit without them refuses the bounded outcome. On a yes/no
outcome, `simulated_outcome` with the default `BernoulliAdjustmentOutcome` reports the bias only. A
nonzero effect then differs from row to row, and because the test keeps your rows and redraws only
the outcome, the spread of its estimates leaves out the uncertainty from which rows were sampled,
which the standard error includes; [all three numbers on a yes-or-no
outcome](#all-three-numbers-on-a-yes-or-no-outcome) shows how to get the standard-error ratio and
the coverage too. [Which fits take a generated outcome](#which-fits-take-a-generated-outcome) lists
the fits.

## Test a control outcome the treatment cannot affect

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
==============================================================================
Test statistics
test                       statistic                        value    p-value
------------------------------------------------------------------------------
negative-control outcome   z of the estimate against 0      0.784     0.4330
```

The estimate of 0.11, with a p-value of 0.43, agrees with no effect. A clearly nonzero estimate
flags leftover bias, or a bad control: one the treatment can affect after all, or one that does not
share your outcome's confounding. The test cannot say which, and a zero estimate does not prove
that nothing is missing.

The test needs that array, so it is not one of the defaults, and a fit of the time to an event
(survival or competing risks) refuses it. `negative_control_kind="pre_treatment"` declares a control
measured before the treatment. The call then refuses a control equal to a covariate you adjust for,
or one the covariates predict so well that less than `negative_control_floor` of its variance is
left (by default 0.01, so 1%). The "Negative controls" part of [refutation
operations](../technical-reference/validation-methods.md#refutation-operations) gives every refusal,
and [A later treatment as a control](#a-later-treatment-as-a-control) covers the control test for a
treatment over time.

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
print(measurement.summary())
measurement_draws = measurement.draws_frame("bootstrap_measurement_error")
```

```text
test                    mean         SD   expected            result
------------------------------------------------------------------------------
measurement error       1.27       0.11   moves from 1.2864   no pass or fail
```

The 40 refits average 1.27 against the fit's 1.29. Part of any drift comes from resampling alone:
`bootstrap_stability` with the same seed and draw count averages 1.26, so on this fit the declared
noise moved the estimate by less than 0.01. Error in a covariate weakens the adjustment for it, so a
larger error moves the estimate toward the one you would get without adjusting for it. The yardstick
is your conclusion: the fit's 95% interval runs from 1.03 to 1.54, so a drift this small would not
move it near 0. Name your own covariates in `variables=` and choose a noise size you find plausible
for them; [More on measurement error](#more-on-measurement-error) gives the rest.

## Go deeper

### Count the refits before a long test

To see what a refutation will cost before you run it, call `planned_refits()` with the same
arguments you would pass to `refute()`. It fits nothing, and it raises the same error `refute()`
would raise before its first refit.

```python
plan = ate_result.diagnostics.planned_refits(tests=("placebo", "subset", "random_common_cause"))
print(plan.summary())
```

```text
test                     draws     refits   model fits   candidate fits
------------------------------------------------------------------------------
placebo                     20         20   at most 40       at most 40
subset                      20         20   at most 40       at most 40
random_common_cause         20         20           40               40
```

Each refit here fits two models, so `model fits` counts 40 per test; `candidate fits` also counts
the models inside a learner that combines several, and equals `model fits` here. Tests that refit
drawn data, such as `placebo` and `subset`, show an upper bound, because a refit can skip a model
when the rows drawn for it hold only one value of a yes/no outcome or treatment. [The technical
reference](../technical-reference/validation-methods.md#refutation-operations) says which counts are
exact.

### Use the report in code

`cleverly.validation.refute(result, ...)` takes the same arguments as the method. A test with no
pass or fail has `passed` equal to `None`, and so does `report.passed` unless another test failed.
`bool(report)` raises `TypeError` for a report without a pass or fail, so test `report.passed`
against `True` and `False`. Without `random_state`, the call uses the fit's seed, or draws one for
an unseeded fit, and records it in the report under `random_state`; passing that value back repeats
the report.

`seed_spread=True` adds as many `random_common_cause` refits at fresh seeds without the noise
column. On a cross-fitted fit, which splits the rows into folds and predicts each row from models
trained without its fold, those refits show how far a new random split alone moves the estimate.

### Estimates with no no-effect value

The placebo needs a known no-effect value: 0 for a difference, 1 for a ratio. A level, such as the
mean outcome of one arm (one treatment group) or the natural course (`NaturalCourseMean`, the mean
outcome under the treatments people actually got), has none. On such an estimate a call that
includes `placebo` or the negative control raises before any refit and names the tests to drop.
`tests=("random_common_cause", "subset")` runs the other two, and they keep their descriptive
numbers for these estimates. In the combined report of `assess()`, the refutation row instead waits
under `Not run`, and its next step asks you to pass `tests=` without those two ([refutation
operations](../technical-reference/validation-methods.md#refutation-operations)).

### All three numbers on a yes-or-no outcome

The tests that keep your covariate rows, `dummy_outcome` and `simulated_outcome`, redraw only the
outcome. The standard-error ratio and the coverage therefore print only when the quantity you
estimate takes the same value at every row of the generated data; otherwise the test reads the bias
alone. A difference of two treatment groups (arms) does so under the Gaussian outcome and under
`link="identity"`. Under the default logit link (the outcome's log-odds are linear in the effect and
the covariates) of `BernoulliAdjustmentOutcome` or `BoundedAdjustmentOutcome`, a nonzero effect
varies from row to row, so a generated yes/no outcome with the default link reads the bias only.
`plasmode` reads all three for every estimate on a fit that reports an interval. The "Generated
point outcomes" part of [refutation
operations](../technical-reference/validation-methods.md#refutation-operations) gives the rule for
each estimate.

Under `link="identity"`, the chance of the outcome is the intercept plus the effect for treated rows
plus a covariate term scaled by `adjustment_scale`. Choose values that keep every row's chance
between 0 and 1; the test refuses a declaration that leaves that range on any row ([refutation
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
| one treatment, fitted with `TMLE`, collaborative TMLE, `DRTMLE` or `AdaptiveIPTW` | all three tests on differences and means of treatment groups, ratios, the natural course (the mean under the treatments people got), treatment rules (regimes), policies and the coefficients of a trend line across treatment levels (working-model coefficients); an incremental estimate, which changes each row's odds of treatment, through `plasmode` only | a pass or fail at 400 draws or more. Bias only where the quantity varies across rows, as with the default logit link, or on a fit that reports no interval |
| a continuous dose | `simulated_outcome` on a policy estimate, with the scalar `effect`; no `plasmode` | as above |
| a treatment over time, with an end-of-study or repeated outcome | `dummy_outcome`, and `simulated_outcome` on a fixed regimen, each with the Gaussian outcome on a fit with no `q_bounds` and one fold (`n_folds=1`); `plasmode` | the numbers, with no pass or fail. `simulated_outcome` reads the bias only on a level, and `dummy_outcome` alone runs on a policy that is not fixed |
| a treatment over time, with survival or competing events | `dummy_outcome` and `simulated_outcome` with the hazard declarations; `plasmode` | the numbers, with no pass or fail. A hazard that reads the covariates gives the bias only |
| a structural nested mean model; a continuous-time survival fit | refused | nothing |

Collaborative TMLE, `DRTMLE` and `AdaptiveIPTW` are variants of the method that build or guard the
treatment model differently. Each case of the refusal table in [refutation
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
on the control once the history is taken into account, and the test checks that. The test means
something only when the control was measured before that treatment and neither affects it nor
responds to it; the code cannot check this, so the report states the condition. The "Negative
controls" part of [refutation
operations](../technical-reference/validation-methods.md#refutation-operations) gives every refusal.

### Stability on bootstrap samples

`bootstrap_stability` refits on bootstrap samples, which draw people with replacement. Each copy of
a person keeps that person's fold and the fit's seeds, so the shift of the mean against the original
fit carries no noise from new folds or seeds.

```python
stability = ate_result.diagnostics.refute(
    tests=("bootstrap_stability",), n_replicates=40, random_state=21
)
print(stability.summary())
```

```text
test                      mean         SD   expected          result
------------------------------------------------------------------------------
bootstrap stability       1.26       0.11   SD about 0.1107   no pass or fail
...
bootstrap stability   shift from the original fit     -0.025          -
```

The report shows the shift of the mean and the spread of the refits; no study sets a threshold for
either. A shift that is large next to the standard error says the estimate is not stable under
resampling. Here the shift is -0.025 against a standard error of 0.13. On a cross-fitted fit the
spread tends to exceed the standard error, because each copy keeps its fold and each resample trains
on fewer distinct people; here, on a fit without folds, it is 0.11 against 0.13. The `expected`
column repeats the spread and is not a reference.

The note under the report points to `subset` as the calibrated check of the standard error. The test
needs at least 2 draws, and a fit given its own fold assignment (`split_plan=`) refuses it.

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
| `dummy_outcome` | replaces the outcome with noise unrelated to the treatment and covariates | a simulated outcome, except on a continuous fit with no `q_bounds` | 400 |
| `simulated_outcome` | replaces the outcome with one whose effect you set | a simulated outcome, except on a continuous fit with no `q_bounds` | 400 |
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
| `dummy_outcome`, `simulated_outcome` | the simulated outcomes the tests draw |
| `outcome_bands` | the bands the generated-outcome numbers must fall in |
| `bootstrap_measurement_error`, `measurement_error_rule` | the declared covariate noise, and the rule behind its tail probability |
| `sample_limit`, `sampling` | refute a random sample of people or clusters |
| `random_state` | the seed of every draw |
| `checkpoint`, `progress` | save each draw as it finishes, and report each one |
