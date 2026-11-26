# Read and save results

Print `result.summary()` to see a fitted result's answer: the effect, how uncertain it is, and the
setup that produced it. A support label on each estimate says whether a simulation study has checked
its interval, and `result.save()` keeps the whole result for later. The checks you run on a result
have their own pages: [Check a fitted result](assessment.md), [Sensitivity analysis](sensitivity.md)
and [Try to break the result](refutation.md).

## Reading a summary

The [quickstart](../getting-started/quickstart.md) fit asks how much a care-navigation offer
(the treatment) changes patients' transition score (the outcome), and prints this:

```text
                            Effect estimate (TMLE)
==============================================================================
Question:      average effect of        Rows:                            2,000
               transition_navigation    Treated share:                   0.506
               on transition_score (1   Out-of-fold predictions:            no
               compared with 0)         Outcome model:       linear regression
Outcome:       transition_score         Treatment model:   logistic regression
               (continuous)             Probability limits:  [0.01471, 0.9853]
Adjusted for:  discharge_risk,
               prior_utilization,
               medication_burden, age
Method:        targeted maximum
               likelihood (TMLE)
==============================================================================
      estimate   std. error           95% CI    p-value
------------------------------------------------------------------------------
ate      1.513        0.046   [1.422, 1.603]    <0.0001
==============================================================================
Notes:
[1] The interval of ate is widened on the side where a skewed outcome makes
    misses more likely, so it need not be centred on the estimate. Its t
    distribution has 250 or more degrees of freedom, so it is close to the
    normal one. The p-value is the level at which the interval would just
    reach the null value.
```

The row `ate` is the average treatment effect: the change in the mean score if every patient got the
offer, compared with no patient. The `estimate` is on the scale of the outcome, here score points.
The `std. error` measures how much the estimate would move from sample to sample, and the `p-value`
tests whether the effect is zero. The `95% CI` is the 95% confidence interval: a range built so
that, if the analysis were repeated on many new samples, about 95 in 100 such ranges would contain
the true effect. That holds in large samples when the models are reasonably accurate.

In words a manager can follow: offering navigation raised the mean score by about 1.5 points, and
the data are consistent with an effect between 1.42 and 1.60 points. That sentence is causal only if
the adjusted covariates include everything that drove both the offer and the score, and no data can
check that. [Sensitivity analysis](sensitivity.md) asks how strong a missing one would have to be.

The header records the question and how it was fitted. The fit uses two models: the
outcome model predicts the score from the offer and the covariates, and the treatment model predicts
each patient's chance of the offer from the covariates. The method, targeted maximum likelihood
estimation (TMLE), starts from the outcome model's predictions and adds a correction step that uses
the treatment model to remove most of the bias the outcome model leaves, when the treatment model is
close to right. The correction weights each patient by one over their predicted chance of the option
they got.

`Out-of-fold predictions: no` means both models were fitted and scored on every row
(`cross_fit=False`). That is sound for simple models like these linear and logistic regressions.
With flexible models, such as random forests or boosting, it can make the interval too narrow, and
the summary then warns. For those, keep the default, cross-fitting: each row's predictions come from
models trained without that row, like out-of-fold predictions in cross-validation. A cross-fitted
continuous outcome needs the range its values can take by how it is measured, such as
`q_bounds=(0, 100)` for a 0-to-100 score, passed to `estimate()`. The quickstart's score has no such
range, so fitting on every row with simple models is its only route ([quickstart, on your own
data](../getting-started/quickstart.md)).

The `Probability limits` cap each predicted chance, so that no patient gets an extreme weight. When
the cap changes some patients' weights, a note says so; no such note prints here, so the cap changed
no patient. [Read cleverly's output](reading-output.md) explains every other header line.

Numbered notes follow the table, and they appear only when something changes how to read a number,
with any warning first; a summary with no notes has nothing to flag. The quickstart's note says its
interval is widened on the side a skewed outcome misses on. A note can also say that an estimate is
experimental (see [the support label](#read-the-support-label) below), or that a row has no interval
and why. `summary(detail="full")` adds the assumptions under which the effect can be read as caused
by the treatment, one per line, and the settings that change how to read the numbers, such as the
limits on predicted chances and how many patients reached them.

## Read the support label

Every estimate carries a support label, `supported` or `experimental`. It answers one question: has
a simulation study checked the coverage of this kind of interval for this kind of fit? Coverage is
the share of intervals that hold the true value across many simulated data sets; a 95% interval
should cover about 95% of the time. The package keeps a fixed list of the setups its simulation
studies checked, each one a combination of method, kind of treatment and data, and question. You do
not request the label, and no argument turns it off.

A `supported` label means that a simulation study checked a setup like your fit and passed. The
label names the study, and `result["ate"].support.limits` lists any conditions the study set, such
as the models it used for the outcome and the treatment or the smallest number of rows. An
`experimental` label means that the package computes the estimate but makes no coverage claim: no
study checked this setup, or a stated condition applies, such as too few clusters (groups of related
rows). The label carries the reason. The quickstart's `ate` reads `supported`, with no extra
conditions.

An `experimental` label does not mean the estimate is wrong. It means no simulation has checked the
number you read: the interval, the band of several estimates described below, or a value computed
from other estimates, such as a ratio. Report such an estimate with a plain caveat, such as "no
simulation study has checked this interval for this kind of analysis". [Run a coverage study of your
own setup](#run-a-coverage-study-of-your-own-setup) shows how to check it yourself. A `supported`
label says how the interval behaved in simulations; it does not show that the effect is causal.

Read the label with `result["ate"].support.label`, or in the `support` column of
`result.to_frame()`. The summary prints a `support` column and a note only when an estimate is
experimental. [Where the label appears](#where-the-label-appears) lists every place, and the
[validation grid](../technical-reference/method-evidence/validation-grid.md) lists the studies
behind the labels.

## Confidence intervals, the bootstrap and bands

A result reports a 95% confidence interval for each estimate by default, built from the standard
error. It is not always the estimate plus or minus two standard errors, and it can sit unevenly
around the estimate. The interval of a yes/no outcome's mean stays between 0 and 1. For a
continuous outcome, the interval is widened on the side where a skewed outcome makes it more likely
to miss the true value. The note under the table says which form an estimate carries, and [which
interval each estimate carries](#which-interval-each-estimate-carries) lists the forms.

The summary adds a `df` column, the degrees of freedom of a t distribution, when the interval uses
one with fewer than 250 of them, which widens it. A treatment group that shows only one outcome
value, or an estimate at exactly 0 or 100 percent, gets no interval, and neither do the effects that
use it. [Small-sample scale](../technical-reference/inference.md#small-sample-scale) gives each
formula and every case.

A fit that reports several estimates also prints a band that covers them all at once, wider than
each interval. Use it when you read several rows together, for example to pick the largest effect
([bands](#bands-that-cover-several-estimates-at-once)). A bootstrap adds a second interval, from
refits on resampled data, beside the usual one.

### Change the level or add a bootstrap

Pass `alpha` to change the level: `alpha=0.1` gives 90% intervals. The support label does not change
with the level, and the studies behind a `supported` label checked 95% intervals. Pass `n_bootstrap`
to add a bootstrap, which refits every model on resampled copies of the data. A result can refit
itself with its own models and settings, whether you fitted it in this session or loaded it from a
file:

```python
bootstrapped = result.identified_effect.estimate(result.method, alpha=0.1, n_bootstrap=20)
print(bootstrapped.summary())
```

`result.identified_effect` is the question the result answered, on the data it was fitted to, and
`result.method` holds its models and settings; keyword arguments after it change only the settings
you name. On the quickstart fit the summary then adds:

```text
      estimate   std. error           90% CI    p-value
------------------------------------------------------------------------------
ate      1.513        0.046   [1.437, 1.589]    <0.0001
==============================================================================
Bootstrap (iid resampling, 20 usable replicates)
      bootstrap SE   percentile 90% CI
------------------------------------------------------------------------------
ate          0.036      [1.472, 1.583]
...
[2] No validation study covers the bootstrap interval of ate for this kind of
    fit, so cleverly marks it experimental.
```

The "Bootstrap" table gives the bootstrap standard error and a percentile interval, the middle 90%
of the replicate estimates. It is reported beside the usual interval and does not replace it. Twenty
replicates keep the example fast; use a few hundred in an analysis, and [count the
fits](methods-learners.md#count-the-model-fits-before-a-long-run) first, since each replicate refits
every model.

The bootstrap interval carries its own label, in `bootstrapped["ate"].bootstrap.support`. A
simulation study covers it for a few fits. It reads `supported` on a yes/no outcome fitted on every
row (`cross_fit=False`), the setup the [full-refit bootstrap
study](../technical-reference/method-evidence/full-refit-bootstrap-and-derived-contrasts.md)
checked, and `experimental` on a continuous outcome such as the quickstart's, as note 2 says. For a
treatment decided once (not at several times) with cross-fitting, the default, it reads
`experimental`, and the label's reason says why. [Bootstrap options](#bootstrap-options) covers
clustered resampling and the fits that refuse a bootstrap.

## Save and load a result

```python
from cleverly import load

result.save("analysis.joblib")
restored = load("analysis.joblib")
assert restored.parameter_keys == result.parameter_keys
assert type(restored.method) is type(result.method)
assert type(restored.method.models.outcome_learner) is type(result.method.models.outcome_learner)
```

Change the file name to your own. The three `assert` lines only show that the loaded result names
the same estimates, method and models; you do not need them. The file holds the whole result,
including unfitted copies of the outcome and treatment models, so a loaded result can still run
the checks that refit its models. `save()` writes the result with `cloudpickle`. It saves an
importable function or class by reference, and a lambda, a closure, or an object defined in a
notebook or in `__main__` by value, so a result keeps the treatment rules, models and other
functions you wrote inline. A by-value object loads only under the Python minor version that saved
it.

The loader is `joblib.load`, which uses pickle. Loading can execute arbitrary code, so load only
trusted files. Use an environment with the same Python minor version and compatible sklearn and
third-party estimator versions.

`result.save()` records the `cleverly` version that wrote the file. When another version saved it,
`cleverly.load()` warns with `VersionMismatchWarning` and loads the file as saved, with no
migration. Its support labels, check reports and saved check answers can differ from what this
version computes, so fit the analysis again with this version. Development snapshots between two
releases share one version string, so the check does not see a change between them.

## Go deeper

### Run a coverage study of your own setup

A support label says whether a simulation study covers your kind of fit. When the label reads
`experimental`, or your learners, sample size or kind of outcome differ from the studied ones, you
can run a small study of your own. A coverage study draws many data sets from a simulated process
whose true effect you know, runs your analysis on each, and counts how often the interval holds the
truth (`CoverageStudy`). One fit cannot tell you this, because coverage describes repeated samples
([coverage studies](../technical-reference/validation-methods.md#coverage-studies)).

```python
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import TMLE
from cleverly.validation import CoverageStudy


def simulate_data(n, seed):
    rng = np.random.default_rng(seed)
    w1, w2 = rng.normal(size=n), rng.normal(size=n)
    a = rng.binomial(1, expit(0.5 * w1))
    y = 1.0 * a + w1 + 0.5 * w2 + rng.normal(size=n)
    frame = pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2})
    return frame, {"ate": 1.0}


coverage = CoverageStudy(
    dgp=simulate_data,
    estimator=lambda: TMLE(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        cross_fit=False,
    ),
    n=300,
    n_replicates=50,
    estimands=("ate",),
    fit_kwargs={"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2"]},
    seed=1,
).run()
print(coverage.summary())
```

`simulate_data` is the part you write: a function that simulates data like yours. It takes a row
count and a seed, and returns a data frame and the true value of each quantity you estimate.
`CoverageStudy` takes a function that builds a fresh estimator from `cleverly.estimators`, not a
`CausalStudy`, and passes the column names through `fit_kwargs`. Give it the learners and the
`cross_fit` setting of your own fit.

The summary has three numbers to read. `coverage` is the share of intervals that held the truth,
which should be near 95%. `bias` is the mean error of the estimates, and `summary(detail="full")`
adds it times the square root of the row count (root-n bias). `SE ratio` divides the average
reported standard error by the actual spread of the estimates (`MC SD`), and a value near 1 means
the reported uncertainty is honest. Monte Carlo error is the noise that comes from running a finite
number of draws. The line under the table flags coverage below 95% by more than three Monte Carlo
standard errors, or a bias more than three of its standard errors from zero.

The numbers hold only for the process you simulated, so make it resemble your data: the row count,
the covariates, the treated share and the kind of outcome. The truth that `simulate_data` returns
must be the quantity your fit estimates, on the same population. Fifty draws keep the example fast,
but at 50 draws the coverage has a Monte Carlo standard error of about 3 points, so it cannot tell
92% from 95%; an analysis needs several hundred. Each draw is one full fit: `n_jobs=` runs draws in
parallel, and `checkpoint=` lets a stopped study resume. A draw whose fit fails stays in the study
as a failure and is not dropped.

Your study does not change the support label. Labels come from the fixed support entries in the
package, each backed by a study in the [validation
grid](../technical-reference/method-evidence/validation-grid.md), and `CoverageStudy` only reports
the label its draws carried. Report your numbers beside the label as your own check. If you run your
own loop of fits, build one `ReplicationRecord` (the estimate, standard error and interval of one
fit) per fit and pass the records to `summarize_replications`, which returns the same numbers for
one estimate.

Without writing a simulator, the plasmode test on [Try to break the result](refutation.md) reads the
same three numbers on simulated data built from your fit: covariates resampled from your rows, a
treatment drawn from a model fitted to them, and an outcome you declare.

### Where the label appears

| place | what it shows |
| --- | --- |
| `repr(estimate)` | the marker `[experimental]` after the estimate |
| `result.summary()` | a `support` column, only when an estimate is experimental, and an `experimental` note below the table that says in plain words why. `summary(detail="full")` prints each reason under Settings |
| `estimate.support` | a `Support` record with `label`, `reason`, `evidence`, `row` (the support entry) and `limits` |
| `estimate.to_dict()` and `result.to_frame()` | `support` on every row. An experimental row adds `support_reason` and `support_evidence`. A row with limits adds `support_limits` |
| `estimate.bootstrap.support` and `bootstrap_support` | the label of the percentile interval, which a support entry can cover apart from the usual interval built from the standard error (the Wald interval) |
| `result.simultaneous.support` | the label of the band |
| `result.config.support_key` | the key that matched the fit to a support entry. `result.save()` and `cleverly.load()` keep it with every label |

The `evidence` of a `supported` estimate holds the short names of its studies. The `evidence` of an
experimental estimate holds the ids of the [roadmap](../roadmap.md) items, such as `X54`, that would
supply the missing evidence. The reason of an experimental label names the support entry it read,
such as `T01.continuous`.

### Labels of ratios and other derived estimates

A derived estimate, computed from other estimates, such as a ratio or a contrast, is `supported`
only when its support entry lists derived estimates (the `derived` kind) and every input is
`supported`. A restricted mean (the average time free of the event up to a set horizon) read from a
grid of check times needs its own kind instead. There is one kind for a single treatment plan's mean
and one for a difference of two plans, since a study that checked the difference does not check
each plan's mean.

### Use a summary in code, and plot a result

A summary is also data. It is a string, so `print` and `str` give the text. The same object holds
the parts, so a program does not need to parse the text.

```python
summary = result.summary()
name = list(result.estimates)[0]
row = summary.table().row(name)
print(summary.heading)
print(name, row["estimate"].value, row["95% CI"].text)
print(summary.note_codes)
```

`summary.table()` returns the estimates table, and `.row(name)` returns its cells, each with the raw
`value` and the printed `text`. `summary.facts` maps each header label to its value, and
`summary.notes` lists the notes. A check's `summary().reading` gives its one-sentence verdict.
`summary.as_html()` returns an escaped HTML fragment, and a notebook shows it without a call.
`repr(result)` is one short line that names the method and the estimates.

To see a result instead of reading its table, draw it with `cleverly.plotting`: the [plot
results](../examples/plots.ipynb) tutorial draws the estimate, the overlap and balance checks, and
the sensitivity analyses. The overlap plot (how the predicted chances of treatment compare between
the groups), the learner plot and the calibration plot also draw a fit of a treatment over time, one
panel per time point, and a fit that adds rows from outside the study, such as an external control
group. [Plotting](../api/plotting.md) lists which results each plot draws.

### Which interval each estimate carries

In a TMLE fit without clusters, the interval of the mean of a yes/no outcome is built on a scale
that keeps it between 0 and 1, and the interval of a difference of two such means is built from the
two means' intervals. For an outcome on a continuous scale, without clusters, the interval uses a t
distribution and is widened on the side where a skewed outcome makes misses more likely. Some
effects, such as the effect among the treated or the untreated (`att`, `atc`), keep the ordinary
interval centred on the estimate, and so does a fit with clusters. On a clustered fit the summary
also adds the `df` column below 40 clusters ([clusters](#clusters)).

Each estimate records which interval it carries in `estimate.interval_scale`: `"range_logit"` for a
mean kept inside its range, `"mover"` for a difference built from two means, `"skew_t"` for the
widened t interval and `"identity"` for the ordinary one. `estimate.effective_df` holds the t
distribution's degrees of freedom. The summary also adds a `df` column on a fit that averages the
estimates of each fold (`fold_evaluation=True`), where a fold is one of the groups of rows that
cross-fitting holds out in turn.

### Bootstrap options

`bootstrap_resampling="cluster"` resamples whole clusters, which `"auto"` does by itself when the
design declares them. Some fits refuse `n_bootstrap`, such as a treatment rule and the dose-response
curve, and the error says what to use instead. Collaborative TMLE, a variant that builds the
treatment model step by step, runs the bootstrap but prints its spread only as a diagnostic, because
that fit reports no interval. A fit of a treatment given once reports the percentile interval only
and refuses the bias-corrected `bootstrap_interval="bca"` ([intervals and
bands](settings.md#intervals-and-bands-inference)). The [inference
status](../technical-reference/inference.md#inference-status) table lists the fits that report no
interval. A treatment that repeats over time has its own bootstrap options: see [bootstrap for
sparse regimens](longitudinal.md#bootstrap-for-sparse-regimens).

### Clusters

A cluster is a group of rows that are not independent of each other, such as patients of one clinic,
declared when you describe the data ([data and study design](data-design.md)). With clusters, the
independent unit is the cluster, not the row. A cluster counts when at least one of its rows has a
nonzero weight. Below 40 such clusters, the interval uses a t distribution with the cluster count
minus 2 degrees of freedom, and the summary prints a `df` column.

Each kind of fit has a floor. Below it, the interval carries the `experimental` label, and below 3
clusters the fit reports no interval. The floor is 10 clusters for most fits of a treatment given
once and 20 for a treatment over time. A lower floor holds only for a contrast that a simulation
study measured at few clusters for that kind of fit, such as the average treatment effect of an
in-sample TMLE fit. [Few clusters](../technical-reference/inference.md#few-clusters) gives each
floor and the estimators it covers.

For TMLE and `DRTMLE` (a TMLE variant whose interval aims to stay valid when one of the two models
is wrong), these rules also apply to each reported subgroup (stratum). Below the floor, the fit's
bootstrap is published as `bootstrap sd` and a percentile range, and its summary labels both as
diagnostics ([few clusters](../technical-reference/inference.md#few-clusters)).

### Bands that cover several estimates at once

A fit that reports several estimates adds a simultaneous band by default. The band is built to cover
all the estimates at once in 95% of samples (at the default `alpha`), so it is wider than each
interval, and it prints as a "Simultaneous 95% bands" table. Use the band when you read several rows
together, for example when you pick the largest of several effects. Pass `simultaneous=False` to
turn it off; the continuous-dose example in [sensitivity analysis](sensitivity.md) does, because it
reads one estimate. The dose-response curve always reports its band and refuses `simultaneous=False`
([study route](../technical-reference/dose-response.md#study-route)).

Some fits build no band unless asked, and some build none at all, such as a clustered fit below 40
clusters, one that averages per-fold estimates, or a fit that withholds its interval;
`summary(detail="full")` then says why. [Evidence for the default
band](../technical-reference/inference.md#evidence-for-the-default-band) lists the fits that publish
no band, and [intervals and bands](settings.md#intervals-and-bands-inference) the settings that
change it.

By default, the band of a probability of a yes/no outcome is built on the log-odds scale, and the
band of a difference of two probabilities on a matching scale for the range -1 to 1, so its limits
stay inside the possible range and can sit unevenly around the estimate. The dose-response curve is
the exception: its band is built on the scale of the outcome itself, so a limit can fall below 0 or
above 1.

The band carries its own label in `result.simultaneous.support`, apart from the labels of the
intervals. A band reads `supported` only when a simulation study measured that band, with the same
settings, for that kind of fit. So the default band can read `experimental` while every interval
reads `supported`. The settings that build the band (`multiplier_kind`, `band_scale`,
`band_reference` and `n_multiplier`, which the [settings
reference](settings.md#intervals-and-bands-inference) describes) can change the band's label either
way. The label's reason names what was measured. [Simultaneous
bands](../technical-reference/inference.md#simultaneous-bands) states the rule.

### Result structure

A result (`CausalResult`) is a read-only mapping from each row name of the summary, such as `ate` or
`ey[1]`, to that row's estimate (`ParameterEstimate`).

```python
names = list(result.estimates)
table = result.to_frame()
covariance = result.covariance(names)
point = result[names[0]]
treated = next(alias for alias, key in result.parameter_keys.items() if key.value == 1)

print(point.psi, point.std_error, point.ci, point.pvalue)
```

`point.psi` is the point estimate, and `std_error`, `ci` and `pvalue` are the other three columns of
the summary table. `result.covariance(names)` gives how the estimates vary together.

A fit that withholds its interval, such as a collaborative fit with the default `"greedy"` strategy,
keeps its point estimate and refuses `std_error`, `ci` and `pvalue` with `CapabilityError`. The
[inference status](../technical-reference/inference.md#inference-status) table lists each case, and
[parametric per-arm
intervals](../technical-reference/collaborative-tmle.md#parametric-per-arm-intervals) gives the
conditions under which a collaborative `"oat"` fit reports one. The property
`result.inference_status` gives the status of the fit, and `point.supplies_inference` gives it for
one estimate. On a fit that withholds its interval, `point.plugin_std_error` and
`point.plugin_interval` give the standard error and interval that the usual formula would report,
which no theory or study shows to be valid for this fit. Each is a diagnostic, and neither is a
confidence statement.

`result.parameter_keys` maps each row name to a `ParameterKey`, which holds the parts of the name as
fields, such as the treatment level (`key.value`) and the subgroup (`key.stratum`). Select rows in
code by those fields, as the `treated` line above does, and not by parsing the printed name. On a
fit with baseline strata, `key.stratum == ("high",)` picks the estimate of the `high` stratum.

`result.estimate` is the sole `ParameterEstimate` only when the result holds one estimate, and
`result.psi()` is its numeric point value. With several, index the row name or pass a name to `psi`.

### Ratios, differences and tests of estimates

The result keeps how much each row moves each estimate, and so how the estimates vary together. A
difference, a ratio or another smooth function of the estimates therefore gets an interval without
refitting any model, and so does the band over the fit's estimates.

```python
if len(names) >= 2:
    difference = result.contrast(lambda values: values[0] - values[1], names[:2])
```

A contrast's interval follows the rules of [confidence intervals, the bootstrap and
bands](#confidence-intervals-the-bootstrap-and-bands). A difference of two yes/no means of one fit
is built from their two intervals, and any other contrast keeps the ordinary interval centred on it.
With clusters it uses the t distribution of the input with the fewest clusters. A contrast has no
band of its own.

A contrast inherits the inference status of its inputs. On a fit whose status supplies no inference,
the contrast refuses `ci` as its inputs do. The fit builds no simultaneous band.

Use `ratio` for the risk ratio or the odds ratio of two levels. The interval is on the log scale.
Use `wald_test(null=...)` to test an estimate against a value other than the default null. Pass
`transform=` to `contrast` to compute the interval on the scale of a monotone map, such as the log.
The example uses `ey[1]` and `ey[0]`, the mean outcome if everyone got treatment 1 and if everyone
got 0, which a `CounterfactualMean()` fit reports; use the names in your own `result.estimates`.

```python
from cleverly.inference import Transform

risk_ratio = result.ratio("ey[1]", "ey[0]")
test = risk_ratio.wald_test(null=1.5)
logged = result.contrast(
    lambda values: values[0] / values[1],
    ["ey[1]", "ey[0]"],
    transform=Transform.log(),
)
print(risk_ratio.ci, test.statistic, test.pvalue, logged.ci)
```

| call | what it reports | the null it tests by default |
| --- | --- | --- |
| `result.ratio(a, b)` | $\psi_a / \psi_b$, interval on the log scale | 1 |
| `result.ratio(a, b, kind="or")` | the ratio of the two odds. The outcome must be binary | 1 |
| `estimate.wald_test(null=c)` | the Wald statistic and its two-sided p-value at $c$ | the scale's default: 0, or 1 for a ratio |
| `result.contrast(fn, names, transform=f)` | $fn$ of the estimates, with the interval $f^{-1}(f(\hat h) \pm z\,se)$ | $f(0)$, so pass `wald_test(null=...)` when 0 is outside the domain of $f$ |

A contrast's default null is no difference, 0, carried to the scale of the transform. The log of a
ratio is not defined at 0, so a ratio built with `contrast` needs its own null, such as
`wald_test(null=1)`.

[Transformed parameters](../technical-reference/inference.md#transformed-parameters) gives the
arithmetic.

### What a loaded result can redo

Print `restored.replayability.summary()` to see what a loaded result can redo: show its stored
checks, re-estimate from its stored models, refit the models, or rerun on changed data. A note names
any part the result lacks. Read `result.diagnostics.capabilities` before you run a check on a loaded
file. Each row declares what it needs to redo, and `available` is `False` when the file cannot
supply it. The row names the missing part (its slot), so you learn the answer before the check runs.

A result can also hold an estimator that was copied or changed after the fit into a setup the
package refuses, such as a cross-fitted continuous outcome without its known range (`q_bounds`).
Such a result still gives its point estimates and the checks that rerun the correction step on its
stored models, such as the curve over caps on extreme weights. Checks that fit new models, such as
`refute()` and `simulated_confounding()`, read `unavailable`, and `estimator.refit()` raises
`CapabilityError` before it fits anything. Fit the analysis again under a setup the package accepts.
[Replay-only
unavailability](../technical-reference/scope-and-refusals.md#replay-only-unavailability) lists each
refused setup and the code the result records, and [truncation
stability](../technical-reference/validation-methods.md#truncation-stability) covers the weight-cap
curve of a guarded `DRTMLE` fit.

The saved file carries the answers of earlier checks (the assessment cache). A result you derive
with `dataclasses.replace` does not. The cache key records the check and its arguments, and it
records nothing about the result that answered them, so a derived result starts with an empty cache
of its own.

A curve from `fit.effect_curve()` on a [conditional effect](conditional-effects.md) saves and loads
the same way as a result. The loaded curve keeps its coefficients, their joint covariance, the
declared modifiers and basis, and any band, so `curve.predict(rows, interval=True)` evaluates it at
new rows after loading.

### Limit the memory of saved answers

A result keeps every check answer it has computed in its assessment cache, so a repeated call
returns at once. Most answers are a few kilobytes. The inputs to the missing-confounder bound that
`result.sensitivity.elements()` returns, and that `run_all()` and `assess()` compute along the way,
hold four numbers per row. At a million rows that is about 32 MB for each estimate and each choice
of `nu2_estimator`, the estimator of the bound's weight term ([sensitivity
analysis](sensitivity.md)). A derived risk ratio holds one number per row. Many distinct requests on
a large fit can therefore add up.

Set `result.assessment_cache.max_bytes` to a number of bytes to cap the memory of all saved answers.
When the total passes the cap, the cache drops the answers it can compute again exactly from the
fitted result, least recently used first, and a later call recomputes a dropped answer with
bit-for-bit the same numbers. It never drops an answer that refits a model, draws a random seed of
its own, fits a learner or runs work in parallel, because computing that again could give different
numbers. Those answers count toward the cap and can hold the cache above it.
`result.assessment_cache.info()` lists each answer with its size and whether the cap may drop it,
and `result.assessment_cache.evict()` drops every answer it may drop now.

There is no cap by default. A cap trades memory for the time to recompute, and `save()` writes only
the answers that survive it. `result.assessment_cache.clear()` removes every answer, including the
refits; a repeated unseeded refit then draws a new seed, as it would after any removal.
