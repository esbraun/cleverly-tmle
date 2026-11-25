# Read and save results

This page reads a fitted result: its summary, its intervals, its support label, and how to save
it. [Check a fitted result](assessment.md), [Sensitivity analysis](sensitivity.md) and
[Try to break the result](refutation.md) cover the checks you run on it, and their reports use the
same layout.

## Reading a summary

Every result and every check has a `summary()` method. Print it to see the answer first and the
setup after it. The estimate of an effect looks like this:

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
```

The block above the table says what was asked and what was fitted: the question, the outcome, the
columns adjusted for, the method, the number of rows, and the models. The table holds the
numbers. The `estimate` is the effect on the scale of the outcome. The `std. error` measures how
much the estimate would move from sample to sample. The `95% CI` is the interval built from it,
and the `p-value` tests whether the effect is zero. A row with no interval says why in a note.

An interval need not be centred on its estimate, and a row can carry none.
[Confidence intervals, the bootstrap and bands](#confidence-intervals-the-bootstrap-and-bands)
says which interval each row gets and how to change it.

Numbered notes follow the table, and they appear only when something changes how to read a number.
A note can say that an estimate is experimental because no validation study covers that kind of
fit, that the models were fitted and evaluated on the same rows, or that the cap on extreme weights
changed some units. A warning prints before an ordinary note. A summary with no notes has nothing
to flag.

`summary()` prints the short form. `summary(detail="full")` adds sections after the notes: the
assumptions, written out one per line, the settings that change how to read the numbers, and the
description of the study. The short form always has the header and the table.

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

`summary.table()` returns the estimates table, and `.row(name)` returns its cells, each with the
raw `value` and the printed `text`. `summary.facts` maps each header label to its value, and
`summary.notes` lists the notes. A check's `summary().reading` gives its one-sentence verdict.
`summary.as_html()` returns an escaped HTML fragment, and a notebook shows it without a call.
`repr(result)` is one short line that names the method and the estimates.

To see a result instead of reading its table, draw it with `cleverly.plotting`: the
[plot results](../examples/plots.ipynb) tutorial draws the estimate, the overlap and balance checks,
and the sensitivity analyses. The overlap, learner and calibration plots also draw a fit over time, one
panel per time point, and a fit that borrows external rows; [Plotting](../api/plotting.md) lists
which results each plot draws.

## Confidence intervals, the bootstrap and bands

A result reports a 95% confidence interval for each estimate by default. It is built from the
standard error, which comes from how much each row moves the estimate, and it is not always the
estimate plus or minus two standard errors. In a TMLE fit without clusters, the interval of the
mean of a yes/no outcome is built on a scale that keeps it between 0 and 1, and the interval of a
difference of two such means is built from the two means' intervals. For an outcome on a
continuous scale, without clusters, the interval uses a t distribution and is widened on the side
where a skewed outcome makes misses more likely. Some effects, such as the effect among the
treated or the untreated (`att`, `atc`), keep the ordinary interval centred on the estimate, and
so does a fit with clusters.

Each estimate records which interval it carries. `estimate.interval_scale` is `"range_logit"` for
a mean kept inside its range, `"mover"` for a difference built from two means, `"skew_t"` for the
widened t interval and `"identity"` for the ordinary one. `estimate.effective_df` holds the t
distribution's degrees of freedom. The summary adds a `df` column when an estimate's t has fewer
than 250 degrees of freedom, since above that it is close to the normal distribution. It also
adds one on a clustered fit below 40 clusters, or one that averages per-fold estimates
(`fold_evaluation=True`) ([clusters](#clusters)). A treatment group that shows only one outcome
value, or an estimate at exactly 0 or 100 percent, gets no interval, and neither do the effects
that use it. [Small-sample scale](../technical-reference/inference.md#small-sample-scale) gives
each formula and every case.

### Change the level or add a bootstrap

Pass `alpha` to change the level: `alpha=0.1` gives 90% intervals. The support label does not
change with the level, and the studies behind a `supported` label checked 95% intervals. Pass
`n_bootstrap` to add a bootstrap, which refits every model on resampled copies of the data. Here
`study` is the `CausalStudy` you built.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE

bootstrapped = study.estimate(
    ATE(),
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    cross_fit=False,
    random_state=7,
    alpha=0.1,
    n_bootstrap=20,
)
print(bootstrapped.summary())
print(bootstrapped["ate"].bootstrap.support.label)
```

The summary then shows a "90% CI" column and a "Bootstrap" table with the bootstrap standard
error and a percentile interval (the middle 90% of the replicate estimates). The bootstrap is
reported beside the usual interval and does not replace it. Twenty replicates keep the example
fast; use a few hundred in an analysis, and
[count the fits](methods-learners.md#count-the-model-fits-before-a-long-run) first, since each
replicate refits every model. `bootstrap_resampling="cluster"` resamples whole clusters, which
`"auto"` does by itself when the design declares them.

The bootstrap interval carries its own label in `estimate.bootstrap.support`, apart from the
usual interval. A simulation study (see [the support label](#read-the-support-label)) covers it for
a few fits, such as the fit above of the average effect on a yes/no outcome, which reads
`supported` ([full-refit bootstrap study](../technical-reference/method-evidence/full-refit-bootstrap-and-derived-contrasts.md)).
That fit is in sample (`cross_fit=False`): its models are fitted and scored on the same rows. For
a treatment given once with cross-fitting (`cross_fit=True`, the default) the bootstrap reads
`experimental`, and the label's reason says why.

Some fits refuse `n_bootstrap`, such as a treatment rule and the dose-response curve, and the error
says what to use instead. Collaborative TMLE runs the bootstrap but prints its spread as a
diagnostic, not as an interval, because that fit reports no interval. A fit of a treatment given
once reports the percentile interval only and refuses the bias-corrected
`bootstrap_interval="bca"` ([intervals and bands](settings.md#intervals-and-bands-inference)).
Fits over time have their own bootstrap options: see [bootstrap for sparse regimens](longitudinal.md#bootstrap-for-sparse-regimens).

### Clusters

With clusters declared in the design, the independent unit is the cluster, not the row. Below 40
clusters that hold at least one row with a nonzero weight, the interval uses a t distribution with the cluster count minus 2
degrees of freedom, and the summary prints a `df` column.

Each kind of fit has a floor: below it, the interval carries the `experimental` label, and below 3
clusters the fit reports no interval. The floor is 10 clusters for most fits of a treatment given once
and 20 for fits over time. It is lower only for the one contrast that a simulation study measured at
few clusters for that kind of fit, such as the average treatment effect of an in-sample TMLE fit.
For TMLE and DR-TMLE these rules also apply to each reported stratum.
[Few clusters](../technical-reference/inference.md#few-clusters) gives each floor and the
estimators it covers.

### Bands that cover several estimates at once

A fit that reports several estimates adds a simultaneous band by default. The band is built to
cover all the estimates at once in 95% of samples (at the default `alpha`), so it is wider than
each interval, and it prints as a "Simultaneous 95% bands" table. Use the band when you read several rows together, for example
when you pick the largest of several effects. Pass `simultaneous=False` to turn it off; the
continuous-dose example in [sensitivity analysis](sensitivity.md) does, because it reads one
estimate. The dose-response curve always reports
its band and refuses `simultaneous=False`
([study route](../technical-reference/dose-response.md#study-route)).

Some fits build no band unless asked, and some build none at all, such as a clustered fit below
40 clusters, one that averages per-fold estimates, or a fit that withholds its interval;
`summary(detail="full")` then says why.
[Evidence for the default band](../technical-reference/inference.md#evidence-for-the-default-band)
lists the fits that publish no band, and [intervals and bands](settings.md#intervals-and-bands-inference)
the settings that change it.

By default, the band of a probability of a yes/no outcome is built on the log-odds scale, and the
band of a difference of two on the matching scale for differences, so its limits stay inside the
possible range and can sit unevenly around the estimate. The dose-response curve is the exception:
its band is built on the scale of the outcome itself, so a limit can fall below 0 or above 1.

The band carries its own label in `result.simultaneous.support`, apart from the labels of the
intervals. A band reads `supported` only when a simulation study measured that band, with the same
settings, for that kind of fit. So the default band can read `experimental` while every interval
reads `supported`, and changing `multiplier_kind`, `band_scale`, `band_reference` or
`n_multiplier` can change the band's label either way. The label's reason names what was
measured. [Simultaneous bands](../technical-reference/inference.md#simultaneous-bands) states the
rule.

## Read the support label

Every estimate carries a support label. The label says whether a simulation study that checks this
method covers the configuration that produced the estimate. You do not request it, and no argument
turns it off.

A `supported` label means that a simulation study checked this kind of fit and passed; the label
names the study and states its limits. An `experimental` label means that the package computes the
estimate but no such study covers this kind of fit, or a stated condition applies, so the label
makes no coverage claim.

An `experimental` label does not mean the estimate is wrong. It means the package has no
registered evidence for the interval, the band, or the derived value you read. A configuration
that no row covers also reads `experimental`. Where a row, a status, or a condition makes the
estimate experimental, the label carries the reason.

The `evidence` of a `supported` estimate holds study slugs. The `evidence` of an experimental
estimate holds roadmap ids, such as `X54`, that name the missing evidence.

A fit with fewer clusters than the interval floor shows its interval under the experimental
label. Its bootstrap is published as `bootstrap sd` and a percentile range, and its summary
labels both as diagnostics. A fit that withholds its
interval, such as a collaborative fit with the default `"greedy"` strategy, keeps its point
estimate and refuses `ci`. The
[inference status](../technical-reference/inference.md#inference-status) table lists each case.

## Save and load a result

```python
from cleverly import load

result.save("analysis.joblib")
restored = load("analysis.joblib")
assert restored.parameter_keys == result.parameter_keys
assert type(restored.method) is type(result.method)
assert type(restored.method.models.outcome_learner) is type(result.method.models.outcome_learner)
```

The artifact contains the complete result graph, including nuisance estimator templates, so
successfully restored results retain refit-based assessment. `save()` writes the graph with
`cloudpickle`. It saves an importable function or class by reference. It saves a lambda, a
closure, and an object defined in a notebook or in `__main__` by value. A result therefore keeps
the rules, policy densities, and learners that you wrote inline. A by-value object loads only
under the Python minor version that saved it.

The loader is `joblib.load`, which uses pickle. Loading can execute arbitrary code, so load only
trusted artifacts. Use an environment with the same Python minor version and compatible sklearn
and third-party estimator versions.

`result.save()` records the `cleverly` version that wrote the artifact. When another version saved
it, `cleverly.load()` warns with `VersionMismatchWarning` and loads the artifact as saved. The load
runs no migration. The statuses, reports, and cached assessments of such an artifact can differ
from what this version computes, so fit the analysis again with this version. Development
snapshots between two releases share one version string, so the check does not see a change
between them.

## Go deeper

### Check your own configuration

A support label says whether a simulation study covers your kind of fit. When the label reads
`experimental`, or your learners, sample size or kind of outcome differ from the studied ones, you
can run a small study of your own. A coverage study draws many data sets from a simulated process
whose true effect you know, runs your analysis on each, and counts how often the interval holds
the truth (`CoverageStudy`). One fit cannot tell you this, because coverage describes repeated
samples ([coverage studies](../technical-reference/validation-methods.md#coverage-studies)).

```python
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import TMLE
from cleverly.validation import CoverageStudy


def my_law(n, seed):
    rng = np.random.default_rng(seed)
    w1, w2 = rng.normal(size=n), rng.normal(size=n)
    a = rng.binomial(1, expit(0.5 * w1))
    y = 1.0 * a + w1 + 0.5 * w2 + rng.normal(size=n)
    frame = pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2})
    return frame, {"ate": 1.0}


coverage = CoverageStudy(
    dgp=my_law,
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

`my_law` is the part you write. It takes a row count and a seed, and returns a data frame and the
true value of each quantity you estimate. `CoverageStudy` takes a function that builds a fresh
estimator from `cleverly.estimators`, not a `CausalStudy`, and passes the column names through
`fit_kwargs`. Give it the learners and the `cross_fit` setting of your own fit.

The summary has three numbers to read. `coverage` is the share of intervals that held the truth,
which should be near 95%. `bias` is the mean error of the estimates, and `summary(detail="full")`
adds it times the square root of the row count (root-n bias). `SE ratio` divides the average
reported standard error by the actual spread of the estimates (`MC SD`), and a value near 1 means
the reported uncertainty is honest. Monte Carlo error is the noise that comes from running a
finite number of draws. The line under the table flags coverage below 95% by more than three Monte
Carlo standard errors, or a bias more than three of its standard errors from zero.

The numbers hold only for the process you simulated, so make it resemble your data: the row count,
the covariates, the treated share and the kind of outcome. The truth that `my_law` returns must be
the quantity your fit estimates, on the same population. Fifty draws keep the example fast, but
at 50 draws the coverage has a Monte Carlo standard error of about 3 points, so it cannot tell 92%
from 95%; an analysis needs several hundred. Each draw is one full fit: `n_jobs=` runs draws in
parallel, and `checkpoint=` lets a stopped study resume. A draw whose fit fails stays in the study
as a failure and is not dropped.

Your study does not change the support label. Labels come from fixed rows in the package, each
backed by a study in the
[validation grid](../technical-reference/method-evidence/validation-grid.md), and `CoverageStudy`
only reports the label its draws carried. Report your numbers beside the label as your own check.
If you run your own loop of fits, build one `ReplicationRecord` per fit and pass the records to
`summarize_replications`, which returns the same numbers for one estimate.

Without writing a simulator, the plasmode test on [Try to break the result](refutation.md) reads
the same three numbers on simulated data built from your fit: covariates resampled from your rows,
a treatment drawn from a model fitted to them, and an outcome you declare.

### Where the label appears

| surface | where the label appears |
| --- | --- |
| `repr(estimate)` | the marker `[experimental]` after the estimate |
| `result.summary()` | a `support` column, only when an estimate is experimental, and an `experimental` note below the table that says in plain words why. `summary(detail="full")` prints each reason under Settings |
| `estimate.support` | a `Support` record with `label`, `reason`, `evidence`, `row` and `limits` |
| `estimate.to_dict()` and `result.to_frame()` | `support` on every row. An experimental row adds `support_reason` and `support_evidence`. A row with limits adds `support_limits` |
| `estimate.bootstrap.support` and `bootstrap_support` | the label of the percentile interval, which a row can license apart from the Wald interval |
| `result.simultaneous.support` | the label of the band |
| `result.config.support_key` | the key the fit was classified under. `result.save()` and `cleverly.load()` keep it with every label |

A derived estimate, such as a ratio or a contrast, is `supported` only when its row lists the
`derived` kind and every input is `supported`. A restricted mean from a grid of check times needs
its own kind instead, one for a single plan's mean and one for a difference of two plans, since a
study that checked the difference does not check each plan's mean.

### Result structure

`CausalResult` is a read-only mapping from stable aliases to `ParameterEstimate` objects.

```python
names = list(result.estimates)
table = result.to_frame()
covariance = result.covariance(names)
point = result[names[0]]
treated = next(alias for alias, key in result.parameter_keys.items() if key.value == 1)

print(point.psi, point.std_error, point.ci, point.pvalue)
```

A fit that withholds its interval, such as a collaborative fit with the default `"greedy"`
strategy, keeps its point estimate and refuses `std_error`, `ci` and `pvalue` with
`CapabilityError`. The [inference status](../technical-reference/inference.md#inference-status)
table lists each case, and
[parametric per-arm intervals](../technical-reference/collaborative-tmle.md#parametric-per-arm-intervals)
gives the conditions under which a collaborative `"oat"` fit reports one. The property `result.inference_status` gives the
status of the fit, and `point.supplies_inference` gives it for one estimate. On a fit that
withholds its interval, read `point.plugin_std_error` and `point.plugin_interval`. Each is a
diagnostic of the reported curve, and neither is a confidence statement.

`result.parameter_keys` maps each alias to a structured `ParameterKey`. Use those fields for
programmatic selection, as the `treated` line above does with `key.value`. On a fit with baseline
strata, `key.stratum == ("high",)` picks the estimate of the `high` stratum the same way. Display
aliases are not a serialization format.

`result.estimate` is the sole `ParameterEstimate` only when the result holds one parameter, and
`result.psi()` is its numeric point value. With several parameters, index the alias or pass a name
to `psi`.

### Contrasts and simultaneous inference

The result retains influence curves and their joint covariance, so smooth contrasts and
simultaneous bands do not refit nuisance models.

```python
if len(names) >= 2:
    difference = result.contrast(lambda values: values[0] - values[1], names[:2])
```

A contrast's interval follows the rules of
[confidence intervals, the bootstrap and bands](#confidence-intervals-the-bootstrap-and-bands). A
difference of two yes/no means of one fit is built from their two intervals, and any other contrast
keeps the ordinary interval centred on it. With clusters it uses the t distribution of the input
with the fewest clusters. A contrast has no band of its own.

A contrast inherits the inference status of its inputs. On a fit whose status supplies no
inference, the contrast refuses `ci` as its inputs do. The fit builds no simultaneous band.

Use `ratio` for the risk ratio or the odds ratio of two levels. The interval is on the log scale.
Use `wald_test(null=...)` to test an estimate against a value other than the default null. Pass
`transform=` to `contrast` to compute the interval on the scale of a monotone map.

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

[Transformed parameters](../technical-reference/inference.md#transformed-parameters) gives the
arithmetic.

### Persistence and replayability

A curve from `fit.effect_curve()` on a
[conditional effect](conditional-effects.md) saves and loads the same way. The loaded curve keeps
its coefficients, their joint covariance, the declared modifiers and basis, and any band, so
`curve.predict(rows, interval=True)` evaluates it at new rows after loading.

Print `restored.replayability.summary()` to see what a loaded result can redo: show its stored
checks, re-estimate from its stored models, refit the models, or rerun on changed data. A note
names any part the result lacks. Read `result.diagnostics.capabilities` before you run an
operation on a loaded artifact. Each row declares the replay it needs, and `available` is `False`
when the artifact cannot supply it. The row says which slot is missing, so you learn the answer
before the operation runs.

A result can hold a copied or modified estimator whose configuration the package refuses before a
refit. A stratified fold policy, a cross-fitted continuous outcome with `q_bounds=None`, and a
`"greedy"` C-TMLE fit on clustered data are examples. The `refit_nuisances` slot runs every check that
`refit()` runs before a learner, so a design refusal of an estimator subclass counts too. The table
gives what such a result keeps.

| operation | on such a result |
| --- | --- |
| point estimates and cached-nuisance retargets, such as `truncation_curve()` | run. `replayability.retarget_cached_nuisances` is `True`. A guarded DR-TMLE curve also runs. It retargets the cached primary nuisances and refits the reduced regressions inside that step. Its row reports the `refit` cost, but it does not require the `refit_nuisances` slot of the outer estimator. The curve refuses a reduction setting that a fit refuses, such as `targeting="one_step"` with `reduced_crossfit="foldwise"`. It also refuses a `guard`, `reduction` or `reduced_crossfit` other than the fitted one |
| refits, such as `refute()` and `simulated_confounding()` | `unavailable`. `replayability.refit_nuisances` is `False`, with the code `point_replay_refit_configuration` |
| `estimator.refit()` | raises `CapabilityError` before any learner |

A new fit under a supported configuration restores the refits.

The saved artifact carries the assessment cache. A result you derive with `dataclasses.replace`
does not. The cache key records the operation and its arguments, and it records nothing about the
result that answered them, so a derived result starts with an empty cache of its own.

### Limit the memory of saved answers

A result keeps every assessment answer it has computed, so a repeated call returns at once. Most
answers are a few kilobytes. The omitted-confounder elements that `elements()` returns, and that
`run_all()` and `assess()` compute along the way, hold four numbers per row: about 32 MB for each
estimand and `nu2_estimator` at a million rows. A derived risk ratio holds one number per row. Many
distinct requests on a large fit can therefore add up.

Set `result.assessment_cache.max_bytes` to a number of bytes to cap the memory of all saved
answers. When the total passes the cap, the cache drops the answers it can compute again exactly
from the fitted result, least recently used first, and a later call recomputes a dropped answer
with bit-for-bit the same numbers. It never drops an answer that refits a model, draws a random
seed of its own, fits a learner or runs work in parallel, because computing that again could give
different numbers. Those answers count toward the cap and can hold the cache above it. `result.assessment_cache.info()` lists each answer with its size and whether
the cap may drop it, and `result.assessment_cache.evict()` drops every answer it may drop now.

There is no cap by default. A cap trades memory for the time to recompute, and `save()` writes only
the answers that survive it. `result.assessment_cache.clear()` removes every answer, including the
refits; a repeated unseeded refit then draws a new seed, as it would after any removal.
