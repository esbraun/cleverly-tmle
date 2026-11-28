# Confidence intervals, the bootstrap and bands

Inference is the step that says how uncertain an estimate is. This page covers every interval a fit
reports, for any design: the interval from the standard error, the bootstrap, bands over several
estimates, and intervals of ratios and differences. Whether a simulation study has checked an
interval is the job of the [support label](reading-output.md#read-the-support-label).

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

## Change the level or add a bootstrap

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
fits](run-time-and-memory.md#count-the-model-fits-before-a-long-run) first, since each replicate refits
every model.

The bootstrap interval carries its own label, in `bootstrapped["ate"].bootstrap.support`. A
simulation study covers it for a few fits. It reads `supported` on a yes/no outcome fitted on every
row (`cross_fit=False`), the setup the [full-refit bootstrap
study](../technical-reference/method-evidence/full-refit-bootstrap-and-derived-contrasts.md)
checked, and `experimental` on a continuous outcome such as the quickstart's, as note 2 says. For a
treatment decided once (not at several times) with cross-fitting, the default, it reads
`experimental`, and the label's reason says why. [Bootstrap options](#bootstrap-options) covers
clustered resampling and the fits that refuse a bootstrap.

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

Without writing a simulator, the plasmode test of [refutation](refutation.md) reads the
same three numbers on simulated data built from your fit: covariates resampled from your rows, a
treatment drawn from a model fitted to them, and an outcome you declare.

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
sparse regimens](#bootstrap-a-fit-over-time).

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

[Flexible models and shared teams](../examples/cross-fitting.ipynb) compares the interval with
patients and with navigator teams as the units.

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

### Ratios, differences and tests of estimates

The result keeps how much each row moves each estimate, and so how the estimates vary together. A
difference, a ratio or another smooth function of the estimates therefore gets an interval without
refitting any model, and so does the band over the fit's estimates.

```python
names = list(result.estimates)
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

### Bootstrap a fit over time

Pass `n_bootstrap=` to refit the whole estimator on resamples. Each estimate then carries a
percentile interval beside its influence-curve interval.
[The full-refit bootstrap](../technical-reference/longitudinal-tmle.md#the-full-refit-bootstrap)
states what each replicate resamples and refits. The percentile interval carries the support
label of the fit, so a fit no registered study covers says under its summary that the bootstrap
interval is experimental.

On a clustered fit, an estimate with fewer clusters than its interval floor prints a
`bootstrap SD (diagnostic)` column and a percentile range, both diagnostics.
[Clustered units and survey weights](longitudinal.md#clustered-units-and-survey-weights) gives the floors for a fit
like the one there and links the full table. At or above the floor the bootstrap prints a percentile
interval under the experimental label, because no simulation study has checked a clustered
bootstrap's coverage ([clusters](../technical-reference/inference.md#clusters)).

Three registered studies measured this feature at three to five decisions: [binary `LTMLE`](../technical-reference/method-evidence/long-horizon-longitudinal-tmle.md), [three-level `LTMLE`](../technical-reference/method-evidence/long-horizon-categorical-longitudinal-tmle.md) and [`SDR`](../technical-reference/method-evidence/long-horizon-sequentially-doubly-robust-estimation.md).
The label of a fit follows the support rows `G46`. A fit outside those rows prints `experimental`, and the label makes no coverage claim.

Use the refit bootstrap for `LTMLE` or `SDR` when few units follow a regimen, or when the
influence curve has heavy tails. The Wald interval reads its variance from those few rows. The
interval then under-covers, because the variance estimate is noisy and the sampling distribution is
skewed.

A Wald fit warns with `SparseInfluenceWarning` when this happens. A fit with a percentile
bootstrap warns too, because the percentile interval keeps the skew. A fit with BCa does not warn.
The bootstrap is opt in. A fit never turns it on.

A study fit computes each plan's estimate and each difference between plans, then reports only the
ones you asked for, and it checks all of them for this warning. So a fit of the differences
(`RegimeContrast`) can warn about one plan's estimate, which its table does not show. The warning
then says so, and names each reported difference whose interval is built from that estimate's
interval. It does not name a restricted mean that adds up the flagged estimate.

A policy that shifts a continuous dose needs another interval, because the bootstrap does not
help there. [When few customers carry the interval](longitudinal-plans.md#when-few-customers-carry-the-interval)
gives it.

| option | what it does | label today |
| --- | --- | --- |
| `n_bootstrap=B` | refits the whole estimator on `B` resamples and attaches an interval to each estimate | `supported` only for a covered fit, otherwise `experimental` |
| `bootstrap_interval="percentile"` | takes the quantiles of the `B` draws. This is the default | as above |
| `bootstrap_interval="bca"` | reads bias-corrected and accelerated limits from the draws. It needs `n_bootstrap` | `supported` for a single-fold fit that matches a long-horizon row, within the limits below, otherwise `experimental` |
| `simultaneous=True` | adds a studentized bootstrap sup-t band to the fit, for two or more estimates | `supported` for a three-level `LTMLE` fit at five decisions that matches `G46.tmle.categorical_k5`, otherwise `experimental` |
| `variance="policy_weighted"` | for a continuous-dose policy, reports the larger of the usual standard error and one read at each row's policy dose. Not combined with `n_bootstrap` | `supported` where a support row licenses it ([policy-weighted variance study](../technical-reference/method-evidence/mtp-policy-variance.md)), otherwise `experimental` |
| `result.diagnostics.influence()` | reports how few rows carry each influence curve. Call it directly: `run_all()` does not include it | a diagnostic, with no label |

The label checks the plan family of the fit. It does not check $n$, the learners or the quantity.
The long-horizon studies gate the BCa interval at $n$ from 4,000 to 128,000, and the first note of
each row lists its sizes. At three and four decisions, BCa is `supported` for static and dynamic
plans only. A known stochastic policy at those decisions reads `experimental`.

`LTMLE` and `SDR` take each of these options, and `IPSI` takes all of them except `variance`.
`CausalStudy.estimate()` takes the same keywords and passes them to its estimator, with
`cross_fit=False` in place of `n_folds=1`.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_longitudinal
from cleverly.longitudinal import LTMLE

frame, _ = make_longitudinal(n=300, seed=0)
result = LTMLE(
    {"always": 1, "never": 0},
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    censoring_learner=LogisticRegression(max_iter=1000),
    n_folds=1,
    n_bootstrap=20,
    bootstrap_interval="bca",
    simultaneous=True,
    random_state=0,
).fit(
    frame,
    outcome="Y",
    treatment=["A1", "A2"],
    baseline=["W1", "W2"],
    time_varying=[[], ["L2"]],
    censoring=["C1", "C2"],
)
print(result.summary())
print(result.bootstrap_band.critical_value)
```

The example uses 20 resamples to run fast. Use at least 199 in an analysis. With `B` resamples a
BCa limit is an order statistic of the draws, so a small `B` gives a coarse limit.

To see whether the resampled estimates look normal, plot them:

```python
import cleverly.plotting

fig = cleverly.plotting.bootstrap(result, name="ate_regimen[never vs always]")
```

The histogram shows the 20 resampled differences, the curve is a normal density with their mean
and spread for comparison, and the vertical lines mark the reported estimate and the BCa limits.
When the draws sit off-centre around the estimate or trail off to one side, the BCa limits move
away from the percentile limits, as "Percentile or BCa" explains.

**Percentile or BCa.** The percentile interval takes the 2.5% and 97.5% quantiles of the draws. It
assumes the draws are symmetric about the estimate. The BCa interval shifts both limits by a bias
correction, from the share of draws below the estimate, and by an acceleration, from the skew of
the influence curve. Choose BCa when `diagnostics.influence()` reports a large skewness. For a
symmetric curve the two intervals are close.

**The sup-t band.** Pass `simultaneous=True` with `n_bootstrap` to get `result.bootstrap_band`.
Each replicate records its own standard error. The band takes the largest studentized deviation
across the estimates in each replicate. Its critical value is a high order statistic of those
maxima.

The band is $\hat\psi_j \pm c^* \widehat{SE}_j$ for each estimate. The multiplier band in
`result.simultaneous` does not change. The summary lists both. A `CausalStudy` fit computes each
plan's estimate and each difference, and its band covers only the estimates its table reports: a
fit of the differences (`RegimeContrast`) bands the differences, and a fit of the plans' estimates
(`RegimeMean`) bands those estimates. When the fit computed more than its table reports, the band
reads `experimental`, because no simulation study measures a band over part of what the fit
computed.

**The influence report.** `result.diagnostics.influence()` returns one row for each estimate.

| column | meaning |
| --- | --- |
| `kish` | the effective number of rows that carry the curve, $n / \hat\kappa$. With `id=` the unit is the cluster, and $n$ counts clusters |
| `skewness`, `kurtosis` | the third and fourth standardized moments of the curve. A normal curve has kurtosis 3 |
| `variance_rel_sd` | the relative standard deviation of the variance estimate |
| `edgeworth_loss` | the Wald coverage shortfall that the second-order Edgeworth term predicts. A positive value is a loss and a negative value is a predicted excess. The flag does not read this column |
| `bca_z0`, `bca_acceleration` | the BCa bias correction and acceleration. They are `nan` without a bootstrap |

The warning fires when `variance_rel_sd` is above 0.125, which is close to a Kish count of 64. The
warning lists the relative SD of each flagged row, with its Kish count and Edgeworth loss. A
flagged row can show a negative loss, because the skew term can predict an excess while the noisy
variance estimate still triggers the flag. The
pilot runs set that threshold, so it is not a pre-registered rule.
A cross-fitted fit, a hazard-model fit and an isotonic fit warn too, but the message does not
advise the bootstrap. For the cross-fitted fit the bootstrap over-covers, and the roadmap owner
`X20-bootstrap` in the roadmap holds that result.

To see how few rows carry the standard error, plot the influence values:

```python
fig = cleverly.plotting.influence(result, name="ate_regimen[never vs always]")
```

The plot ranks the rows by how much each one moves the estimate and draws the share of the
variance that the top rows carry. A curve on the diagonal means every row counts equally, and a
steep rise means a few rows carry the interval. `cleverly.plotting.notes(fig)` gives the Kish
count, the effective number of rows.

**Cost.** Each resample refits every nuisance. A fit of 64,000 rows and five decisions takes about
0.2 seconds for each resample on the fast path, a development measurement that no committed
artifact records yet. The long-horizon studies declare `B = 199`, so one such fit costs about 40 seconds of
bootstrap time.

The fast path fits the distinct rows with the resample counts as weights. It
applies to an in-sample, unclustered, unweighted fit with declared treatment and censoring
mechanisms, categorical treatment, and learners that declare `frequency_exact`. Any other fit
refits each materialized resample, which costs much more. Set `n_jobs` to run resamples in
parallel.

**Evidence.** The three long-horizon studies fit the bootstrap with `B = 199`. Each study gates a BCa interval only where its Wald interval does not gate, that is where $n$ is below twice the exact-law tail size. The three-level `LTMLE` study gates the sup-t band over six plans at five decisions.

The `SDR` bootstrap band, the cross-fitted bootstrap and an MSM fit stay `experimental`. The R packages `ltmle` and `lmtp` give the same estimates and standard errors as `cleverly`, to within $10^{-6}$, on the cells the studies compare exactly. Neither package fits a bootstrap, so the BCa interval and the band are checked against the known truth only. No study gates the warning threshold.

### Bootstrap a time-to-event fit

Pass `n_bootstrap=` to refit the whole estimator on resamples of the data. Each estimate then
carries a percentile interval beside its usual interval. The example fits the risk of an event by
two time points under a treatment decided at each one, the panel of [treatments over time with an
event outcome](survival.md#treatments-over-time-with-an-event-outcome).

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, LongitudinalTreatment, RegimeMean
from cleverly.datasets import make_longitudinal_survival

panel, _ = make_longitudinal_survival(n=2_000, seed=12)
panel_study = CausalStudy(
    panel,
    design=LongitudinalTreatment(
        outcome=("Y1", "Y2"),
        treatment=("A1", "A2"),
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    ),
)
bootstrapped = panel_study.estimate(
    RegimeMean({"always": 1, "never": 0}, horizons=(1, 2)),
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    n_bootstrap=200,
    random_state=0,
)
print(bootstrapped.summary())
```

`ratio` and `rmst` carry the bootstrap of their inputs. Unless a simulation study covers the fit,
a note under the summary says the bootstrap interval is experimental. Report the usual interval
as the main result. When the fit warns with `SparseInfluenceWarning` that an estimate rests on
few people, show the bootstrap interval beside it.

A fit with exact times, `grid="continuous"`, takes `n_bootstrap=` too. Each resample refits every
model and the risks, so it costs about as long as the fit itself. A person drawn twice into a
resample stays in one group when the fit splits the rows to fit predictions on the other rows, so
a copy never predicts its twin. The bootstrap interval of each risk sits in `estimate.bootstrap`
and in the `bootstrap_ci_lower` and `bootstrap_ci_upper` columns of `result.to_frame()`;
`bootstrap_interval="bca"` gives the BCa interval, which corrects the
percentile interval for skew. If more than 1% of the refits fail, the fit warns, drops every
bootstrap interval and keeps the usual ones. Below that share the bootstrap intervals use only the
refits that worked, without a warning; the summary's bootstrap note gives the count, and
`result.bootstrap_failures` says why the others failed.

On these fits the bootstrap interval always reads
`experimental`. With the default cross-fitting no theory yet shows that the bootstrap is valid,
and for a fit without cross-fitting the conditions of the theory that would cover it have not
yet been checked. [The full-refit bootstrap](../technical-reference/continuous-time-survival.md#the-full-refit-bootstrap)
gives both reasons and the study planned to measure it.

:::{dropdown} Details: which bootstrap fits a study covers
The registered study covers an in-sample, single-cause survival fit on the wide layout with one
treatment decision at each time point, static plans, and no weights, measured with correctly
specified cell-mean models on finite binary laws at $n = 1000$. A `TimeToEvent` fit, whose one
treatment is held over the time points, and a competing-risk fit print an experimental bootstrap. A fit with fewer clusters than its floor (groups such as care teams whose members' outcomes move
together; the floor is 20 for most longitudinal fits) prints its bootstrap standard deviation and percentile range as diagnostics.
[The full-refit bootstrap](../technical-reference/longitudinal-tmle.md#the-full-refit-bootstrap)
states what each resample redraws and refits, and
[Bootstrap for sparse regimens](#bootstrap-a-fit-over-time) covers the BCa
interval and the band for plans that few people follow.
:::
