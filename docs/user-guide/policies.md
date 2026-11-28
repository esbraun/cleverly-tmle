# Dose shifts and odds changes

These estimands change the treatment each person would get instead of fixing it at one level. A
modified treatment policy shifts or scales each person's dose, an incremental intervention
multiplies each person's odds of treatment, and a dose-response curve gives the mean outcome at
every dose. An incremental intervention does not need every person to have a real chance of each
option, which the fixed contrasts of [average effects and fixed plans](estimands.md) need.

The examples use a `dose_study` with a continuous treatment `dose` and a `study` with a binary
treatment `A`, both on a binary outcome, and share one configuration, `quick`:

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CrossFitting, ModelSpec, TMLEMethod

quick = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LinearRegression(), treatment_learner=LogisticRegression(max_iter=1000)
    ),
    cross_fitting=CrossFitting(n_folds=5, learner_folds=3),
)
```

## Modified treatment policies

For a continuous dose, `Shift` maps each observed dose to a policy dose. `cap=` is part of the
policy and its support argument; it is not estimated from the sample maximum. `q_bounds` follows
the same rule for the outcome. Declare it from known support, and never from the observed range.

```python
from cleverly import ModifiedTreatmentPolicyEffect
from cleverly.interventions import Shift

policies = (Shift(0.0, cap=None), Shift(0.25, cap=4.0), Shift(0.5, cap=4.0))
result = dose_study.estimate(
    ModifiedTreatmentPolicyEffect(policies),
    method=quick,
    random_state=5,
)
```

A shifted dose above the cap holds the row at its own dose. The example cap of 4.0 lies inside
the observed doses. The fit issues a `PositivityWarning` when a declared cap lies above the largest
observed dose and the policy assigns a dose above that maximum.

The correction step needs, for each row, how much more or less common the policy dose is than
the dose the row received: a ratio of two densities of the dose given the covariates. By default
the fit learns that ratio directly. It stacks each observed row with its policy copy, the
treatment learner learns to tell the two apart, and the ratio comes from its predicted odds.

The learner also sees, for each row, whether the policy moved its dose, so a cap or a piecewise
policy is learned where it holds the dose. When you declare the dose's range, the learner also
sees whether the dose the policy moves from lies inside it. Without a declared range the fit
treats the smallest or largest dose it trains on as an edge, but only on the side where some moved
dose would have come from beyond it, as with a shift up and the smallest dose. That edge is a guess
of where the doses stop. Declare the range with `treatment_support=` (below) when you know it, such
as a dose that cannot fall below zero. A declaration always wins, so a dose with no limit can pass
`Support(-np.inf, np.inf)` to turn the edge off.

This route fits no density, and it accepts doses that `PointTreatment(dose_atoms=...)` declares
as recorded on many rows, such as zero. The result's summary prints the route it used and why
(`Policy ratio`). A learner that reads its input columns by position can set the class attribute
`policy_steps = "omit"` to receive the plain design.

The fit warns, with a `DataWarning` that names the range, when a few doses sit far from the rest
beyond an empty stretch of the dose. The learned odds can smooth over such a cluster, so pass
`ratio="density"` there. In the design study the warning fired on every sample of 20,000 rows of
a dose with such a cluster, and on 12 of 20 samples of 2,000. It never fired on two smooth doses in
80 samples, and once in 40 samples of a heavy-tailed dose. A cluster of one or two doses is not
reported. `cleverly.learners.isolated_dose_clusters` runs the same check on any column of doses.

With `ratio="density"` the fit estimates the density on a grid of dose intervals instead.
`density_bins` sets their number, and setting it also selects this route. The fit adds narrower
intervals where the doses thin out, then draws a smooth curve through the interval probabilities,
so even a small shift changes the ratio. The estimate is only as good as the treatment learner
behind it, so pass a flexible one, such as gradient boosting. A simple parametric learner can fit
the narrow outer intervals poorly. Both `ratio=` and `density_bins=` are settings of `ModelSpec`,
which `estimate` also takes as keywords ([settings reference](settings.md)).

Gradient boosting can also misfit a few rows in a thin tail. In
the [design study](../technical-reference/method-evidence/mtp-density-tails.md#reported-readings),
at 2,000 rows the default booster's ratio was far off on up to 3 of 20 samples for four of six
dose shapes, most often for a dose capped at the top of its range. In each such sample a handful
of rows carried most of the error. When the tail matters to the policy, compare the estimate with
a second treatment learner.

```python
density_fit = dose_study.estimate(
    ModifiedTreatmentPolicyEffect(policies),
    method=quick,
    ratio="density",
    density_bins=40,
    random_state=5,
)
```

The fit also uses the density route when it reads no policy ratio, as a dose-response curve does,
beside a declared treatment mechanism, and with an estimator that only reads a density. The
[resolution table](../technical-reference/scope-and-refusals.md#density-ratio-route) lists
which route each fit takes.

When the possible doses depend on the covariates, declare that range with
`Support(lower, upper)` from `cleverly.interventions`, where each bound is a number or a function
of the covariates. Pass it as `treatment_support=` to `PointTreatment` or to `TMLE.fit`.

On the density route it lets the fit tell a hard limit from doses that are merely rare. Without it the fit
spreads density past the limit, which distorts the ratio near it. On the default route the declared
range tells the learner where the dose a policy moves from leaves the possible doses. Without it
the fit uses the smallest or largest dose it trains on, only on the side the policy reaches past.
[Modified treatment policies](../technical-reference/point-treatment-tmle.md#modified-treatment-policies)
states both routes.

A cap can also come from the data. `Shift(1.0, cap=DoseQuantile(0.9), target="population")` holds
every dose whose shift would pass the 90th percentile of the observed doses.
`SupportBoundary(LinearQuantileRegression(features=...), q=0.9)` sets the cap from a quantile
regression of the dose on the named covariates, so the cap varies by row. `DoseQuantile`,
`SupportBoundary` and `LinearQuantileRegression` come from `cleverly.interventions`.

A tuned cap needs `target=`, and the fit refuses one without it. `target="population"` estimates
the mean under the cap that the whole population would give, and the interval includes the error
of estimating that cap. It needs an outcome model that is accurate near the cap.
`target="fold_average"` averages the means under each fold's own cap. The [tuned-cap study](../technical-reference/method-evidence/mtp-tuning.md) checks the
population target over two decisions. A tuned cap at one decision rests on exact-law tests only
(`tests/unit/test_point_tuned_caps.py`).

Other maps of the dose use the same `policies=` axis.

| class | use it for |
| --- | --- |
| `Scale(factor, cap=...)` | a dose multiplied by a factor, held at its own value above the cap |
| `Piecewise(pieces)` | a `Shift` or a `Scale` on each interval of the dose |
| `ModifiedPolicy(name, pieces=..., policy_kind="known")` | your own map, with its inverse and its derivative on each piece |
| `ModifiedPolicy(name, apply=..., policy_kind="known")` | a map of the levels of a categorical treatment |
| `ModifiedPolicy(name, pieces={value: pieces}, randomizer=Randomizer(values, probabilities), policy_kind="known")` | a policy that picks one of several maps at random with known chances, such as raising the dose for a random quarter of customers |
| `RiskRatioTilt(delta)` | the `ipsi` tilt of `lmtp` on a 0/1 treatment, reported as `ey_rr_tilt`. Fit it with `TMLE(policies=...)`. The typed study requests refuse it |
| `JointPolicy(name, components={column: policy}, policy_kind="known")` | a vector treatment of several columns. It sets one policy for each component, as the [vector treatments](../technical-reference/point-treatment-tmle.md#vector-treatments) section states |

```python
from cleverly.interventions import ModifiedPolicy, Piece, Scale

halve_below_3 = ModifiedPolicy(
    "halve below 3",
    pieces=(
        Piece(
            -float("inf"),
            3.0,
            lambda a, h: (a + 3.0) / 2.0,
            lambda b, h: 2.0 * b - 3.0,
            lambda b, h: 2.0,
        ),
        Piece(3.0, float("inf")),
    ),
    policy_kind="known",
)
policies = (Shift(0.0, cap=None), Scale(1.25, cap=5.5, name="x1.25"), halve_below_3)
```

A `Piece` without a map leaves its doses unchanged. The fit checks that every observed dose lies in
exactly one piece. It also checks that each declared inverse inverts its map, and that each declared
derivative matches the map. In a longitudinal fit, a policy is a node of
a `DynamicRegimen` plan, and
[Shift a continuous dose over time](longitudinal-plans.md#shift-a-continuous-dose-over-time) works an
example. The
[longitudinal reference](../technical-reference/longitudinal-tmle.md#modified-treatment-policies-at-a-node)
gives the details, and
[limits of the continuous-policy evidence](../technical-reference/longitudinal-tmle.md#limits-of-the-continuous-policy-evidence)
lists what the studies do not cover. The [monthly prices](../examples/sequential-pricing.ipynb)
example raises a price at each of three months.

A randomized policy draws one of its maps for each row, with the chances you declare in
`Randomizer`, and the fit averages over the draw. A `randomizer=` that is not a `Randomizer`,
such as a continuous draw, is refused
([modified treatment policies](../technical-reference/point-treatment-tmle.md#modified-treatment-policies)). In the example, a random quarter of rows get 0.5
more, held at their own dose where the raise would pass 4.0, as with the capped shifts above.
`ModifiedTreatmentPolicy` reports the mean outcome under each policy, where
`ModifiedTreatmentPolicyEffect` reports each policy's difference from the first one by default.
`Shift(0.0, cap=None)` changes no dose, so its row is the mean under current practice, named
`natural course`.

```python
from cleverly import ModifiedTreatmentPolicy
from cleverly.interventions import ModifiedPolicy, Piece, Randomizer

raise_some = ModifiedPolicy(
    "raise a random quarter by 0.5",
    pieces={
        "raise": (
            Piece(-float("inf"), 3.5, lambda a, h: a + 0.5, lambda b, h: b - 0.5, lambda b, h: 1.0),
            Piece(3.5, float("inf")),
        ),
        "keep": (Piece(-float("inf"), float("inf")),),
    },
    randomizer=Randomizer(("raise", "keep"), (0.25, 0.75)),
    policy_kind="known",
)
policy_means = dose_study.estimate(
    ModifiedTreatmentPolicy((Shift(0.0, cap=None), raise_some)), method=quick, random_state=5
)
```

[Three offer policies](../examples/interventions.ipynb) shifts a continuous intensity of navigation, and
[More contacts and more service time](../examples/schedules-and-intensity.ipynb) shifts two doses
at once.

## Incremental propensity-score interventions

`Incremental(delta)` multiplies the observed treatment odds by `delta`. It avoids a conventional
positivity requirement, but the target depends on the true treatment mechanism and therefore does
not have ordinary two-sided double robustness.

A treatment with three or more levels takes one factor for each level. Declare the factors as a
mapping from the level label to the factor. A level that the mapping does not name keeps the
factor 1. The odds of level $a$ against level $b$ are multiplied by the ratio of their factors.

| declaration | meaning |
| --- | --- |
| `Incremental(2.0)` | two levels only: the odds of the second sorted level times 2 |
| `Incremental({"high": 2.0})` | the odds of `"high"` against every other level times 2 |
| `Incremental({"high": 2.0, "low": 0.5})` | the factors of `"high"` and `"low"`, and 1 for the other levels |

A scalar on a treatment with other than two levels is refused before any learner, and so is a key
that is not a level. Keys are the labels that the data reports, and a position is not a label.
A vector treatment whose columns are all categorical is one treatment over its observed tuples.
Its keys are the tuples. In a longitudinal fit, one `Incremental` applies to every node, and its
keys are read against the levels of each node in turn.

A fit with three or more levels carries the label of the registered study
[categorical incremental policies](../technical-reference/method-evidence/categorical-incremental.md)
when it matches that study's rows. Any other such fit is labelled `experimental`.

```python
from cleverly import IncrementalEffect
from cleverly.interventions import Incremental

tilts = (Incremental(1.0), Incremental(2.0), Incremental(0.5))
result = study.estimate(IncrementalEffect(tilts), method=quick, random_state=3)
```

`IncrementalMean(tilts)` reports the mean outcome under each tilt rather than the differences.

```python
from cleverly import IncrementalMean

odds_means = study.estimate(IncrementalMean(tilts), method=quick, random_state=3)
```

[Three offer policies](../examples/interventions.ipynb) and
[More contacts and more service time](../examples/schedules-and-intensity.ipynb) each multiply the
odds of treatment, the second on a schedule with three levels.

## A curve over a continuous dose

`DoseResponseCurve` asks how the mean outcome changes with a continuous dose, such as a price or
a discount. A curve at one exact dose has no interval that narrows at the usual rate as the data
grow, so the fit reports a smoothed curve: at each declared dose it averages the curve over
nearby doses, within a declared bandwidth. You choose the doses, the bandwidths and the dose range
before looking at the results. The fit gives an interval at each dose and bandwidth, and one band
that covers all of them at once.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from cleverly import CausalStudy, DoseResponseCurve, PointTreatment
from cleverly.dose_response import DoseResponse

rng = np.random.default_rng(1)
n = 800
income = rng.normal(size=n)
discount = np.clip(2.5 + 0.8 * income + rng.normal(size=n), -1.0, 6.0)
bought = rng.binomial(1, 1 / (1 + np.exp(-(0.6 * np.sin(discount) + 0.5 * income))))
orders = pd.DataFrame({"income": income, "discount": discount, "bought": bought})

study = CausalStudy(
    orders,
    design=PointTreatment(
        outcome="bought",
        treatment="discount",
        adjustment=("income",),
        treatment_kind="continuous",
    ),
)
curve = DoseResponse(doses=[1.0, 2.5, 4.0], bandwidths=[0.8, 1.2], support=(0.5, 4.5))
dose_fit = study.identify(DoseResponseCurve(curve)).estimate(
    outcome_learner=HistGradientBoostingClassifier(max_depth=2, max_iter=50),
    treatment_learner=LogisticRegression(),
    density_bins=10,
    n_folds=5,
    random_state=0,
)
table = dose_fit.to_frame()
```

Each row of `table` is one dose and bandwidth. `lower` and `upper` are the interval for that row
alone. `band_lower` and `band_upper` hold for every row together, so read the band when you compare
bandwidths or pick one after seeing the results. A wider bandwidth usually gives a narrower interval
and more smoothing, unless it reaches doses with few observations. The smoothing bias grows with
the square of the bandwidth where the curve bends, so prefer the narrower bandwidth when its
interval is usable. `dose_fit.summary()` prints the same rows, and its full form adds the
conditions the answer rests on.

The intervals need both models to be good: the outcome model, and the model of which dose each
customer received given their characteristics. The dose range must lie inside the doses you
observed, with a margin: the fit refuses a range that reaches past them. A row whose dose has few
nearby observations is flagged in the `low_support` column, and its estimate's `inference` reads
`low_dose_support_plugin`: its interval rests on few effective rows. Each estimate's
`support.label` says whether a registered simulation study covers the fit's setup;
[the support-label table](../technical-reference/dose-response.md#support-label) lists the setups
that read `supported`. A continuous outcome also needs `q_bounds=`, its known range.
`dose_fit.assess()` runs the checks that apply, `dose_fit.diagnostics.influence()` shows each
row's influence curve, and `cleverly.plotting.curve(dose_fit)` draws one line per bandwidth with
the band. `DoseResponseTMLE(curve, ...).fit(orders, outcome=..., treatment=..., covariates=...)`
is the same fit without a study.
[Smoothed dose-response curve](../technical-reference/dose-response.md) states the estimand, the
conditions and the refusals.

[Renewal discounts](../examples/dose-response.ipynb) fits this curve on a discount and reads its
band.
