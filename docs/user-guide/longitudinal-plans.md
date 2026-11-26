# Plans and policies over time

A plan over time says which treatment each person would get at each time point. The simplest
plans fix the treatment, such as always or never treat, and [describe data over
time](longitudinal.md) fits them. This page covers the others: a rule that reads the history so
far, a random draw from known chances, a shift in a continuous dose, a change in each person's
odds of treatment, and a trend line that summarizes several plans.

The first examples use this panel of two treatment times, with censoring:

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, LongitudinalTreatment, RegimeContrast
from cleverly.datasets import make_longitudinal

frame, truth = make_longitudinal(n=2_000, seed=11)
study = CausalStudy(
    frame,
    design=LongitudinalTreatment(
        outcome="Y",
        treatment=("A1", "A2"),
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    ),
)
```

## Declare a dynamic rule

Write each plan that holds a rule as a `DynamicRegimen`, and declare `rule_kind="known"`. The
declaration states that every callable node is a fixed rowwise function of the history.

```python
from cleverly.longitudinal import DynamicRegimen

adaptive = DynamicRegimen(
    "adaptive",
    (1, lambda history: (history["L2"] > 0).astype(int)),
    rule_kind="known",
)
rule_result = study.estimate(
    RegimeContrast({"never": 0, "adaptive": adaptive}, reference="never"),
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    n_folds=3,
    learner_folds=3,
    random_state=0,
)
```

This plan treats at the first node. At the second node it treats each unit whose `L2` is positive.
`DynamicRegimen` raises `CapabilityError` for `rule_kind=None` and for `"estimated"`. The fit
refuses a callable written inline in `regimens=`, such as `{"adaptive": (1, rule)}`. An inline
callable carries no declaration. A plan of labels alone, such as `{"never": 0}`, needs no
declaration. Code cannot inspect a closure, so the fit accepts a false `"known"` declaration.

To learn a two-node rule from the data instead, fit `LTMLE(regimens=LearnedRegimen())`. The fit
splits the rows into folds. Each fold learns a rule for each node from the other folds' rows, and
each row is scored under the rules its own fold learned. The result
`ey_regimen[learned regimen]` estimates the average outcome had every unit followed those learned
rules, which is not the value of the best possible rules.

```python
from cleverly.longitudinal import LTMLE, LearnedRegimen

learned = LTMLE(
    regimens=LearnedRegimen(),
    outcome_learner=LogisticRegression(max_iter=1000),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    censoring_learner=LogisticRegression(max_iter=1000),
    n_folds=10,
    random_state=0,
).fit(
    frame,
    outcome="Y",
    treatment=["A1", "A2"],
    baseline=["W1", "W2"],
    time_varying=[[], ["L2"]],
    censoring=["C1", "C2"],
)
learned_value = learned["ey_regimen[learned regimen]"]
print(learned_value.ci, learned_value.support.label)
print(learned.learned_regimen.describe())
```

Had everyone followed the learned rules, the average outcome would be between 0.735 and 0.811, and
the fit reads `supported`. The description says how the rules behave. By fold, they treat 97% to
100% of the people still following them at the first time point and 94% to 100% at the second.
Across folds, the rules disagree on 3% of rows at the first time point and 14% at the second. Rules
that treat almost everyone should be worth about as much as treating always, and `truth` gives 0.780
for that plan.

A [learned-regimen study](../technical-reference/method-evidence/learned-regimen-ltmle.md) measured
it with 2,000 rows, ten folds, dropout at each node and correct logistic models for the treatment
and the dropout. Its 95% interval covered the target 0.954 of the time over 1,200 samples, and
that coverage rests on those two models being right. A fit with two yes-or-no treatments, dropout
and one yes-or-no outcome at the end, every option at its default, is marked supported, and other
fits are marked experimental. Where the best rule is close to indifferent for many units, the
interval can cover less often, and no study measures that case.

The fit learns only two time points with a yes-or-no treatment at each. It refuses the default
library, or any learner that picks among models by an internal cross-validation, for the model of
the first time point (`pseudo_learner`, or `outcome_learner` when `pseudo_learner` is not passed),
because that cross-validation would score rules learned from its own rows. Pass a single learner
there, as here. It also refuses `n_folds=1`, `n_bootstrap=` and `repeats=` above 1, and a study
cannot declare a learned plan, so call `LTMLE` directly.
[Learned regimens](../technical-reference/longitudinal-tmle.md#learned-regimens) lists every
refusal, and [two decisions after discharge](../examples/two-decision-rules.ipynb) works through a
learned pair of rules from the data to the conclusion.

## Declare a stochastic policy

A node can draw its arm from a known distribution. Write the node as a `Stochastic` node with
`density_kind="known"`, inside a `DynamicRegimen`. The density function receives the history and
the earlier treatments, and it returns one column per treatment level, in sorted level order.

```python
import numpy as np
from cleverly.interventions import Stochastic


def half(history):
    return np.full((len(history), 2), 0.5)


def by_marker(history):
    treat = np.where(history["L2"] > 0, 0.75, 0.25)
    return np.column_stack([1.0 - treat, treat])


mostly = DynamicRegimen(
    "mostly",
    (
        Stochastic(half, "half", density_kind="known"),
        Stochastic(by_marker, "marker", density_kind="known"),
    ),
)
policy_result = study.estimate(
    RegimeContrast({"never": 0, "mostly": mostly}, reference="never"),
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    n_folds=3,
    learner_folds=3,
    random_state=0,
)
```

This plan treats half of the units at the first node. At the second node it treats three quarters
of the units whose `L2` is positive and a quarter of the others. The parameter is the mean outcome
if every unit drew its arms this way.

A rule node cannot read an earlier treatment, because the history it receives holds none. To
continue the arm that an earlier policy node drew, write a one-hot `Stochastic` node that reads
it. The fit resolves a one-hot density as the rule it equals.

```python
def continue_drawn(history):
    drawn = (history["A1"] == 1).to_numpy(dtype=float)
    return np.column_stack([1.0 - drawn, drawn])


continued = DynamicRegimen(
    "continued",
    (
        Stochastic(half, "half", density_kind="known"),
        Stochastic(continue_drawn, "continue", density_kind="known"),
    ),
)
```

Positivity is required wherever the policy puts mass. A policy that can draw a level that no unit
on the plan received at that node raises `LongitudinalError` before any learner.
[Known stochastic policies](../technical-reference/longitudinal-tmle.md#known-stochastic-policies)
gives the estimator, its conditions, and its evidence. Code cannot inspect a closure, so the fit
accepts a false `"known"` declaration.

## Shift a continuous dose over time

A plan can change each person's dose at each time point instead of fixing it. A dose is an amount
such as a price, and the plan might raise every month's price by 10%. Name each dose column in
`continuous_treatment=`, mapped to the band of values its doses can take. Write each time point's
change as a policy from `cleverly.interventions`, and put one policy per time point in a
`DynamicRegimen`, the plan object of [Declare a dynamic rule](#declare-a-dynamic-rule).
[Modified treatment policies](policies.md#modified-treatment-policies) describes each kind of
change, including a cap that holds a dose where it is.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from cleverly import CausalStudy, LongitudinalTreatment, RegimeContrast
from cleverly.datasets import make_sequential_pricing
from cleverly.interventions import Scale, Shift, Support
from cleverly.longitudinal import DynamicRegimen

prices, _ = make_sequential_pricing(n=2_000, seed=1)


def band(month):
    engaged = f"engaged_{month}"
    return Support(lambda h: 4 + 2 * h[engaged], lambda h: 8 + 2 * h[engaged])


price_design = LongitudinalTreatment(
    outcome="renewed",
    treatment=("price_1", "price_2", "price_3"),
    baseline=("usage",),
    time_varying=(("engaged_1",), ("engaged_2",), ("engaged_3",)),
    continuous_treatment={f"price_{month}": band(month) for month in (1, 2, 3)},
)
current_prices = DynamicRegimen("current", (Shift(0.0, cap=None),) * 3)
rise_plan = DynamicRegimen("rise 10%", (Scale(1.1, cap=8.0),) * 3)
price_fit = CausalStudy(prices, design=price_design).estimate(
    RegimeContrast({"current": current_prices, "rise 10%": rise_plan}, reference="current"),
    outcome_learner=LogisticRegression(max_iter=1000),
    pseudo_learner=LinearRegression(),
    treatment_learner=make_pipeline(
        PolynomialFeatures(2, include_bias=False),
        StandardScaler(),
        LogisticRegression(max_iter=5000),
    ),
    cross_fit=False,
    random_state=0,
)
rise_effect = price_fit["ate_regimen[rise 10% vs current]"]
low, high = rise_effect.ci
print(round(rise_effect.psi, 3), round(low, 3), round(high, 3), rise_effect.support.label)
```

The rise plan multiplies each month's price by 1.1, but a customer whose raised price would pass 8
keeps their own price that month. Against current prices it lowers renewal by about 4 percentage
points, with a 95% confidence interval from about 2 to 6 points. `price_fit[...]` picks that
comparison by the name the fit gives it, and `.psi` is the estimate and `.ci` the interval.
`Shift(0.0, cap=None)` adds nothing to any price, so the `"current"` plan is the prices customers
actually got.

The label `supported` means a simulation study checked fits like this one: three monthly prices, a
yes-or-no outcome, and the default way of learning the weights described below. The example fits
and checks its models on the same rows (`cross_fit=False`). The study covers that fit, and also
the fit with out-of-fold predictions, where models fitted on other rows predict each row. The
policy-weighted interval of [When few customers carry the interval](#when-few-customers-carry-the-interval)
needs `cross_fit=False`.

### Keep the new doses inside the band

The fit can only learn what customers would do at prices that customers like them actually paid,
so each new price must stay inside the band of prices they could get. Here the band depends on
that month's engagement, which is 0 or 1: prices run from 4 to 8 for a customer who is not engaged
and from 6 to 10 for one who is. The example declares the band with `Support(lower, upper)`. Each
bound is a function of `h`, the history the fit hands it, so the band can read that month's
`engaged_` column. The cap of 8 is the top of the lower band, so a raised price either stays at or below 8 or
goes back to the customer's own price, and the plan never leaves either band.

A plan that never moves a price outside the band needs no cap, as with `Shift(0.0, cap=None)`.
`Scale(1.1, cap=None)` would raise some prices past the top of their band, and the fit refuses it.
The message names the time point and how many customers the raise would send outside the band. Add
a cap that keeps every raised price inside, as the example does with 8.
[Modified treatment policies at a node](../technical-reference/longitudinal-tmle.md#modified-treatment-policies-at-a-node)
states the rule.

### How the fit weighs each customer

The fit corrects its outcome model's predictions with a second model, the treatment model. That
step needs, for each customer and time point, how much more or less common the plan's price is
than the price the customer got, given their history. This number is the density ratio, which the
summary calls the policy ratio. By default the fit learns it with a classifier: the treatment
learner learns to tell the plan's prices from the observed ones, and the summary's `Policy ratio`
line names this route.

The classifier must follow how the plan changes prices. A 10% rise held at 8 moves high and low
prices by different amounts, so the example gives the classifier squares and products of the
history and the price. With a plain logistic regression the same fit warns with
`SparseInfluenceWarning` that its interval rests on few customers. When your fit gives that
warning, try a richer classifier before you read the interval. If the fit warns with a
`DataWarning` about a few prices far from the rest, pass `ratio="density"`, which learns the ratio
from a binned density of the price instead
([density-ratio route](../technical-reference/scope-and-refusals.md#density-ratio-route)).

### Revenue and other continuous outcomes

For an outcome measured on a scale, such as revenue, pass the lowest and highest values it can
take as `q_bounds=`. Revenue here is renewal (0 or 1) times the last price, which is at most 10, so
the fit takes `q_bounds=(0.0, 10.0)`. Use the values the outcome can take, not the smallest and
largest in your sample. The plan moves customers toward prices where few outcomes were recorded,
and the bounds keep the fit's predictions there inside a possible range. Under a dose plan the fit
refuses a continuous outcome without `q_bounds`, and it stops with an error, rather than trimming
the value, when an outcome falls outside them
([limits of the continuous-policy evidence](../technical-reference/longitudinal-tmle.md#limits-of-the-continuous-policy-evidence)).

### Other dose plans

If each time point records both whether a customer got a discount and how large it was, declare
both columns as one time point
([vector nodes with a continuous component](../technical-reference/longitudinal-tmle.md#vector-nodes-with-a-continuous-component)).
To set the cap from the data, such as the 90th percentile of observed prices, write
`Shift(0.5, cap=DoseQuantile(0.9), target="population")`, and read
[modified treatment policies](policies.md#modified-treatment-policies) for what `target=` means.
To see which larger rises the data can support before you pick one, the
[sequential pricing tutorial](../examples/sequential-pricing.ipynb) compares a 12% rise and a 15%
cut under the same band. To check a fit against a known answer, `price_rise(factor)` from
`cleverly.datasets` builds a rise that stops at the top of each band, `pricing_truth(factors=...)`
gives its exact effect, and `PRICE_BANDS` holds the band.

### What the studies cover

The label (`supported` or `experimental`) checks the method, the number of time points, whether
the fit uses out-of-fold predictions, whether customers drop out, the outcome type, how the ratio
is learned and whether the cap comes from the data. It does not check your plan, your classifier
or your band. The price study learned the ratio with a logistic classifier on squares and
products, like the example's.

With the default way of learning the ratio, the
[price study](../technical-reference/method-evidence/mtp-pricing.md) covers two and three time
points, out-of-fold predictions at three, and revenue at three. At five time points, on a price
cut, its default interval covered the truth about 92% of the time instead of 95%. So a five-period
fit reads `experimental` unless it takes the policy-weighted interval below. Customers who drop
out between prices are covered only at three time points with out-of-fold predictions
([censored price study](../technical-reference/method-evidence/mtp-pricing-censored.md)).

When your plan differs from the price study's, the nearest evidence is in another study. The
[boundary study](../technical-reference/method-evidence/mtp-boundaries.md) covers plans that run
into a band that depends on the history, the
[tuned-cap study](../technical-reference/method-evidence/mtp-tuning.md) a cap set from the data at
two time points, and the
[vector study](../technical-reference/method-evidence/vector-modified-treatment-policies.md) a
dose recorded beside a yes-or-no treatment. The
[longitudinal policy study](../technical-reference/method-evidence/longitudinal-modified-treatment-policies.md)
checks that the fits match R's `lmtp` package.

### When few customers carry the interval

A dose plan can rest on a few customers with large density ratios, and `SparseInfluenceWarning`
then fires as it does for a sparse regimen. The refit bootstrap of
[Bootstrap for sparse regimens](inference.md#bootstrap-a-fit-over-time) does not help here. On the
five-period price cut of the pricing study, the bias-corrected bootstrap interval (BCa) covered
about 91% to 92% of the time, no better than the default interval
([the interval under heavy cumulative density ratios](../technical-reference/longitudinal-tmle.md#the-interval-under-heavy-cumulative-density-ratios)).

Pass `variance="policy_weighted"` instead. It needs `n_folds=1` (`cross_fit=False` on a study) and
`simultaneous=False`, which turns off the joint band across several estimates. The option also
works out the standard error from every row at the price the plan would set, and reports the
larger of that and the usual one. The interval is never narrower than the default, and it no
longer runs too narrow when a sample happens to miss the rare rows with extreme weights, but it
can be wider than it needs to be.

In large samples its coverage is close to 95% when either the density ratio is known exactly or
the outcome model is right. The option reads `supported` on the price plans at two and five time
points and `experimental` at three. At three time points, as in the example, the study covers the
default interval, so report that one. When the warning fires there, add the policy-weighted
interval beside it to show how much the rare heavy rows could widen it.
[The interval under heavy cumulative density ratios](../technical-reference/longitudinal-tmle.md#the-interval-under-heavy-cumulative-density-ratios)
states which fits it refuses and what it was validated against.

## Estimate an incremental intervention over time

Pass `IncrementalEffect` to `identify()` and `method="ipsi"` to `estimate()`. Each
`Incremental(delta)` multiplies every unit's odds of treatment by `delta` at every node.
`Incremental(1.0)` is the natural course, and the example uses it as the reference. A node with
three or more levels takes one factor for each level, such as
`Incremental({"high": 2.0, "low": 0.5})`. A level that the mapping does not name keeps the
factor 1.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, IncrementalEffect, LongitudinalTreatment
from cleverly.datasets import make_longitudinal
from cleverly.interventions import Incremental

frame, _ = make_longitudinal(n=300, seed=0)
study = CausalStudy(
    frame,
    design=LongitudinalTreatment(
        outcome="Y",
        treatment=("A1", "A2"),
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    ),
)
effect = study.identify(IncrementalEffect([Incremental(1.0), Incremental(2.0)]))
result = effect.estimate(
    method="ipsi",
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(),
    censoring_learner=LogisticRegression(),
    n_folds=2,
    random_state=0,
    simultaneous=False,
)
assert result.config.estimator == "ipsi"
assert list(result.estimates) == ["ate_regimen[odds x2 vs natural course]"]
print(result.summary())
```

The `ipsi` method fits the one-step estimator of Kennedy (2019), with the dropout weights of Kim,
Kennedy and Naimi (2021). It needs no treatment positivity at a tilt node. It is not doubly
robust, so give `treatment_learner` a learner that can fit the mechanism at every node.
`available_methods()` lists `tmle` and `sdr` as unavailable for an estimated tilt, with the
reason. The [weekly orders tutorial](../examples/weekly-outcomes.ipynb) fits the odds tilt,
doubling each customer's odds of a free-shipping code over three weeks.

| task | how |
| --- | --- |
| fit the engine directly | `IPSI([Incremental(1.0), Incremental(2.0)], ...)` from `cleverly.longitudinal` |
| use one tilt per node | `DynamicRegimen("ramp", (Incremental(1.5), Incremental(2.0)))` |
| tilt one node and fix another | `DynamicRegimen("tilt then stop", (Incremental(2.0), 0))` |
| tilt a node with three levels | `Incremental({"high": 2.0, "low": 0.5})`. A scalar there is refused before any learner, and so is a key that is not a level of the node |
| declare a known treatment mechanism | pass `treatment_probabilities=` to the design. The tilt is then a known policy. Beside weights, `LTMLE` reads it as one only with `mechanism_law="target"`; with `"sample"` it refuses and `IPSI` estimates the tilt |

A level can leave the outcome range, because the estimate is an average of influence-function
values. The fit then warns with `ParameterSpaceWarning`. A risk curve can also be non-monotone
across horizons. A mean of a non-absorbing outcome $Y_t$ is the end-of-study estimand on the panel
truncated after node $t$. Earlier $Y_s$ are covariates there (Kim et al. 2021, Theorem 3.1).

A fit with a node of three or more levels carries the label of the registered study
[categorical incremental policies](../technical-reference/method-evidence/categorical-incremental.md)
when it matches that study's rows. Any other such fit is labelled `experimental`.

`IPSI` fits Kennedy's odds tilt. The `ipsi` function of R's `lmtp` package fits a different plan,
`RiskRatioTilt(delta)` from `cleverly.interventions`. With `delta` below 1, each treated person
keeps their treatment with chance `delta` at each time point, and with `delta` above 1, each
untreated person stays untreated with chance `1 / delta`. Declare it as a node, as in
`DynamicRegimen("rr 0.5", (RiskRatioTilt(0.5),) * 2)`, and fit it with the default method.
[Incremental interventions over time](../technical-reference/longitudinal-tmle.md#incremental-interventions-over-time)
states the contract.

## Summarize several plans with a trend line

Instead of one mean per plan, you can fit a straight line through the plans' means, such as the
mean outcome against how many time points a plan treats. You choose the line's form, called a
working model (`MSM`), and ask for its coefficients with `MSMProjection`. The coefficients estimate the best line through the true plan
means, with each plan weighted equally, whether or not those means fall on a line. Only when they
do is the slope the effect of one more treated time point.

```python
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, LongitudinalTreatment, MSMProjection
from cleverly.datasets import make_longitudinal
from cleverly.msm import MSM

trend_frame, trend_truth = make_longitudinal(n=2_000, seed=11)
trend_study = CausalStudy(
    trend_frame,
    design=LongitudinalTreatment(
        outcome="Y",
        treatment=("A1", "A2"),
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    ),
)
treated_points = {"never": 0, "early": 1, "always": 2}


def duration(plan, horizon, covariates):
    n = len(covariates)
    return np.column_stack([np.ones(n), np.full(n, treated_points[plan])])


line = MSM(design=duration, terms=("intercept", "time points treated"), design_kind="known")
trend = trend_study.estimate(
    MSMProjection(line, regimens={"never": 0, "early": (1, 0), "always": 1}),
    outcome_learner=LogisticRegression(max_iter=1000),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    censoring_learner=LogisticRegression(max_iter=1000),
    random_state=0,
)
print(trend.coefficients()[["estimand", "psi", "ci_lower", "ci_upper"]])
```

The three plans treat at neither time point, at the first only, and at both. The slope row,
`msm_regimen[time points treated]`, reads 0.171 (0.139 to 0.203): on this line, each extra time
point treated raises the chance of the outcome by about 17
percentage points. Only simulated data come with the
truth. Here the true plan means in `trend_truth` are 0.419, 0.644 and 0.780, and the
least-squares line through them has intercept 0.434 and slope 0.181, inside both intervals.

The design function receives each plan's name, the time point the mean is taken at, and the
covariates, and returns one column per term. `design_kind="known"` declares that you wrote it
without looking at the data. The fit refuses a design function without that declaration, because
a design computed from the sample could leave the standard error wrong. `MSM.linear()` is refused
over plans, because a plan's name is not a dose, so code each plan as a number yourself, as
`treated_points` does. A design whose columns repeat each other across the plans, such as two
constant columns, is refused too.
[MSM projections](../technical-reference/msm-projections.md#variations) lists the other options.

The fit cross-fits by default (`n_folds=10`), so each row's predictions come from models trained
without that row.
[The longitudinal projection](../technical-reference/msm-projections.md#the-longitudinal-projection)
states the construction, and two simulation studies check it,
[in sample](../technical-reference/method-evidence/ordinary-longitudinal-msm-projection.md) and
[cross-fitted](../technical-reference/method-evidence/cross-fitted-longitudinal-msm-projection.md),
so both coefficients here carry the `supported` label.
[Marginal structural models](estimands.md#marginal-structural-models) shows the same summary for
one treatment, and [a trend across three contact schedules](../examples/msm-projections.ipynb)
works a full example.
[Summarize the risk curves with a trend line](survival.md#summarize-the-risk-curves-with-a-trend-line)
fits such a line over risk curves.
