# Estimators over time

Longitudinal TMLE (`LTMLE`) is the default estimator for a treatment decided at several times.
The other estimators on this page answer the same plan means and contrasts by leaning on
different models: the sequentially doubly robust estimator, sequential g-computation and inverse
probability weighting, structural nested mean models and the parametric g-formula. Fitting more
than one is a check, since estimators that rest on different models should agree when each is
close to right. The examples build their own panels with `make_longitudinal`; [describe data over
time](longitudinal.md) declares the roles.

## Longitudinal TMLE, the default

`study.estimate(...)` on a `LongitudinalTreatment` design fits `LTMLE`, which works backward from
the last time point, fitting an outcome regression at each one and correcting it with the treatment
and censoring models.

With `n_folds > 1`, each outer training fold runs an untargeted backward regression sequence. One
pooled fluctuation per node then targets the out-of-fold predictions over every follower. The fit
stores the realized assignment in `result.folds`.
[Cross-fitting the recursion](../technical-reference/longitudinal-tmle.md#cross-fitting-the-recursion)
gives the steps and the published result they follow.

Each node's fluctuation solves its score over every follower, as on a single-fold fit.
`res.diagnostics.score_equations()` reports one `solver` row per node.

The [cross-fitted end-of-study study](../technical-reference/method-evidence/cross-fitted-end-of-study-longitudinal-tmle.md)
measured standard-error ratios from 0.9797 to 1.0130 over 1,600 replications at `n=2000`. Those
results apply only to that study's law and estimator settings. They do not establish calibrated
variance for other laws, weights, clusters, survival outcomes, or sample sizes.

## Fit the sequentially doubly robust estimator

Pass `method="sdr"` to `estimate()` to fit the sequentially doubly robust estimator. It reports
the same parameters as the default method. It stays consistent when, at each node, either the
mechanism or the regression of the node's transform is consistent.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, LongitudinalTreatment, RegimeContrast
from cleverly.datasets import make_longitudinal

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
result = study.identify(RegimeContrast({"always": 1, "never": 0}, reference="never")).estimate(
    method="sdr",
    outcome_learner=LogisticRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(),
    censoring_learner=LogisticRegression(),
    n_folds=2,
    random_state=0,
    simultaneous=False,
)
print(result.summary())
```

Give `pseudo_learner` a learner of a real-valued target. A transform can leave $[0, 1]$, so a
level can leave the outcome range too. The fit then warns with `ParameterSpaceWarning`, and
`ratio()` refuses a scale that is undefined at that level.
[The sequentially doubly robust estimator](../technical-reference/longitudinal-tmle.md#the-sequentially-doubly-robust-estimator)
states the contract and what differs from the default method. The
[weekly orders tutorial](../examples/weekly-outcomes.ipynb) fits it beside `LTMLE` on a repeated
outcome.

## Report the sequential baselines

`SequentialGComputation` and `SequentialIPW` estimate the regimen levels and contrasts that
`LTMLE` estimates. Report them beside an `LTMLE` or `SDR` fit as reference estimators.

| estimator | the estimate reads | use it to |
| --- | --- | --- |
| `SequentialGComputation` | the outcome and pseudo-outcome regressions | check the targeted estimate against the plug-in |
| `SequentialIPW` | the treatment and censoring mechanisms | check the targeted estimate against the weighted mean |

```python
from cleverly.datasets import make_longitudinal
from cleverly.learners import GLM
from cleverly.longitudinal import SequentialGComputation, SequentialIPW

frame, _ = make_longitudinal(n=300, seed=0)
columns = dict(
    outcome="Y",
    treatment=["A1", "A2"],
    baseline=["W1", "W2"],
    time_varying=[[], ["L2"]],
    censoring=["C1", "C2"],
)
regimens = {"always": 1, "never": 0}
gcomp = SequentialGComputation(regimens, reference="never", outcome_learner=GLM()).fit(
    frame, **columns
)
ipw = SequentialIPW(regimens, reference="never", treatment_learner=GLM()).fit(frame, **columns)
print(gcomp.summary())
print(ipw.summary())
```

`gcomp()` and `ipw()` are the one-call forms. Both take the same columns and `regimens=`.

The interval of each baseline holds only when every `GLM` that the estimate reads is correctly
specified. Under misspecification the interval covers the limit of the misspecified fit, not the
target. A targeted estimator has no such condition on one of its two models.

When a regression that the estimate reads uses a data-adaptive learner, the result reports the
status `adaptive_baseline_plugin`. It gives the point estimate only. The standard error is a
diagnostic under `plugin_std_error`. Fit `LTMLE` or `SDR` when you need an interval with
data-adaptive learners.

The answers can differ from `LTMLE` for three reasons.

| reason | effect |
| --- | --- |
| g-computation applies no fluctuation | its estimate reads the outcome regressions alone |
| IPW uses no outcome regression | its estimate reads the mechanisms alone, and a large weight moves it |
| both baselines fit `GLM` models by default | `LTMLE` fits a Super Learner by default |

With the same outcome learner, the g-computation estimate equals the untargeted first step of
`LTMLE`. [Sequential g-computation and IPW](../technical-reference/sequential-baselines.md) gives
the estimators, the stacked influence curve, the settings, and the refusals. The
[sequential offers tutorial](../examples/sequential-offers.ipynb) fits g-computation and `LTMLE`
with the same misspecified outcome model on one campaign.

### Use GLM for the sequential baselines

`cleverly.learners.GLM` is an unpenalized maximum-likelihood fit with a canonical link. Its score
and information are public, so `SequentialGComputation` and `SequentialIPW` stack them into an
interval. See [Report the sequential baselines](#report-the-sequential-baselines).

| behavior | what `GLM` does |
| --- | --- |
| `family="binomial"` on a target with three or more labels | raises `ValueError`. Pass `family="multinomial"` or `"auto"` |
| a design column that is constant, or a combination of other columns, on the fit rows | aliases the column. It gets no coefficient |
| `alias_` | holds each aliased column as a combination of the active columns |
| `outside_span(x)` | returns `True` for each row where an aliased column departs from that combination |

A prediction through an aliased column assumes that the column has no effect. On a row that
departs from the fit rows, the data cannot support that assumption. Sequential g-computation
refuses such a fit with `DataError`, because no follower has that history.
[Sequential g-computation and IPW](../technical-reference/sequential-baselines.md#refusals) lists
this refusal for g-computation only.

```python
import numpy as np

from cleverly.learners import GLM

x = np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0]])
model = GLM().fit(x, np.array([0, 1, 1, 0]))
print(model.active_)
print(model.outside_span(np.array([[0.0, 1.0], [1.0, 1.0]])))
```

The first column is constant on the fit rows, so the fit aliases it with the intercept. The row
`[0.0, 1.0]` departs from that relation.

## Estimate structural blips

`StructuralNestedMean` fits a linear structural nested mean model. Use it to report the effect of
each treatment node on the end-of-study outcome, and to let that effect vary with declared
features. It is a separate method. `LTMLE` and `SDR` fit no model for the effect.

| you want | fit |
| --- | --- |
| the effect of node $t$ with later treatment at the reference, as a coefficient matrix | `StructuralNestedMean` |
| the mean outcome under a static regimen, with no model for the effect | `LTMLE` or `SDR` |
| the value of a rule, a policy or an incremental intervention | `LTMLE`, `SDR` or `IPSI` |

Declare the features of each node in `blip_features`. A key is a node counted from one, or its
treatment column. A value lists columns. A tuple of columns declares a product. The intercept is
always a feature. A feature may read the baseline columns, the covariates through its node, and
the earlier treatments.

```python
from cleverly.datasets import make_longitudinal
from cleverly.learners import GLM
from cleverly.longitudinal import StructuralNestedMean

frame, _ = make_longitudinal(n=300, seed=0, censoring=False)
result = StructuralNestedMean(
    {"always": 1, "never": 0},
    reference="never",
    blip_features={1: ["W1"], 2: ["W1", "A1"]},
    outcome_learner=GLM(family="binomial"),
    treatment_learner=GLM(family="binomial"),
    n_folds=2,
    random_state=0,
).fit(
    frame,
    outcome="Y",
    treatment=["A1", "A2"],
    baseline=["W1", "W2"],
    time_varying=[[], ["L2"]],
)
print(result.summary())
print(result.blip(2))
```

`snmm()` is the one-call form. `result.blip(node)` returns the coefficients of a node as a matrix
with one row per feature and one column per coding component. It is the transpose of the matrix
$\Psi_t$ of the
[technical page](../technical-reference/structural-dynamic-effects.md#the-model). A categorical node has one column for each level
after the first sorted level. A continuous node has one column, the dose. The regimen levels and
contrasts follow from the blips and the mean outcome at the reference level `ey_reference`.

Pass `regression_learner=` when the outcome is binary and a node holds a dose or a feature is
continuous. A classifier takes no target outside $\{0, 1\}$, and `regression_learner` fits those
targets. It defaults to `outcome_learner`.

Read the result with this condition. The intervals are valid under the linear structural nested
mean model with the declared blip features. Every row of the summary and of `to_frame()` carries
that label. A blip that depends on an undeclared earlier treatment or covariate makes the
coefficients answer a different question.

The evidence label is "comparator-backed", against EconML 0.17.0 `DynamicDML`. The
[structural nested mean study](../technical-reference/method-evidence/structural-nested-mean.md)
covers cross-fitted fits with `GLM` nuisances on binary, three-level and dose nodes. It also
covers known weights and 40 or more clusters. A fit inside those rows reads `supported`. Any other fit, such as an in-sample fit, reads `experimental`. The summary
prints a `support` column and the reason.

Check the model against `LTMLE`.

```python
check = result.check_model()
print(check.summary())
```

`check_model()` fits `LTMLE` on the same rows and folds. A cross-fitted check refits both
estimators on `n_splits` seeded splits (default 5) and averages the difference. It reports a joint
Wald statistic on the difference of the contrasts. A rejection says that the model fails for these contrasts. A
non-rejection does not confirm the model.

| the fit | what it reports |
| --- | --- |
| `n_folds > 1` | the influence-curve intervals |
| `n_folds=1` and every learner a `GLM` | the influence-curve intervals, valid when every `GLM` is correct |
| `n_folds=1` and another learner | the point estimate, with a diagnostic standard error |

The estimator refuses a rule, a policy, censoring, survival and repeated outcomes. It refuses a
static level, a contrast and `check_model` when a feature reads a time-varying covariate.
[Structural dynamic effects](../technical-reference/structural-dynamic-effects.md#refusals) lists
each refusal and its alternative. The
[three offers after discharge tutorial](../examples/model-based-over-time.ipynb) fits blip models
to ask how much each single offer adds.

## Simulate the parametric g-formula

`ParametricGFormula` estimates the same regimen levels and contrasts as `LTMLE`. It fits a model for
each time-varying covariate and for the outcome. It then simulates each unit forward under each
regimen and averages the simulated outcomes. `SequentialGComputation` reaches the same target by
regression backward through the nodes.

| reason | choose |
| --- | --- |
| you need the forward-simulation form that `gfoRmula` and `pygformula` report, with an analytic interval | `ParametricGFormula` |
| a plan fixes a continuous dose at a node, or sets it by a known rule | `ParametricGFormula`. `LTMLE` and `SequentialGComputation` refuse both |
| you doubt the law of a time-varying covariate | `SequentialGComputation`, `LTMLE` or `SDR`. They model no covariate law |
| you want a fit that a misspecified model of one kind cannot bias | `LTMLE` or `SDR` |

The parametric g-formula needs a correct model for every covariate and for the outcome. It reads
more models than the backward form, and each one must be right.

```python
from cleverly.datasets import make_longitudinal
from cleverly.longitudinal import ParametricGFormula

frame, _ = make_longitudinal(n=300, seed=0)
result = ParametricGFormula(
    {"always": 1, "never": 0},
    reference="never",
    covariate_types={"L2": "normal"},
    n_draws=20,
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
print(result.simulation.summary())
```

`gformula()` is the one-call form. The fit refuses an undeclared time-varying column that is not
coded 0/1, before any learner. Declare it in `covariate_types=` as `"binary"`, `"categorical"` or
`"normal"`.

The interval includes the simulation error. `result.simulation.summary()` lists the models the
simulation draws from and the share of each variance that the simulation contributes, and the fit
warns above 10%. Raise `n_draws=` to lower it. `result.simulation.monte_carlo_share` holds the same
shares as a dictionary. A fit with the same `random_state=` repeats exactly.

A data-adaptive learner for any model that the estimate reads gives the status
`adaptive_baseline_plugin`: the point estimate only. A fixed number at a continuous treatment node
is a fixed dose, and a `DynamicRegimen` rule with `rule_kind="known"` there sets the dose from the
simulated history. The result then states its model reliance and reports a support diagnostic. No
published analytic variance exists for the simulation form, so the interval is a natural
extension. [Parametric g-formula](../technical-reference/parametric-gformula.md) gives the
estimator, the argument, the settings, the refusals and the validation. The
[three offers after discharge tutorial](../examples/model-based-over-time.ipynb) uses it for its
main answer and compares it with structural blips and longitudinal TMLE.

### Use GLM for the parametric g-formula

`ParametricGFormula` fits a `GLM` for every covariate, outcome and hazard model by default. The
fit chooses the family of each model from its response: binomial for a binary column, multinomial
for a categorical one and Gaussian for a normal one. A `GLM` with another family is refused.
The forward draw of a Gaussian column needs the residual variance of the fit, so a Gaussian
covariate accepts a `GLM` only. A data-adaptive learner for any other model gives the status
`adaptive_baseline_plugin`. See
[Simulate the parametric g-formula](#simulate-the-parametric-g-formula).

Choose `ParametricGFormula` over `SequentialGComputation` when the plan fixes a continuous dose,
or when you need the forward-simulation form. Choose `SequentialGComputation` when you doubt a
covariate law, because it models no covariate.

## Weight trimming, shorter histories, and node bounds

Four settings change how `LTMLE` and `SDR` fit their models. None of them changes the quantity
being estimated.

`ratio_trim=q` caps each node's weight at the `q` quantile of all the node weights, the rule of
R `lmtp`. Through a study, pass `Targeting(ratio_trim=q)`. The cap tames extreme weights, but no
published result gives an interval for the capped fit. A fit whose cap moves any weight therefore
reports its point estimate with a diagnostic standard error, under the status
`trimmed_ratio_plugin`. `RegimenFit.trimmed_share` gives the share of weights the cap moved.
`IPSI` and a fit with `survival_model="hazard"` refuse the option.

`LongitudinalTreatment(markov_order=k)` lets each model at node `t` read the baseline covariates
and only the last `k + 1` nodes of history. It assumes that the earlier history adds nothing to those
models. The causal conditions still need the full history: every confounder of each decision must
be recorded somewhere in it. The estimate stays on target in large samples when one family of
models, the outcome models or the treatment and censoring models, is right on the window. The
interval needs both. `ParametricGFormula` and `StructuralNestedMean` refuse the declaration.

A treatment given once and held stays in every model, since it is the treatment at every
node. A covariate measured before that treatment leaves the models once it is out of the window,
so declare it in `baseline=` to keep it.

`LTMLE(hazard_pooling="pooled")` fits one censoring model over every node, with the time as a column,
in place of one model per node. On a hazard survival fit it also fits each cause's hazard once.
Through a study, pass `ModelSpec(hazard_pooling="pooled")`. The option needs one baseline
treatment and no covariate measured after the first time point.

`LTMLE(node_bounds={t: (lower, upper)})` declares the known range of the outcome regression at chosen
nodes, keyed by grid time. Through a study, pass `Targeting(node_bounds=...)`. The node's
regression and its correction step then stay inside the range. When the weighted mean outcome of
the units still following the plan at a node lies outside the declared range, the node runs no correction, and the fit
warns with `ConvergenceWarning` and names the node. `SDR` refuses the option, because it runs no
correction step.

The [survival node-bounds study](../technical-reference/method-evidence/survival-node-bounds.md)
fits node bounds and pooled censoring in repeated samples. The
[ratio-trim study](../technical-reference/method-evidence/option-ratio-trim.md) checks the capped
point estimate, but only when every outcome model is a `LinearRegression`, so a capped fit with
another outcome learner prints `experimental`. The
[Markov-order study](../technical-reference/method-evidence/option-markov-order.md) checks the
estimate and interval of a windowed `LTMLE` fit on four treatment nodes. Exact-law tests check
both settings, and [Ratio trimming](../technical-reference/longitudinal-tmle.md#ratio-trimming),
[Markov order](../technical-reference/longitudinal-tmle.md#markov-order),
[Pooled censoring](../technical-reference/longitudinal-tmle.md#pooled-censoring) and
[Node bounds](../technical-reference/longitudinal-tmle.md#node-bounds) state each rule and its
refusals.

## Repeat the fold draws

Pass `repeats=R` to `estimate()` to fit the whole estimator at `R` independent fold draws. The
result reports the median of the draws, and its variance adds the spread of the draws. Use it when
one split moves the estimate by more than you accept.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, LongitudinalTreatment, RegimeContrast
from cleverly.datasets import make_longitudinal

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
repeated = study.identify(RegimeContrast({"always": 1, "never": 0}, reference="never")).estimate(
    outcome_learner=LogisticRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(),
    censoring_learner=LogisticRegression(),
    n_folds=2,
    repeats=3,
    random_state=0,
)
assert repeated.n_repeats == 3
assert len({draw.seed for draw in repeated.repeats}) == 3
print(repeated.repeat_spread())
```

To see how much the split moves the answer, plot one point for each draw against the reported
estimate and its interval:

```python
import cleverly.plotting

fig = cleverly.plotting.repeats(repeated)
```

| task | how |
| --- | --- |
| read one draw | `repeated.repeats[r]` holds its folds, seed, fits and estimates |
| see how much the split moves each estimate | `repeated.repeat_spread()`, the standard deviation of the draws |
| contrast or ratio the levels | `contrast()` and `ratio()` run in every draw and report the median |
| set it on a method | `CrossFitting(repeats=3)` |

The fit refuses `repeats` above 1 with `n_folds=1`, because one fold has no split to draw again. It
also refuses `repeats` above 1 with `n_bootstrap` above 0. `SequentialGComputation` and
`SequentialIPW` refuse `repeats` above 1, because they fit in sample. A fit with `repeats` above 1
carries the `experimental` label until a registered study covers its configuration. With
`isotonic=True` each draw projects its levels and the fit builds no simultaneous band.

The rows of the support and nuisance reports describe draw 1. The support warning of `validate()` grades every draw and names the draw.
The score report has one row per node and draw. A repeated outcome, `Repeated([...])`, takes
`repeats` too.
[Repeated fold draws](../technical-reference/longitudinal-tmle.md#repeated-fold-draws) states the
rule, the seeds and the refusals.
