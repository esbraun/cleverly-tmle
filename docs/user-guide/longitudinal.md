# Longitudinal treatment

## Declare temporal roles

`LongitudinalTreatment` aligns treatment nodes, time-varying histories, censoring indicators, and
outcome processes. At treatment node `t`, a dynamic rule sees only history available by `t`.
For the reported influence-curve inference, the rule must be fixed before the fit and rowwise. A
callable may vectorize over the history frame, but it must not estimate a threshold from the
sample or otherwise make one row's assignment depend on other rows. The rule needs a declaration,
which [Declare a dynamic rule](#declare-a-dynamic-rule) shows.

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
result = study.estimate(
    RegimeContrast({"always": 1, "never": 0}, reference="always"),
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    n_folds=3,
    learner_folds=3,
    random_state=0,
)
```

The resolved regimen matrix is shared by nuisance fitting, follower masks, targeting, and report
keys. A regimen cannot look ahead or change interpretation between stages. An empirically learned
policy needs inference for the policy-learning step and is outside this estimator's contract.

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

## Survival outcomes

An outcome sequence declares one absorbing event process and makes horizon part of the estimand.

```python
from cleverly import RegimeMean
from cleverly.datasets import make_longitudinal_survival

frame, truth = make_longitudinal_survival(n=2_000, seed=12)
study = CausalStudy(
    frame,
    design=LongitudinalTreatment(
        outcome=("Y1", "Y2"),
        treatment=("A1", "A2"),
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    ),
)
risks = study.estimate(
    RegimeMean({"always": 1, "never": 0}, horizons=(1, 2)),
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
)
```

Each node fit uses the relevant uncensored, event-free risk set. The result key preserves regimen
and horizon.

## Competing risks

A mapping from cause labels to absorbing outcome sequences declares competing risks. Cause,
horizon, and regimen remain separate key fields. The engine uses a cause-specific recursion while
keeping all prior competing events out of later risk sets.

## Longitudinal MSM projections

`MSMProjection` can project regimen-specific longitudinal means onto a declared working model. The
regimen grid must identify the coefficient vector; a rank-deficient working design is refused.

Use `n_folds=1` for a longitudinal MSM projection. Cross-fitted coefficient inference is refused
until an unsaturated projection property and a repeated-sampling study validate that construction.
A saturated reduction alone does not validate an unsaturated projection or its coefficient
influence curve.

## Diagnostics

Longitudinal results provide cumulative support, targeting-score, and nuisance-model reports by
node. Call `result.diagnostics.support()` for leverage and truncation by stage.

The nuisance report separates four roles. Each fitted treatment and censoring model appears once
per node. Outcome and pseudo-outcome models appear per fitted regimen, cause, horizon, and node.
Each row labels its loss as `out_of_fold` or `in_sample`.

The report uses weighted negative log likelihood for treatment and censoring. It uses weighted
Brier loss for a binary final target and weighted mean squared error otherwise. Earlier
pseudo-outcomes use weighted mean squared error. A complete-data fit records why no censoring
learner appears in `nuisance.omissions`.

Use an explicit grid to describe point-estimate movement across cumulative mechanism bounds:

```python
curve = result.diagnostics.truncation_curve(bounds=[0.01, (0.05, 0.95)])
```

A scalar `b` means `(b, 1)`, not the point-treatment shorthand `(b, 1 - b)`. The pair bounds each
cumulative treatment and censoring product after multiplication. The diagnostic keeps raw
mechanism predictions fixed and reruns the complete backward recursion at every pair. It refits
every bound-dependent outcome or pseudo-outcome regression and targeting update.

The curve is descriptive. It reports no standard error, confidence interval, preferred bound, or
pass threshold. It refuses `mechanism=True`, which is a point-treatment option. It also refuses a
learner it cannot replay.
[Replay-only unavailability](../technical-reference/scope-and-refusals.md#replay-only-unavailability)
states the `random_state` rule each outcome and pseudo-outcome learner must satisfy.

[Truncation stability](../technical-reference/validation-methods.md#truncation-stability) holds the
returned frame's columns and its two score-cell counting rules.

A combined report requires both the grid and permission for expensive refits:

```python
report = result.diagnostics.run_all(
    include_refits=True,
    arguments={"truncation_curve": {"bounds": [0.01, 0.05]}},
)
```

Point-only sensitivity formulas remain unavailable unless a longitudinal derivation exists.

Tan (2025) derives population sensitivity bounds for binary, static longitudinal strategies, but
no sample estimator for them. The
[roadmap](../roadmap.md#f16-longitudinal-sensitivity-bound-estimation) records that boundary.

## Persistence

`result.save()` and `cleverly.load()` carry a longitudinal result through a round trip. The
artifact keeps the folds, the fitted mechanisms, the sequential steps, the targeting state, and
the causal metadata, and the truncation replay recipe. [Persistence and
replayability](results-assessment.md#persistence-and-replayability) states the shared contract for
every result, including the assessment cache and the capability rows a restored artifact refuses.
`tests/unit/test_serialization.py`'s
`test_longitudinal_result_retains_the_complete_fitted_graph_and_assessment` checks the round trip
against one cross-fitted, weighted, censored fit.
