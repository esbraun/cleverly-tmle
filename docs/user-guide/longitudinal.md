# Longitudinal treatment

## Declare temporal roles

`LongitudinalTreatment` aligns treatment nodes, time-varying histories, censoring indicators, and
outcome processes. At treatment node `t`, a dynamic rule sees only history available by `t`.

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
keys. A regimen cannot look ahead or change interpretation between stages.

With `n_folds > 1`, each outer training fold runs the complete backward recursion. The fit stores
the realized assignment in `result.folds`. Each sequential step stores its targeting details in
`step.fluctuation.folds`.

Each fold targets on its own training rows, so the fit does not drive the pooled score equation to
zero. `res.diagnostics.score_equations()` reports one row for the per-fold solves and one for the
stitched residual.

One repeated-sampling study observed standard-error ratios from 1.0170 to 1.1007 at `n=2000`.
Those results apply only to the named `make_longitudinal` law and estimator settings. They do not
establish conservative variance for other laws, weights, clusters, survival outcomes, or sample
sizes.

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

Longitudinal results provide stagewise cumulative support, targeting-score, and node-regression
loss reports. Point-only sensitivity formulas remain unavailable unless a longitudinal derivation
exists.

Tan (2025) derives population sensitivity bounds for binary, static longitudinal strategies, but
no sample estimator for them. The
[roadmap](../roadmap.md#f16-longitudinal-sensitivity-bound-estimation) records that boundary.

## Persistence

`result.save()` and `cleverly.load()` carry a longitudinal result through a round trip. The
artifact keeps the folds, the fitted mechanisms, the sequential steps, the targeting state, and
the causal metadata. [Persistence and
replayability](results-assessment.md#persistence-and-replayability) states the shared contract for
every result, including the assessment cache and the capability rows a restored artifact refuses.
`tests/unit/test_serialization.py`'s
`test_longitudinal_result_retains_the_complete_fitted_graph_and_assessment` checks the round trip
against one cross-fitted, weighted, censored fit.
