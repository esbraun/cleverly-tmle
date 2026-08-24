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
stitched residual. The reported standard error is conservative under this construction.

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

This projection accepts `n_folds > 1` and runs the same construction the regimen means do: one
complete alternation per outer fold, stitched on held-out rows. A saturated working model therefore
reproduces the per-regimen report at any fold count. Under a link each fold reaches its own
coefficient vector, in `result.msm_fits[k].alternation.folds`, and the reported one is the
projection of the stitched fit.

## Diagnostics

Longitudinal results provide stagewise cumulative support, targeting-score, and node-regression
loss reports. Point-only sensitivity formulas remain unavailable unless a longitudinal derivation
exists.
