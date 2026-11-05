# Conditional effects

`ConditionalContrast` asks how the effect of a treatment changes with a few baseline columns,
the *modifiers*. Declare the modifiers, a small set of fixed functions of them called the
*basis*, and fit. The result is a curve over the modifiers, with a confidence interval at each
point, a band over a grid you choose, and a test of whether the effect varies at all along the
basis.

```python
from sklearn.linear_model import LogisticRegression
from cleverly import CausalStudy, ConditionalContrast, PointTreatment
from cleverly.bases import Linear, Modifiers

study = CausalStudy(
    frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("age", "W2", "W3"))
)
effect = study.identify(
    ConditionalContrast(Modifiers(["age"], domains={"age": (18, 90)}), model=Linear(["age"]))
)
fit = effect.estimate(
    outcome_learner=LogisticRegression(max_iter=1000),
    treatment_learner=LogisticRegression(max_iter=1000),
    random_state=0,
)
fit.summary()  # one coefficient per basis term, each with a 95% interval
fit.heterogeneity_test()  # does the effect change with age at all?
curve = fit.effect_curve(grid={"age": [30, 50, 70]}, random_state=0)
curve.predict({"age": [40, 60]}, interval=True)  # the effect at new ages, with intervals
```

Every modifier must also be an adjustment column. A column outside the adjustment set is
refused, because the effect given a column can be worked out from the outcome model only when
that column is one of the confounders adjusted for.

## What the curve estimates

On the default difference scale the question is the conditional average treatment effect
(CATE): the average difference between the outcome under treatment and under control among
units with the same modifier values. With `model=Linear(["age"])` the fit reports the straight
line in age that is closest to that effect, in mean squared error over the study's age
distribution. If the true effect is a straight line in age, the line is the effect. If it
curves, the line is the best straight-line summary, and its slope is the average change in
effect per year of age in that sense.

`scale="log_ratio"` asks for the log of the ratio of the two outcome means given the
modifiers, the log conditional risk ratio for a binary outcome. A continuous outcome on this
scale must be nonnegative, with its declared range starting at 0 (`q_bounds=(0, upper)`),
because a ratio of means changes when the outcome is shifted.

A treatment with more than two levels contrasts one declared pair:
`ConditionalContrast(..., treatment="high", reference="low")`.

## Choose a basis

The basis is fixed before the fit, so the coefficients carry ordinary confidence intervals.

| basis | use it for | needs |
| --- | --- | --- |
| `Linear(columns)` | a straight-line trend in each modifier | nothing more |
| `Indicators(column)` | one effect per level of a categorical modifier | the levels in `Modifiers(levels=...)` |
| `Cosine(columns, size)` | a smooth curve with `size` wiggles per modifier | each domain in `Modifiers(domains=...)` |
| `BSpline(columns, n_interior_knots)` | a smooth curve with local flexibility | each domain |
| `Additive((...))`, `Tensor(left, right)` | several modifiers side by side, or their interactions | what each part needs |

A domain or a set of levels must be declared rather than read off the data. A basis whose knots
came from the sample would make the target depend on the sample, and the interval would miss
that. `Modifiers(..., outside="clip")` maps a value outside its domain to the nearest end
instead of refusing it, for both the fit and `predict`.

More terms follow the true effect more closely and give wider intervals. A size chosen by
cross-validation (`Sieve(basis)` with its default `size="cv"`) is refused, because no reviewed
result gives an interval after that choice. A fixed `Sieve(basis, size=3)` is accepted.

## Read the result

`fit.coefficients()` lists the coefficients. On the log-ratio scale,
`fit.coefficients(scale="ratio")` exponentiates them: the intercept to a baseline ratio, each
slope to a ratio per unit of its modifier.

`fit.heterogeneity_test()` tests whether every coefficient but the intercept is zero, with a
chi-square test on as many degrees of freedom as there are such terms. A small p-value says the
effect changes along the basis. A large one does not say the effect is constant: it can change
in a way the basis cannot draw.

`fit.effect_curve(grid)` returns the curve. `curve.predict(rows)` evaluates it at new rows, and
`interval=True` adds a 95% interval at each row. With `grid=` the curve also carries a
simultaneous band over those points, which holds at all of them together; it is wider than the
pointwise intervals. `curve.summary()` prints the coefficients, the spread of the in-sample
predictions and the band. `curve.save(path)` writes it, and `cleverly.load(path)` reads it back.

## Check a score fitted elsewhere

A risk score built on other data can be checked against this study's conditional effect:

```python
check = effect.assess_score(
    "risk_score",
    computed_from=["age", "W2"],
    score_kind="external",
    outcome_learner=LogisticRegression(max_iter=1000),
    treatment_learner=LogisticRegression(max_iter=1000),
)
check.to_frame()
```

`risk_difference` compares the score's mean squared error for the CATE with that of the
average effect; a negative value means the score predicts the conditional effect better than
the average does. `calibration[(intercept)]` and `calibration[risk_score]` are the line from the
score to the effect; an intercept of 0 and a slope of 1 mean the score is calibrated. Both carry
intervals.

The intervals hold only for a score that was not fitted on this study's outcomes or
treatments. `score_kind="external"` states that, and no code can check it. A column that
matches a curve fitted from this effect, or that carries the `effect_curve:` prefix of
`curve.to_frame()`, is refused.

## What is not supported yet

The curve without a basis (`model=None`), the efficient plug-in learner (`method="ep"`),
checks of how much hidden confounding would move the coefficients, missing treatment values, an
intermediate variable, `strata=`, repeated out-of-fold fits on new splits and the bootstrap each
refuse before any model is fitted and say what to do instead. Weights, clusters and missing outcomes are supported.

The [conditional effects reference](../technical-reference/conditional-effects.md) gives
the exact targets, the formulas behind the intervals and the evidence.
