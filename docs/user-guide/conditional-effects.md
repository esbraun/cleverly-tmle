# Conditional effects

`ConditionalContrast` answers questions such as these. In a trial, does the treatment help
older patients more than younger ones? In a marketing campaign, which customers does a coupon
move most, the *uplift* of each customer? Declare the baseline columns the effect may change
with, the *modifiers*, and a small set of fixed functions of them called the *basis*, and fit.
The result is a curve over the modifiers, with a confidence interval at each point, a band over
a grid you choose, and a test of whether the effect varies at all along the basis.

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

Every fit of a conditional effect is labelled `experimental` for now. The study that measures
how often these intervals cover the true curve is declared and has not run yet, so no one has
checked the intervals in repeated samples. They are computed the way the validated methods
compute theirs; read them with that caution, and see the status line of `fit.summary()`.

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
| `Cosine(columns, size)` | a smooth curve with `size` cosine terms per modifier | each domain in `Modifiers(domains=...)` |
| `BSpline(columns, n_interior_knots)` | a smooth curve with local flexibility | each domain |
| `Additive(left, right, ...)`, `Tensor(left, right)` | several modifiers side by side, or their interactions | what each part needs |

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

## Check the curve

`curve.diagnostics` checks a fitted curve against the data it came from. Each check refits the
same procedure on part of the rows and compares its predictions on the rows it left out with a
per-row estimate of each unit's effect built from the outcome and treatment models (a
*pseudo-outcome*, whose average over a group estimates the group's effect).

- `curve.diagnostics.calibration()`: the line from the held-out predictions to the
  pseudo-outcome, where a slope of 1 and an intercept of 0 mean the curve is calibrated, and a
  table of the two side by side within five bins of the prediction.
- `curve.diagnostics.risk_comparison()`: how well the curve predicts the effect on held-out
  rows, beside the average effect, the difference of the two outcome models, and, for a
  difference, a regression of the pseudo-outcome on the same basis. Lower is better.
- `curve.diagnostics.stability()`: how much the curve moves when the fit is repeated on other
  random splits. It warns when that movement is more than half the curve's own spread.
- `curve.diagnostics.modifier_stability()`: how much worse the held-out prediction gets when
  each modifier is left out of the basis.

These are single numbers with no intervals. They describe this fit and do not test it.

## Fit the effect without choosing a basis

When you have no shape in mind for the effect, leave `model=` out and fit `method="ep"`:

```python
effect = study.identify(ConditionalContrast(Modifiers(["age"], domains={"age": (18, 90)})))
curve = effect.estimate(
    "ep",
    outcome_learner=LogisticRegression(max_iter=1000),
    treatment_learner=LogisticRegression(max_iter=1000),
    random_state=0,
)
curve.predict({"age": [40, 60]})  # the estimated effect at those ages, with no interval
curve.summary()
```

This is the efficient plug-in (EP) learner of van der Laan, Carone and Luedtke (a preprint).
It fits the outcome and treatment models so that no row is predicted by a model fitted on it,
corrects the outcome models with one
weighted regression on smooth functions of the modifiers, and fits the curve to the corrected
difference between the arms. The correction is what keeps the curve on target when the
outcome model is wrong and the treatment model is right. It then stays on target but is noisier,
and when the outcome itself has little noise the curve's error can be several times what a right
outcome model would give. The smooth functions are cosines of
each modifier, so each continuous modifier needs a declared domain, and their number is chosen
by cross-validation. `sieve=Sieve(Cosine(["age"], 6), size=3)` fixes the number instead.

The EP curve has no confidence interval, and `curve.predict(rows, interval=True)` is refused.
Its theory says how fast the curve approaches the true effect, not how far off one fit can be.
Check it with `curve.diagnostics` as above, and `curve.diagnostics.score_equations()` shows that
the correction solved its equations. A difference curve is kept inside the outcome's range
(`curve.clipped_share` says how often that bound was hit). Every EP fit is labelled
experimental: its study is declared and has not run.

With `model=` declared, `method="ep"` estimates the same best line as the default fit, without
intervals, so the default fit is the better choice there.

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

An interval for the EP curve, a learner of your own as its last step on the log-ratio scale,
checks of how much hidden confounding would move the coefficients, missing treatment values, an
intermediate variable, `strata=`, repeated out-of-fold fits on new splits and the bootstrap each
refuse before any model is fitted and say what to do instead. Weights, clusters and missing
outcomes are accepted, and those fits are labelled experimental like every other.

The [conditional effects reference](../technical-reference/conditional-effects.md) gives
the exact targets, the formulas behind the intervals and the evidence.
