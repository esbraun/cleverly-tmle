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
    outcome_by_arm=True,  # one outcome model per treatment arm, as the validation study fits
    random_state=0,
)
fit.summary()  # one coefficient per basis term, each with a 95% interval
fit.heterogeneity_test()  # does the effect change with age at all?
curve = fit.effect_curve(grid={"age": [30, 50, 70]}, random_state=0)
curve.predict({"age": [40, 60]}, interval=True)  # the effect at new ages, with intervals
```

A simulation study checked these intervals in repeated samples: they covered the true
coefficients about 95% of the time, on both scales and with three treatment levels, weights,
clusters and missing outcomes. The status line of `fit.summary()` reads `supported` for a fit
like the one the study ran, with one outcome model per arm (`outcome_by_arm=True`). A fit with
one pooled outcome model, a combination the study did not run (weights together with clusters,
for example) and the band over a grid read `experimental`: they are computed the same way, and
no study has checked them.

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
chi-square test on as many degrees of freedom as there are such terms. With fewer than 40
clusters it uses an $F$ reference that allows for the few clusters, and it refuses a basis with
more non-constant terms than the number of clusters minus 2. A small p-value says the
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
- `curve.diagnostics.stability()`: how much the curve moves when the fit is repeated on the
  same rows with other random splits for the held-out fits. It warns when that movement is more
  than half the curve's own spread. It measures sensitivity to the split, not to the sample, so
  a low value does not mean the curve is well determined: a curve fitted to pure noise can read
  low. Whether the effect changes at all is what `fit.heterogeneity_test()` and the intervals
  of the default fit answer.
- `curve.diagnostics.modifier_stability()`: how much worse the held-out prediction gets when
  each modifier is left out of the basis.

These are single numbers with no intervals. They describe this fit and do not test it.

## Fit the effect without choosing a basis

When you have no shape in mind for the effect, leave `model=` out and fit `method="ep"`:

```python
from cleverly.bases import Cosine, Sieve

effect = study.identify(ConditionalContrast(Modifiers(["age"], domains={"age": (18, 90)})))
curve = effect.estimate(
    "ep",
    sieve=Sieve(Cosine(["age"], 6), size=lambda n: 3 if n < 2000 else 5),  # set before the fit
    outcome_learner=LogisticRegression(max_iter=1000),
    treatment_learner=LogisticRegression(max_iter=1000),
    outcome_by_arm=True,
    random_state=0,
)
curve.predict({"age": [40, 60]})  # the estimated effect at those ages, with no interval
curve.summary()
```

The result predicts the effect for each row you pass it, but it has no confidence interval, and
`curve.predict(rows, interval=True)` is refused. For intervals, declare a basis and use the
default fit above, or check a score with `assess_score` below.

This is the efficient plug-in (EP) learner of van der Laan, Carone and Luedtke (a preprint).
It fits the outcome and treatment models so that no row is predicted by a model fitted on it,
corrects the outcome models with one weighted regression on simple functions of the modifiers,
and fits the curve to the corrected difference between the arms. The functions are cosines of
each continuous modifier, which therefore needs a declared domain, and indicators of each
categorical one. `size=` sets how many: a number, or a rule in the sample size `n` as above.
Left out, the number is chosen by cross-validation. A number larger than the basis holds is
refused.

Two simulation studies checked the curve against a reference fit that knew the true outcome and
treatment models, with the number of functions set by a rule like the one above. On the
difference scale, at 4,000 rows, the curve's mean squared error was within about 8% of the
reference's on six simulated populations, and the gap did not grow from 1,000 to 4,000 rows. On
the log-ratio scale its error was no larger than the reference's. The status line of
`curve.summary()` reads `supported` for a fit like the one the studies ran: a rule for the
number of functions, one outcome model per arm (`outcome_by_arm=True`), two arms, an estimated
treatment model and no weights, clusters or missing outcomes. The method's theory backs that
label only under the paper's conditions on the models and the rule, which you have to judge.

A fit outside that setting reads `experimental`. That covers the default, a number chosen by
cross-validation, and a final learner you supply, which the theory does not cover. In the study
the cross-validated number did about as well as the rule. It also covers a pooled outcome model,
weights, clusters, missing outcomes, three or more arms, treatment probabilities you declare,
and a fixed number of functions with `model=None`. They are computed the same way, and no study
has checked them.

The correction is what keeps the curve on target when the outcome model is wrong and the
treatment model is right. With a deliberately wrong outcome model, at the outcome noise of the
paper's simulations, the corrected curve's error was about twice the reference's, against about
twenty times without the correction. The gap depends on the noise: with a less noisy outcome the
reference's error shrinks and the wrong model's leftover error does not, and the ratio was about
four times (see the [study page](../technical-reference/method-evidence/ep-cate.md)).

Check the curve with `curve.diagnostics` as above. A third study checked those checks on an EP
curve whose effect varies strongly: its held-out calibration slope came out at 0.99, and the
risk comparison picked the curve over the average effect in every run.
`curve.diagnostics.score_equations()` shows that the correction solved its equations. A
difference curve is kept inside the outcome's range (`curve.clipped_share` says how often that
bound was hit).

With `model=` declared, `method="ep"` estimates the same projection as the default fit, without
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

To choose between two scores, pass the second as `compare=`:

```python
both = effect.assess_score(
    "risk_score",
    computed_from=["age", "W2"],
    score_kind="external",
    compare="age_score",
    compare_computed_from=["age"],
    outcome_learner=LogisticRegression(max_iter=1000),
    treatment_learner=LogisticRegression(max_iter=1000),
)
both["risk_difference[risk_score - age_score]"]
```

The paired row is `risk_score`'s mean squared error for the CATE minus `age_score`'s. A negative
value means `risk_score` predicts the conditional effect better, and an interval that excludes
0 means the data can tell the two apart. The result also holds each score's own three numbers.
They come from the same fit as the paired row, so they can differ slightly from a call without
`compare=`; both are valid. The interval covers this one pair, named before you looked at the
results. Trying many scores and reporting the best one is a selection the interval does not
account for.

`check.status` reads `supported` for a check like the ones a simulation study ran and
`experimental` otherwise; [Status and evidence](../technical-reference/conditional-effects.md#status-and-evidence)
lists them. The check works on the difference scale or, with `scale="log_ratio"`, on the log
risk ratio scale.

The intervals hold only for scores that were not fitted on this study's outcomes or
treatments, which applies to both columns. Out-of-fold predictions from a model trained on these
rows do not count as external. `score_kind="external"` states that, and no code can check it. A column that
matches a curve fitted from this effect, or that carries the `effect_curve:` prefix of
`curve.to_frame()`, is refused.

## Check how well a score ranks who to treat first

When a score decides who is treated first, the useful question is how much more the treated
rows gain than a random choice of the same number would. `targeting_curve` answers it from the
study's outcomes, for a score column fitted on other data:

```python
gain = effect.targeting_curve(
    "risk_score",
    computed_from=["age", "W2"],
    score_kind="external",
    outcome_learner=LogisticRegression(max_iter=1000),
    treatment_learner=LogisticRegression(max_iter=1000),
    random_state=0,
)
gain.summary()
gain.plot()
```

The rows are sorted by the score, highest first. For each treated fraction $q$ (10%, 20%, up to
100%), the TOC column is the average effect among the top $q$ of rows minus the average effect
over everyone. A TOC above 0 means the score puts rows with larger effects first. AUTOC
averages the TOC over every fraction, and the Qini coefficient does the same with more weight
on large fractions. Each carries an interval, and the TOC column also carries a band that holds
at all the listed fractions at once. `gain.plot(view="gain")` draws the cumulative effect
against the straight line of treating at random.

A second column passed as `compare=`, with its own `compare_computed_from=`, is scored on the
same rows, and `gain.summary(detail="full")` shows the difference between the two scores
with its interval.
Every quantity is a difference of outcome means, whatever the scale of the score itself.

The intervals hold only for a score that was not fitted on this study's outcomes or
treatments, as for `assess_score`. To check a curve fitted on this study, use
`curve.diagnostics.targeting_curve(split=0.5)`. It refits the curve on half of the rows and
scores the other half, so the result describes that refit, not the curve fitted on every row.
A fraction with fewer than 50 rows in its top group reports its point without an interval.

Each row of the result is labelled `supported` or `experimental`, and the `status_reason` column
of `gain.to_frame()` says why. A row reads `supported` when a
[simulation study](../technical-reference/method-evidence/targeting-curves.md) checked its
setting and its intervals covered at close to 95%. That setting is a two-arm treatment, the default
cross-fitting, rows without weights, clusters or missing outcomes, and at least 200 rows in the
top group. The score comes from other data for a continuous or a binary outcome, or from the
half-and-half split for a continuous outcome. Known treatment probabilities
(`treatment_probabilities=`) were checked with a score from other data and a continuous outcome. The difference between two scores was checked for a continuous outcome
only, and its interval at 10% came out a little wide: it covered 96% of the time.

## Other settings for the coefficients

Four more settings fit the coefficients of a working model. Their numbers are computed, but they
are labelled `experimental` until their validation study has run.

- **Rows with no recorded treatment.** Declare the column that marks a recorded treatment with
  `PointTreatment(treatment_missingness=...)` and fit with `cross_fit=False`. Every row's
  covariates still count, and the fit reweights the rows with a recorded treatment by how likely
  a row like it was to have one. This is valid when whether the treatment was recorded depends
  only on the treatment and the covariates.
- **Effects within groups.** `PointTreatment(strata=["S"])` reports, next to the overall
  coefficients, each group's own coefficients, named like `cate_projection[W1][S=1]`. Each group
  needs every basis column to vary inside it, or the fit stops and names the group.
- **Repeating the split.** `repeats=3` refits on three random splits of the rows and reports the
  median coefficient. Its standard error adds how much the coefficient moves between splits, and
  `se_without_displacement` gives it without that addition for comparison.
- **The bootstrap.** `n_bootstrap=200` refits the whole analysis on 200 resamples of the rows
  and reports an interval from the spread of the refitted coefficients.

## What is not supported yet

An interval for the EP curve, a learner of your own as its last step on the log-ratio scale,
checks of how much hidden confounding would move the coefficients, an intermediate variable, and
rows with no recorded treatment combined with cross-fitting each refuse before any model is
fitted and say what to do instead.

The [conditional effects reference](../technical-reference/conditional-effects.md) gives
the exact targets, the formulas behind the intervals and the evidence.
