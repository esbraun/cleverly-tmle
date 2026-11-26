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
curve.plot(modifier="age")  # the curve over age, with its intervals and band
```

A simulation study checked these intervals in repeated samples: they covered the true
coefficients about 95% of the time, on both scales and with three treatment levels, weights,
clusters and missing outcomes. The `support` column of `fit.to_frame()` reads `supported` for a
fit like the one the study ran, with one outcome model per arm (`outcome_by_arm=True`), and
`curve.summary()` prints the same label on its `Status:` line. A second study checked one pooled
outcome model on plain rows, so that fit reads `supported` too. A combination no study ran (a
pooled model with weights, or weights together with clusters, for example) reads
`experimental`: it is computed the same way, and no study has checked it. The label appears in
`fit.summary()` only when it is experimental
([read the support label](results-assessment.md#read-the-support-label)).

The band over a grid has its own label, in its note in `curve.summary()` and in the notes of the
`curve.plot()` figure, because a study can check the coefficients without checking the band. A
study checked the default band (random signs as multipliers,
1,000 draws) on fits like the one above, on both scales, with clusters and with repeated
cross-fitting, and it held at all grid points together about 95% of the time. Other band
settings, and fits that study did not run, read `experimental`. `fit.summary()` also prints a
band over the coefficients themselves, and its own note says whether a study measured it
([bands](results-assessment.md#bands-that-cover-several-estimates-at-once)).

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
| `interactions(Additive(Linear(["age"]), Linear(["W2"])), order=2)` | the main terms plus their products, for an effect of one modifier that changes with another | what each part needs |
| `UserBasis(function, terms=..., columns=..., basis_kind="known")` | your own fixed function of the modifiers; it receives a mapping from each column name to its array of values | `basis_kind="known"`, which states that the function was fixed without reading this study's rows; without it the basis is refused |

`Constant()` alone has one term, the intercept, so its coefficient is the average effect on the
fit's scale.

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

`curve.plot(modifier="age")` draws the curve with its pointwise interval, the band at the grid
points, a tick at each row's age and, on the difference scale with an intercept, the average
effect. A straight line's interval widens away from the average age, because any error in the
slope grows with distance from the centre.

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

`curve.assess(include_refits=True)` runs every check in one report: the overlap and model checks
of the fit the curve came from, the check that its final adjustment step solved its equations,
and the four checks above. It adds the gain curve of `curve.diagnostics.targeting_curve()`,
which ranks rows held out of a refit by their predicted effect and reads how much effect the
top-ranked rows capture. With `include_retargets=True` it also runs the truncation curve, read
from the fit the curve came from, which refits the coefficients at each limit on the treatment
probabilities. `curve.fit.diagnostics.influence()` reads how much single rows sway each
coefficient. A check that cannot run on this curve, such as
`modifier_stability()` on a basis of one modifier, is listed with its reason.

The sensitivity analyses run on the fit the curve came from, and the curve's report points
there.

:::{dropdown} Details: sensitivity analyses and refutation on the fit

On the difference scale, and without `strata=`, that fit gives the bound on how far a
missing confounder could move a coefficient, its robustness value and its contour plot, as in
`fit.sensitivity.omitted_confounding("cate_projection[age]", cf_y=0.05, cf_d=0.05)`. No study
has checked these bounds for a coefficient, so their limits are descriptive.
`fit.diagnostics.refute(estimand="cate_projection[age]")` runs the
[refutation tests](refutation.md) on that fit too, and prints each reading with no pass or fail.
On the log-ratio scale, or with groups, the bound is refused. The E-value and the other analyses
written for an average effect are refused on every fit of a conditional contrast
([assessing a projection fit](../technical-reference/conditional-effects.md#assessing-a-projection-fit),
[refusals](../technical-reference/conditional-effects.md#refusals)). The
[sensitivity guide](sensitivity.md) explains each analysis.
:::

Every check needs the fit the curve came from, which `effect_curve()` and a saved curve keep. A
curve built without that fit refuses each check and asks you to fit it again.

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

On the difference scale the last step fits the curve to the corrected differences with least squares
on those functions. `final_learner=` replaces it with a learner of your own: a scikit-learn
regressor on the difference scale, or, on the log-ratio scale, a classifier with `predict_proba`
whose `fit` takes `sample_weight`. Such a fit reads `experimental`, because the theory does not
cover it.

```python
from sklearn.ensemble import GradientBoostingRegressor

boosted = effect.estimate(
    "ep",
    sieve=Sieve(Cosine(["age"], 6), size=lambda n: 3 if n < 2000 else 5),
    final_learner=GradientBoostingRegressor(max_depth=2, n_estimators=50),
    outcome_learner=LogisticRegression(max_iter=1000),
    treatment_learner=LogisticRegression(max_iter=1000),
    outcome_by_arm=True,
    random_state=0,
)
```

The status line of `curve.summary()` reads `supported` for a fit like the one the EP simulation
studies ran: a rule for the
number of functions, one outcome model per arm (`outcome_by_arm=True`), two arms, an estimated
treatment model and no weights, clusters or missing outcomes. The method's theory backs that
label only under the paper's conditions on the models and the rule, which you have to judge.

A fit outside that setting reads `experimental`. That covers the default, a number chosen by
cross-validation, and a final learner you supply, which the theory does not cover. It also covers
a pooled outcome model, weights, clusters, missing outcomes, three or more arms, treatment
probabilities you declare, and a fixed number of functions with `model=None`. They are computed
the same way, and no study has checked them.

The correction is what keeps the curve on target when the outcome model is wrong and the
treatment model is right.

:::{dropdown} Details: what the EP studies measured

The studies compared the curve with a reference fit that knew the true outcome and treatment models,
with the number of functions set by a rule like the one above. On the difference scale, at 4,000
rows, the curve's mean squared error was within about 8% of the reference's on six simulated
populations, and the gap did not grow from 1,000 to 4,000 rows. On the log-ratio scale its error was
no larger than the reference's. In the study the number of functions chosen by cross-validation did
about as well as the rule.

With a deliberately wrong outcome model, at the outcome noise of the
paper's simulations, the corrected curve's error was about twice the reference's, against about
twenty times without the correction. The gap depends on the noise: with a less noisy outcome the
reference's error shrinks and the wrong model's leftover error does not, and the ratio was about
four times (see the [study page](../technical-reference/method-evidence/ep-cate.md)).
:::

Check the curve with `curve.diagnostics` as above. A third study checked those checks on an EP
curve whose effect varies strongly: its held-out calibration slope came out at 0.99, and the
risk comparison picked the curve over the average effect in every run.
`curve.diagnostics.score_equations()` shows that the correction solved its equations. A
difference curve is kept inside the outcome's range (`curve.clipped_share` says how often that
bound was hit).
On an EP curve the truncation curve, refutation and sensitivity analyses
point to the default fit on `model=curve.model.basis`, which you fit yourself
([assessing a curve](../technical-reference/conditional-effects.md#assessing-a-curve)).

With `model=` declared, `method="ep"` estimates the same projection as the default fit, without
intervals. That fixed-basis fit reads `supported`, because the
[EP study](../technical-reference/method-evidence/ep-cate.md) measured it, but the default fit is
the better choice there, because it adds intervals.

`method="drtmle"` also fits the projection on the difference scale. Like the default fit, its
estimate is on target when either the outcome model or the treatment model is right. It adds extra
regressions on the fitted treatment probabilities and outcome predictions, which aim to keep the
interval valid when one of the two models is wrong. It reports intervals only when you pass the
true treatment probabilities, as in a trial with `treatment_probabilities=`; those intervals stay
valid when the outcome model is wrong, and they can be narrower than the default fit's. With
estimated treatment probabilities it reports the estimates without intervals, because its error
can then depend on the treatment model's error in a way no check on the data can rule out.
Its `curve.plot()` then draws the curve with no interval, and the figure's notes say why.

In a simulation with known treatment probabilities and a wrong outcome model, its 95% intervals
contained the true coefficients about 94% to 95% of the time
([projection methods study](../technical-reference/method-evidence/projection-methods.md)). The
[reference page](../technical-reference/conditional-effects.md#the-dr-tmle-projection) gives the
conditions and the cases it refuses.

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
    grid=[0.2, 0.4, 0.6, 0.8, 1.0],  # the treated fractions to report
    random_state=0,
)
gain.summary()
gain.plot()
```

The rows are sorted by the score, highest first. For each treated fraction $q$ in `grid=` (by
default every tenth, from 10% to 100%), the TOC column is the average effect among the top $q$ of
rows minus the average effect over everyone. A TOC above 0 means the score puts rows with larger
effects first. AUTOC averages the TOC over every fraction, and the Qini coefficient does the same
with more weight on large fractions. Each carries an interval, and the TOC column also carries a
band that holds at all the listed fractions at once. `gain.plot(view="gain")` draws the cumulative
effect against the straight line of treating at random.

A second column passed as `compare=`, with its own `compare_computed_from=`, is scored on the
same rows, and `gain.summary(detail="full")` shows the difference between the two scores
with its interval.
Every quantity is a difference of outcome means, whatever the scale of the score itself.

The intervals hold only for a score that was not fitted on this study's outcomes or
treatments, as for `assess_score`. To check a curve fitted on this study, use
`curve.diagnostics.targeting_curve(split=0.5)`. It refits the curve on half of the rows and
scores the other half, so the result describes that refit, not the curve fitted on every row.
A fraction with fewer than 50 rows in its top group reports its point without an interval.

:::{dropdown} Details: when the band is left out

When any fraction below 100% reports no interval, the band is left out at every fraction, so the
example drops 10%, whose top group has 40 rows here. At 100% the TOC is 0 by definition and
carries no interval, which does not withhold the band
([targeting curves](../technical-reference/conditional-effects.md#targeting-curves)).
:::

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

Four more settings fit the coefficients of a working model. A
[validation study](../technical-reference/method-evidence/projection-compositions.md) checked
groups and repeated splits with the default cross-fitting, and a missing treatment and the
bootstrap with `cross_fit=False`. Those read `supported`, and other combinations `experimental`
([status and evidence](../technical-reference/conditional-effects.md#status-and-evidence)).

- **Rows with no recorded treatment.** Declare the column that marks a recorded treatment with
  `PointTreatment(treatment_missingness=...)` and fit with `cross_fit=False`. Every row's
  covariates still count, and the fit reweights the rows with a recorded treatment by how likely
  a row like it was to have one. This is valid when whether the treatment was recorded depends
  only on the treatment and the covariates.
- **Effects within groups.** `PointTreatment(strata=["S"])` reports, next to the overall
  coefficients, each group's own coefficients, named like `cate_projection[age][S=1]`. The group
  column must also be an adjustment column, or the design is refused. Inside each group no basis
  column may be redundant, such as a modifier that is constant there, or the fit stops and names
  the group. `fit.heterogeneity_test(stratum="S=1")` and `fit.effect_curve(..., stratum="S=1")`
  read one group, named by its label.
- **Repeating the split.** `repeats=3` refits on three random splits of the rows and reports the
  median coefficient. Its standard error adds how much the coefficient moves between splits. The
  summary prints the standard error without that addition in the next column, for comparison
  (`result.se_without_displacement`).
- **The bootstrap.** `n_bootstrap=200` refits the whole analysis on 200 resamples of the rows
  and reports an interval from the spread of the refitted coefficients. Its interval reads
  `supported` with `cross_fit=False` and `experimental` when cross-fitted (`bootstrap_support`
  in `fit.to_frame()`).

## Hold a later variable fixed

Sometimes a variable that the treatment can change, such as whether a patient was admitted, sits
between the treatment and the outcome. The effect with that variable held at one value for
everyone is a *controlled direct effect*. Declare the variable as `intermediate=` and wrap the
contrast in `ControlledDirectEffect`, with the value to hold it at, 0 or 1:

```python
from cleverly import ControlledDirectEffect

study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y", treatment="A", adjustment=("age", "W2", "W3"), intermediate="Z"
    ),
)
direct = study.identify(
    ControlledDirectEffect(
        intermediate=0.0,
        contrast=ConditionalContrast(
            Modifiers(["age"], domains={"age": (18, 90)}), model=Linear(["age"])
        ),
    )
)
direct_fit = direct.estimate(
    outcome_learner=LogisticRegression(max_iter=1000),
    treatment_learner=LogisticRegression(max_iter=1000),
    intermediate_learner=LogisticRegression(max_iter=1000),  # P(Z = 1 | A, covariates)
    outcome_by_arm=True,
    random_state=0,
)
direct_fit.summary()  # "effect of A on Y with Z held at 0 for every unit"
```

The coefficients describe how the effect with `Z` held at 0 changes with age. The interval needs
two more conditions than the plain conditional effect: no unmeasured variable that affects both
`Z` and the outcome, given the treatment and the covariates, and some rows with `Z` at the held
value in every treatment arm at every covariate value. A simulation study checked this fit, with
one outcome model per arm and the default cross-fitting, and its intervals covered the true
coefficients about 95% of the time, so it reads `supported`. Other settings, such as weights or a
fit without cross-fitting, read `experimental`. The score checks and the held-out curve checks refuse on it,
because they would check the total effect instead.

## What is not supported yet

An interval for the EP curve, a variable held fixed together with groups (`strata=`) or with the EP
curve, and rows with no recorded treatment combined with cross-fitting each refuse before any model
is fitted and say what to do instead. The hidden-confounding checks on the log-ratio scale or with
groups refuse when you call them on the fitted result. A continuous treatment, or one with several
components, is refused too, because a conditional contrast compares two arms. Cut the treatment into
levels and declare `treatment=` and `reference=` on the contrast, or estimate a [curve over a
continuous dose](estimands.md#a-curve-over-a-continuous-dose). The reference's
[refusals](../technical-reference/conditional-effects.md#refusals) table lists the combinations that
the score checks and the targeting curves refuse.

## Go deeper

The [worked example](../examples/conditional-effects.ipynb) fits the EP curve on simulated data,
and [treatment rules](../examples/treatment-rules.ipynb) learns from records whom to treat.
Under [learned treatment rules](estimands.md#learned-treatment-rules),
`LearnedRule(loss="ep", modifiers=...)` fits this EP curve in each fold and treats where it is
above zero, and [its study](../technical-reference/method-evidence/learned-rule-ep.md) gives its
label.

The [conditional effects reference](../technical-reference/conditional-effects.md) gives
the exact targets, the formulas behind the intervals and the evidence.
