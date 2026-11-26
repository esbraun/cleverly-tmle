# Fitting choices for event times

The default fit of a time-to-event question estimates each risk with longitudinal TMLE on a grid of
check times. This page covers the estimation options that change how it gets there: a fit that
keeps the risk curve from dipping, a treatment and censoring model chosen by the data, an interval
that holds when one set of models is wrong, and other estimators to check the risk curve against.
None of them changes the question, which [risks at set check times](survival.md) declares.

The examples use the simulated data of that page, with one treatment at baseline and four check
times:

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, RegimeContrast, RegimeMean, TimeToEvent
from cleverly.datasets import make_point_survival

frame, truth = make_point_survival(n=1_000, seed=4, n_times=4)
design = TimeToEvent(
    time="time", event="event", treatment="A", baseline=("W1", "W2"), grid=(1, 2, 3, 4)
)
```

## Keep the risk curve from dipping

A risk by day 60 cannot be lower than the risk by day 30, but the default fit corrects each
horizon on its own, so its curve can dip slightly. Two options prevent that.

| option | what it does | what it needs |
| --- | --- | --- |
| `survival_model="hazard", targeting="one_step"` | fits the hazard at every time point and corrects one set of hazards for every horizon together | one treatment at baseline, held over every time point, or a treatment with categories at each time point. A baseline treatment can also be a dose under a plan that changes each person's dose. A covariate measured after baseline needs `targeting="one_step"` |
| `isotonic=True` | projects each fitted curve and its band limits onto a curve that never decreases | a longitudinal TMLE (`LTMLE`) survival or competing-risk fit without `msm=`. The other longitudinal estimators have no `isotonic=` |

Pass the hazard learner as `outcome_learner`. A hazard fit uses `pseudo_learner` only when the
design has a covariate measured after baseline, as the next section explains.

```python
hazard_curve = CausalStudy(frame, design=design).estimate(
    RegimeMean({"arm1": 1, "arm0": 0}),
    survival_model="hazard",
    targeting="one_step",
    outcome_learner=LogisticRegression(max_iter=1000),
    treatment_learner=LogisticRegression(max_iter=1000),
    censoring_learner=LogisticRegression(max_iter=1000),
    cross_fit=False,
)
print(hazard_curve.curve()[["estimand", "psi", "ci_lower", "ci_upper"]])
```

The risks of each arm do not decrease over the four grid times. The Settings block of
`summary(detail="full")` says whether the correction step converged; if it did not, the estimate
is not fully corrected, so say so when you report it. With the default `targeting="iterative"`, a hazard curve can
dip too.

To keep the default fit and project its curve, pass `isotonic=True`.

```python
projected = CausalStudy(frame, design=design).estimate(
    RegimeMean({"arm1": 1, "arm0": 0}),
    isotonic=True,
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    cross_fit=False,
)
print(projected.isotonic_moves)
```

`isotonic_moves` lists each risk the projection moved. On this sample it is empty, because the
fitted curve already does not decrease. A pointwise interval at a moved risk is valid only where the true
curve strictly increases.

:::{dropdown} Details: what the projection keeps
The projection keeps every influence curve and standard error of the fit; only the point
estimates and band limits move.
[The isotonic projection](../technical-reference/longitudinal-tmle.md#the-isotonic-projection)
states each claim and its condition.
:::

### Covariates measured after baseline

A covariate measured after baseline is one recorded again at later time points, such as a lab
value at each visit, declared with `time_varying=`. When it predicts both dropout and the event, the
fit must adjust for it. With one treatment at baseline, the hazard model takes such covariates under
`targeting="one_step"`. At each time point it fits the chance of the event, then a regression of what
happens later given the history so far, so it needs no model of the covariate itself.
`pseudo_learner` fits that regression. By default it reuses the `outcome_learner` library, so pass
`pseudo_learner` yourself when `outcome_learner` is a single classifier.

The answer it targets is the same as the default fit's, and when the models fit well both have the
same precision in large samples. What the hazard fit adds is a curve that never dips, and with
competing events, risks that add up to at most one for each person. With well-fitting models the gap
between the two estimates shrinks faster than their standard errors as the sample grows, so on a
large sample they nearly agree.

A treatment with categories at each time point fits beside such covariates too. A plan can fix the
treatment at each time point, follow a known rule that reads the latest covariate, or draw the
treatment with known probabilities that may read the covariate and the earlier treatment. At each
time point the fit predicts the event for every treatment level at each person's own history, so it
never builds the treatment sequences a plan could follow. A
[validation study](../technical-reference/method-evidence/hazard-time-varying-multinode.md)
measures this fit with a yes-or-no treatment at three time points and one yes-or-no covariate
measured before the treatment at each later time point, with one event type, in sample and with
cross-fitting.

A covariate declared at the first time point only is measured before the treatment, so it is an
ordinary confounder. The hazard model fits it, and gives the same estimate as declaring it in
`baseline=`. Under `markov_order=` the hazard model, like the default fit, drops that covariate
at later time points. `SurvivalCTMLE`, `SurvivalDRTMLE` and a `Separable` regimen refuse it; declare it
in `baseline=` for those. The
[covariates-after-baseline study](../technical-reference/method-evidence/hazard-time-varying.md)
measures the hazard fit with one yes-or-no covariate per later time point, for two fixed
treatments with one or two event types, in sample and with cross-fitting, and for a policy that
moves everyone up one treatment level.
[Covariates after baseline on the hazard path](../technical-reference/longitudinal-tmle.md#covariates-after-baseline-on-the-hazard-path)
gives the exact estimand, the settings it refuses, and its checks.

:::{dropdown} Details: what the hazard model refuses, and its studies
With a treatment at several time points and a covariate measured after baseline, the hazard model
refuses a plan that moves people to another treatment level, or one that multiplies the odds of
treatment when you supplied the treatment probabilities yourself
([H27 in the hazard refusals](../technical-reference/scope-and-refusals.md#hazard-and-isotonic-survival-refusals)).
Its message points to the default model, which fits that plan.

A treatment with categories at each time point also fits on the hazard model, when no covariate
is measured after baseline. A plan can fix the treatment at each time point, follow a rule that
reads the baseline covariates, or draw the treatment with known probabilities that may depend on
the earlier treatments. The fit corrects the hazards of every treatment sequence the plan can
follow. A plan that draws at many time points has many sequences, and a fit whose predicted
memory use passes 2 GiB is refused before any model is fitted. On a machine with more memory,
`max_memory_bytes=` on `LTMLE` or on `estimate()` sets a different limit, higher or lower.

If a trial design fixes the
treatment probabilities, you can pass them for the first time point only, as a mapping from the
first treatment column to its levels, such as
`treatment_probabilities={"A1": {0: "p0", 1: "p1"}}` with one probability column per level. An
array that covers every time point is refused, even with the later columns left empty
([H26 in the hazard refusals](../technical-reference/scope-and-refusals.md#hazard-and-isotonic-survival-refusals)).
The fit
estimates the probabilities of later treatment and of dropout, because it needs them along
sequences that nobody followed.

The
[multi-node hazard study](../technical-reference/method-evidence/hazard-multinode.md)
measures this fit with a yes-or-no treatment at three and at four time points, one event type and
no cross-fitting, under the default `targeting="iterative"`. Those fits are labelled `supported`.
Fits outside that, including the same regimens with `targeting="one_step"`, are labelled
`experimental`.
[A treatment at several nodes on the hazard path](../technical-reference/longitudinal-tmle.md#a-treatment-at-several-nodes-on-the-hazard-path)
gives the exact estimand and its checks.

A plan that changes each person's dose, such as `Shift(1.0, cap=2.0)`, also fits on the hazard
model. The treatment can be categories, a dose, or a dose recorded beside a yes/no treatment.
Each such plan gets its own correction of the hazards, so with `targeting="one_step"` its curve
never dips. The [hazard policy study](../technical-reference/method-evidence/hazard-mtp.md)
measures such fits in sample and with cross-fitting. With `ratio="density"`, which bins the
dose to compare the shifted and the observed dose, it covers only a dose with a fixed range,
because on an unbounded dose the outer bins are wider than the shift.

A few settings cannot be combined with such a plan, and
[the hazard refusals](../technical-reference/scope-and-refusals.md#hazard-and-isotonic-survival-refusals)
list them.
[A modified treatment policy on the hazard path](../technical-reference/longitudinal-tmle.md#a-modified-treatment-policy-on-the-hazard-path)
gives the exact estimand and its checks.
[Hazard-based survival targeting](../technical-reference/longitudinal-tmle.md#hazard-based-survival-targeting)
gives the contract, both algorithms, and the evidence.

Three registered studies measure hazard fits on a design with one baseline treatment:
[per-horizon targeting](../technical-reference/method-evidence/point-survival-hazard.md),
[one-step targeting](../technical-reference/method-evidence/point-survival-curve.md) and
[five-fold cross-fitting](../technical-reference/method-evidence/point-survival-curve-crossfit.md).
The first two agree with R packages fitted to the same samples (`survtmle` and `tmle3`). The
cross-fitted one has no comparison, because no package fits that construction.

The one-step study also holds one cell that fits the default iterated-mean model with
`isotonic=True` on a law with treatment at two time points. It fits every model by group
averages, and with such models the projection cannot move a risk. A second study,
[binding isotonic projection](../technical-reference/method-evidence/isotonic-binding.md), fits
logistic and linear models with a continuous covariate, so the projection moves a risk on about a
third of samples. There the 95% interval around the moved risk contained the true risk on 94% of
samples, inside the range the study accepts for a 95% interval. The moves it measured were at
most about half a standard error.
:::

## Let the data choose the treatment and censoring models

With many covariates, a treatment model that uses all of them can give some people a tiny
predicted chance of the treatment they got. Their weights then grow large and the estimate
becomes noisy. `method=CollaborativeTMLEMethod()` on a `TimeToEvent` design with a two-level
treatment builds the treatment and censoring models one covariate at a time instead, keeping a
covariate only when it improves the fit of the event data. It reports point estimates of the
risk, with no confidence interval.

A [simulation study](../technical-reference/method-evidence/survival-ctmle.md) on the laws of
Stitelman and van der Laan (2010) found the points unbiased when the event model is correct. With
a covariate that strongly predicts treatment but not the event, their typical error was 40% to
65% of that of the ordinary hazard fit with every covariate in the treatment model. When the
event model was wrong, some points were biased by up to about one standard deviation of the
estimate. `cleverly.plotting.learners(result)` draws each arm's sequence of added covariates with
the one the search chose marked, and `cleverly.plotting.overlap(result)` draws each arm's chosen
treatment model. [Methods and learners](methods-learners.md#method-presets) gives the settings, and
[Time-to-event outcomes](../technical-reference/collaborative-tmle.md#time-to-event-outcomes)
gives the algorithm and what it refuses.

## Keep the interval when one set of models is wrong

The ordinary hazard fit gives a valid confidence interval when the event model and the treatment
and censoring models are all close to the truth. `method=DRTMLEMethod()` adds a correction, from Díaz (2019), that
keeps the interval valid when only one side is right: the event model, or the treatment and
censoring models together. It fits a `TimeToEvent` design with a two-level treatment and one
event type. The side that is wrong still has to keep the ordering of people by
risk, or by chance of treatment, that its truth gives, and no fit can check that. When you know
the treatment and dropout probabilities, as in a randomized trial, give them to `TimeToEvent`
(`treatment_probabilities=` and `censoring_probabilities=`): the interval
then holds for any event model, as long as no one's chance of treatment or of staying in the
study is tiny.

A [simulation study](../technical-reference/method-evidence/survival-dr-tmle.md) with logistic
models on one law with five check times measured this. With one side wrong the points stayed
unbiased at 16,000 people, and with every model right the intervals covered 94% to 95% of the
time at 1,000 to 16,000 people. The correction loop did not always settle: in about 11% of the fits at 1,000 people
it stopped at its iteration cap with a warning, and those fits are part of the measured coverage.
Fit flexible treatment and censoring models with cross-fitting, the default: fitted on all rows
beside a wrong event model, the points were badly biased. [Time-to-event outcomes](../technical-reference/dr-tmle/theorem.md#time-to-event-outcomes)
gives the estimator, the conditions and what it refuses.

## Check the risk curve with other estimators

Other estimators reach the same risks by leaning on different models, so fitting a few of them
beside the default, `method="tmle"`, makes a quick check. A large gap between them says that one
set of models is off. The example fits four estimators to a second sample from the same
simulation; on your own data, fit them to the data of your main fit.

```python
from cleverly.learners import GLM

check_frame, check_truth = make_point_survival(n=1_000, seed=1, n_times=4)
check_study = CausalStudy(check_frame, design=design)
check_plans = RegimeContrast({"arm1": 1, "arm0": 0}, reference="arm0", horizons=(4,))
plain_models = dict(
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
)
models_by_method = {
    "tmle": plain_models,
    "sdr": plain_models,
    "gcomp": dict(outcome_learner=GLM(family="binomial"), pseudo_learner=GLM()),
    "ipw": dict(treatment_learner=GLM(family="binomial"), censoring_learner=GLM(family="binomial")),
}
for method, models in models_by_method.items():
    check_fit = check_study.estimate(check_plans, method=method, cross_fit=False, **models)
    check_estimate = check_fit["ate_regimen[arm1 vs arm0 @ t=4]"]
    lower, upper = check_estimate.ci
    print(
        method,
        round(check_estimate.psi, 3),
        round(lower, 3),
        round(upper, 3),
        check_estimate.support.label,
    )
print("truth", round(check_truth["ate_regimen[arm1 vs arm0 @ t=4]"], 3))
```

The four estimates of the drop in risk by time 4 agree to within 0.003, while each interval is
about 0.13 wide. Agreement is what you would expect if no set of models is badly off, but it
cannot show that any of them is right. No comparison can detect a confounder that the data do
not record. The `tmle` and `sdr` fits read the same models here, so the informative pair is
`gcomp`, which reads only the event models, against `ipw`, which reads only the treatment and
dropout models. Here the true drop, -0.127, sits inside every interval and about 0.02 from each
estimate. The last column is the support label: `cleverly` marks a result experimental when no
simulation study covers its setup, and of these four fits only the default reads `supported`.

The sequential g-computation and weighting fits get generalized linear models such as logistic
regression (`GLM`), because they report an interval only when every model they read is a `GLM`,
and that interval holds only when those models have the right form.
[Report the sequential baselines](longitudinal-estimators.md#report-the-sequential-baselines) explains both
estimators, and [their inference](../technical-reference/sequential-baselines.md#inference) says
what they report with other learners.

| estimator | `method=` | the estimate leans on |
| --- | --- | --- |
| sequentially doubly robust | `"sdr"` | the event models and the treatment and dropout models. It stays on target in large samples when, at each time point, either set is right |
| sequential g-computation | `"gcomp"` | the event models alone |
| sequential weighting (IPW) | `"ipw"` | the treatment and dropout models alone, so a large weight moves it |
| parametric g-formula | `"parametric_gformula"` | a model of each event's chance at each time point and of every covariate measured after baseline, from which it simulates each person forward. Each model must be right |

Each of these fits one event or competing events, with the same result keys as the default fit.
None takes `survival_model="hazard"` or `isotonic=True`, which are settings of the default fit
([hazard and isotonic survival refusals](../technical-reference/scope-and-refusals.md#hazard-and-isotonic-survival-refusals)).
A simulation study covers the parametric g-formula's competing-event risks on the wide layout,
with a treatment at each time point
([support label](../technical-reference/parametric-gformula.md#support-label)).
[Fit the sequentially doubly robust estimator](longitudinal-estimators.md#fit-the-sequentially-doubly-robust-estimator)
and [Simulate the parametric g-formula](longitudinal-estimators.md#simulate-the-parametric-g-formula) cover
the other two.

An incremental intervention (`method="ipsi"`) answers a different question: the risk had each
person's odds of treatment been multiplied by a factor at every time point, not the risk under
treating everyone against no one. It needs a treatment decided at each time point, on
`LongitudinalTreatment`, and refuses `TimeToEvent`
([IPSI refusals](../technical-reference/longitudinal-tmle.md#ipsi-refusals)). It is not doubly robust, so the treatment
model must be right.
[Estimate an incremental intervention over time](longitudinal-plans.md#estimate-an-incremental-intervention-over-time)
fits it.
