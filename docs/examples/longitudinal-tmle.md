# Longitudinal TMLE: rounding across two admissions, and members who leave

Rounding given twice is not rounding given once with extra columns. This test of change shows why,
by running the two analyses side by side on the same data. The point-treatment analysis is wrong in
both of the ways it can be wrong, and neither version can be fixed by adding or removing a term.

The page then follows the members who leave the plan. Churn appears three times, in three different
roles, and each role needs different machinery.

Read [Longitudinal TMLE](../technical-reference/longitudinal-tmle.md) for the sequential regression,
the cumulative clever covariate, and the event-process extensions.

## The applied question

The plan follows members through two admissions in a year. At each admission the member's ward
either uses hourly rounding or does not. Between the admissions the plan records how many service
issues the member raised and how many are still open.

Members also leave. Some disenroll before the second admission. Some leave before the year closes.

The program question is about a *plan*, not about a single stay. If every member had been cared for
under hourly rounding at both admissions, what share would give a top-box rating? Compare that with
the share if no ward had rounded at either.

## Why this method

The open service issues are the problem, and they are the whole problem.

| open service issues | what they do |
| --- | --- |
| respond to rounding at the first admission | a ward that rounds catches problems early, so the count is lower. That puts the issues on the causal path from the first admission's rounding to the rating |
| drive rounding at the second admission | a member with unresolved issues is flagged, and the second ward rounds on them. That makes the issues a confounder of the second admission's rounding |

Those two facts are incompatible with a single regression.

| what you do with the issue count | what goes wrong |
| --- | --- |
| adjust for it | you block part of the first admission's effect. The estimate is biased toward the null |
| leave it out | the second admission's rounding stays confounded. The estimate is biased the other way |

This is time-varying confounding. No choice of covariate set in one regression resolves it, because
the same variable must be conditioned on for one admission and not for the other.

Sequential regression resolves it by working backward through the nodes. Each node's regression
conditions on the history available *at that node*, and the result is averaged back over the earlier
history under the plan.

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| repeated rounding with time-varying confounders | the mean outcome under a plan, identified by the g-formula and estimated as a plug-in | one regression per node per regimen, and positivity is now a statement about a cumulative product |
| members disenroll over time | churn enters the same cumulative product as rounding | a churn model per node |
| the plan depends on the history | a dynamic rule receives the history available at its node | the rule is part of the estimand. Two rules are two parameters |

## The data

The generator is `make_longitudinal`. It produces a wide frame with one row per member and one
column per node, in time order. `cluster_size` puts members into hospitals, and the hospital effect
is genuine rather than decorative.

```python
from cleverly.datasets import make_longitudinal

frame, truth = make_longitudinal(n=8_000, seed=41, cluster_size=20)
frame = frame.rename(
    columns={
        "W1": "age",
        "W2": "baseline_experience",
        "A1": "rounding_admit1",
        "C1": "enrolled_admit2",
        "L2": "open_service_issues",
        "A2": "rounding_admit2",
        "C2": "enrolled_at_close",
        "Y": "top_box",
        "id": "hospital",
    }
)
print(frame.head())
for name, value in truth.items():
    print(f"{name:56s} {value:.4f}")
```

| column | node | role |
| --- | --- | --- |
| `age`, `baseline_experience` | before admission 1 | baseline covariates |
| `rounding_admit1` | admission 1 | the first ward's practice |
| `enrolled_admit2` | after admission 1 | 1 if the member is still in the plan |
| `open_service_issues` | between admissions | responds to `rounding_admit1`, and drives `rounding_admit2` |
| `rounding_admit2` | admission 2 | the second ward's practice |
| `enrolled_at_close` | after admission 2 | 1 if the member is still in the plan at year end |
| `top_box` | year end | the survey outcome |
| `hospital` | fixed | the cluster |

Nodes after a member's disenrollment are missing. That is the shape the estimator expects, and it is
why a complete-case frame would already have thrown information away.

## Design and identification

The design places every column at its node. `time_varying` is a sequence with one entry per node, so
the empty first entry says there is no time-varying covariate before the first admission.

```python
from cleverly import CausalStudy, LongitudinalTreatment, RegimeContrast

study = CausalStudy(
    frame,
    design=LongitudinalTreatment(
        outcome="top_box",
        treatment=("rounding_admit1", "rounding_admit2"),
        baseline=("age", "baseline_experience"),
        time_varying=((), ("open_service_issues",)),
        censoring=("enrolled_admit2", "enrolled_at_close"),
        cluster="hospital",
    ),
)
plan = RegimeContrast({"always": 1, "never": 0}, reference="never")
effect = study.identify(plan)
print(effect.summary())
for assumption in effect.identification.assumptions:
    print("-", assumption)
```

The placement of `open_service_issues` is the scientific decision on this page. It is declared as
time-varying at the second node. That single statement tells the estimator to condition on it when
modelling the second admission's rounding, and to average over it when propagating the first
admission's effect backward.

**Churn is declared as censoring here.** A member who disenrolls stops being observed, so their
later nodes are missing rather than zero. Putting churn in the `censoring=` role tells the estimator
to model it at each node and to carry it in the same cumulative product as rounding.

The assumptions change shape from the point-treatment case.

| assumption | what it becomes here |
| --- | --- |
| exchangeability | sequential. It must hold at every node, given the recorded history at that node |
| positivity | cumulative. Every member needs a positive probability of following the plan **and** staying enrolled, through both nodes |
| consistency | unchanged, plus no interference |

Cumulative positivity is the one that bites. Two nodes of rounding and two nodes of enrollment
multiply into one probability, and that product can be small even when no single factor is.

## Estimate

Four learner slots correspond to four kinds of nuisance. The pseudo-outcome learner fits the
intermediate regressions, whose targets are continuous even when the final outcome is binary.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CrossFitting, ModelSpec, Runtime, TMLEMethod

sequential = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LogisticRegression(max_iter=1000),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(n_folds=3, learner_folds=2),
    runtime=Runtime(random_state=41, n_jobs=1),
)
result = effect.estimate(method=sequential)
print(result.summary())
print("population contrast:", truth["ate_regimen[always vs never]"])
```

Every reported parameter carries a structured key rather than only a display label.

```python
for alias, key in result.parameter_keys.items():
    print(alias, "|", key.value, "vs", key.reference, "| horizon:", key.horizon)
```

Use those keys rather than parsing the alias string. A regimen name is chosen by the analyst, and
the alias is built from it.

## The failure mode: a point-treatment analysis of the same data

Now do it the wrong way, twice. Keep the members who stayed enrolled, keep those whose two wards
agreed, and treat "was rounded on at both admissions" as a single exposure.

```python
enrolled = frame[(frame["enrolled_admit2"] == 1) & (frame["enrolled_at_close"] == 1)]
consistent = enrolled[enrolled["rounding_admit1"] == enrolled["rounding_admit2"]].copy()
consistent = consistent.rename(columns={"rounding_admit1": "rounded_throughout"})
print("members kept:", len(consistent), "of", len(frame))
```

That subsetting is already a loss. Members who disenrolled are discarded rather than modelled, and
members whose two wards differed are discarded because a point-treatment analysis has nowhere to put
them.

```python
from cleverly import ATE, PointTreatment

target = truth["ate_regimen[always vs never]"]


def naive(adjustment, label):
    naive_study = CausalStudy(
        consistent,
        design=PointTreatment(
            outcome="top_box",
            treatment="rounded_throughout",
            adjustment=adjustment,
            cluster="hospital",
        ),
    )
    point = naive_study.estimate(
        ATE(reference=0),
        outcome_learner=LogisticRegression(max_iter=1000),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=3,
        learner_folds=2,
        random_state=41,
    )["ate"]
    low, high = point.ci
    print(
        f"{label:34s} psi={point.psi:6.3f}  CI=({low:.3f}, {high:.3f})  "
        f"covers={low <= target <= high}"
    )


naive(("age", "baseline_experience", "open_service_issues"), "adjusting for open issues")
naive(("age", "baseline_experience"), "baseline only")

point = result["ate_regimen[always vs never]"]
low, high = point.ci
print(
    f"{'sequential regression':34s} psi={point.psi:6.3f}  "
    f"CI=({low:.3f}, {high:.3f})  covers={low <= target <= high}"
)
print("population contrast:", target)
```

At the documented sample size the two naive analyses miss the population value in opposite
directions, and neither interval covers it.

| analysis | direction of the error | why |
| --- | --- | --- |
| adjusting for open issues | too small | the issue count is on the path from the first admission's rounding to the rating, so conditioning on it removes part of the effect being estimated |
| baseline only | too large | the second admission's rounding stays confounded by the issue count |
| sequential regression | covers the population value | each node conditions on its own history, and the earlier history is averaged over under the plan |

Neither naive analysis can be repaired by moving a term. That is the claim the method rests on, and
this is what it looks like on data.

The subsetting contributes a third error the table does not separate. Disenrolled members were
dropped, and dropping them is only harmless when churn is unrelated to the outcome. The longitudinal
fit instead models enrollment at each node and puts it into the same cumulative product as rounding.

## A rule instead of a plan

A dynamic rule reads the history available at its node. This one rounds on everyone at the first
admission, then keeps rounding only for members who came out of it with open issues.

A plan is one entry per node. An entry is either an arm for everybody, or a callable handed that
node's history frame. Mixing them is the ordinary case.

```python
from cleverly.datasets import RULE_LABEL

rule_effect = study.identify(
    RegimeContrast(
        {
            "always": 1,
            "never": 0,
            RULE_LABEL: (1, lambda history: (history["open_service_issues"] > 0).astype(float)),
        },
        reference="never",
    )
)
rule_result = rule_effect.estimate(method=sequential)
print(rule_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
print("population contrast:", truth[f"ate_regimen[{RULE_LABEL} vs never]"])
```

The first node is the constant `1`, so every ward rounds. The second node is a rule, so the issue
count decides who keeps being rounded on. `RULE_LABEL` is the name the generator publishes a truth
under, which is why it is imported rather than typed.

This is the targeted-rounding policy a program would actually consider, because nursing time is
scarce. No static plan expresses it, and no reweighting of the always-versus-never contrast produces
its value. The rule is part of the estimand, and two rules are two parameters.

## Churn as the outcome: a retention curve

Churn was a nuisance in the analysis above. Now make it the question. Does rounding keep members in
the plan?

The declaration changes shape rather than gaining a keyword. Passing **one outcome column per time
point** says the outcome is an absorbing event, and the fit reports cumulative risk at each horizon.

```python
from cleverly import RegimeMean
from cleverly.datasets import make_longitudinal_survival

churn_frame, churn_truth = make_longitudinal_survival(n=4_000, seed=52, cluster_size=20)
churn_frame = churn_frame.rename(
    columns={
        "W1": "age",
        "W2": "baseline_experience",
        "A1": "rounding_p1",
        "C1": "tracked_p2",
        "Y1": "disenrolled_p1",
        "L2": "open_service_issues",
        "A2": "rounding_p2",
        "C2": "tracked_close",
        "Y2": "disenrolled_p2",
        "id": "hospital",
    }
)
churn_study = CausalStudy(
    churn_frame,
    design=LongitudinalTreatment(
        outcome=("disenrolled_p1", "disenrolled_p2"),
        treatment=("rounding_p1", "rounding_p2"),
        baseline=("age", "baseline_experience"),
        time_varying=((), ("open_service_issues",)),
        censoring=("tracked_p2", "tracked_close"),
        cluster="hospital",
    ),
)
churn_result = churn_study.identify(
    RegimeMean({"always": 1, "never": 0}, horizons=(1, 2))
).estimate(method=sequential)
print(churn_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
```

Two roles that look alike are doing different jobs here. `disenrolled_*` is the **event**, and
`tracked_*` is the **censoring**: whether the plan could observe the member's status in that period
at all. A member the plan lost track of is censored. A member who left the plan had the event.

Read the same fit on the retention scale.

```python
print(churn_result.curve(scale="survival"))
```

`curve()` returns one row per regimen per horizon, with a `time` column that `to_frame()` does not
carry. The survival scale is not a relabelling. For a level it reports $1 - F$, mirroring the
estimate and the interval. For a contrast it negates the estimate, and it negates **and swaps** the
interval bounds. The standard error is the same either way.

At the documented sample size the retention curve separates. Members whose wards rounded at both
admissions stay enrolled at a visibly higher rate by the second period.

**Horizons are the fit's own time points, not days.** `horizons=(1, 2)` names the two nodes. Nothing
here models the 49-day survey window or any other duration, and asking for a horizon outside `1..T`
is refused rather than interpolated.

```python
churn_contrast = churn_study.identify(
    RegimeContrast({"always": 1, "never": 0}, reference="never", horizons=(1, 2))
).estimate(method=sequential)
print(churn_contrast.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
for name, value in churn_truth.items():
    if name.startswith("ate_"):
        print(f"{name:52s} {value:.4f}")
```

The contrast is negative, because rounding reduces cumulative disenrollment. Its size grows between
the two horizons.

## Two ways to leave, and one of them the program cannot touch

A member does not simply "churn". They leave in one of two ways, and the difference decides what the
program can claim.

| cause | what it is | can the program move it? |
| --- | --- | --- |
| open enrollment | the member chose another plan in the annual window | yes. This is the exit that patient experience is supposed to affect |
| other | the employer group terminated, or a qualifying life event ended eligibility | no. It is administrative, and it removes the member before they could ever choose |

The two are mutually exclusive and absorbing, which is what the container requires. Competing risks
are declared by the **shape** of `outcome=`: a mapping of cause to its indicator columns, one per
time point.

```python
from cleverly.datasets import make_longitudinal_competing

exit_frame, exit_truth = make_longitudinal_competing(n=4_000, seed=53, censoring=False)
exit_frame = exit_frame.rename(
    columns={
        "W1": "age",
        "W2": "baseline_experience",
        "A1": "rounding_p1",
        "L2": "open_service_issues",
        "A2": "rounding_p2",
        "D1": "open_enrollment_exit_p1",
        "D2": "open_enrollment_exit_p2",
        "R1": "other_exit_p1",
        "R2": "other_exit_p2",
    }
)
exit_study = CausalStudy(
    exit_frame,
    design=LongitudinalTreatment(
        outcome={
            "open enrollment": ("open_enrollment_exit_p1", "open_enrollment_exit_p2"),
            "other": ("other_exit_p1", "other_exit_p2"),
        },
        treatment=("rounding_p1", "rounding_p2"),
        baseline=("age", "baseline_experience"),
        time_varying=((), ("open_service_issues",)),
    ),
)
exit_result = exit_study.identify(
    RegimeContrast({"always": 1, "never": 0}, reference="never", horizons=(1, 2))
).estimate(
    method=TMLEMethod(
        models=ModelSpec(
            outcome_learner=LogisticRegression(max_iter=1000),
            pseudo_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
        ),
        cross_fitting=CrossFitting(n_folds=2, learner_folds=2),
        runtime=Runtime(random_state=53, n_jobs=1),
    )
)
print(exit_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
```

Two operational notes, because both are the kind of thing that stops a competing-risks fit rather
than biasing it.

The fold count drops to two here. Each cause needs a regression at each node, and the rarer cause is
thin. A fold whose training rows happen to contain no event of one cause leaves its learner with a
single class, and the fit stops. Fewer, larger folds is the fix. The number of **events of the
rarest cause**, not the number of members, is what bounds the split.

This section also runs without censoring, so the causes are the only way to leave the risk set. The
censoring machinery is unchanged from the first part of this page, and this generator ships no
cluster variant, so hospitals are not declared here.

At the documented sample size the two causes behave differently, and the contrast is the point.
Rounding at both admissions cuts the cumulative incidence of **open-enrollment** exits at both
horizons, and both intervals exclude zero. For **other** exits the effect is small, and by the
second horizon its interval contains zero.

That second row is a built-in negative control. A program cannot make an employer keep its group
plan. An analysis that reported a large effect on administrative exits would be telling you
something was wrong with the analysis, not something good about rounding.

The contrast fit reports differences. To see the levels the differences are built from, and to add
them up, ask for the means.

```python
exit_method = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LogisticRegression(max_iter=1000),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(n_folds=2, learner_folds=2),
    runtime=Runtime(random_state=53, n_jobs=1),
)
exit_levels = exit_study.identify(RegimeMean({"always": 1, "never": 0}, horizons=(1, 2))).estimate(
    method=exit_method
)
print(exit_levels.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
print(exit_levels.incidence_total())
```

`incidence_total()` sums the causes per regimen per horizon. It does not renormalise them onto a
simplex, because that would move each cause off the score equation the fit just solved. The `excess`
column is what a renormalisation would have hidden.

### The question the fit refuses

The tempting next question is what the open-enrollment loss would be if nobody's group had
terminated. That is not a setting on this estimand. It is a different estimand, and it is refused by
name.

```python
from cleverly.longitudinal import LTMLE

try:
    LTMLE({"always": 1, "never": 0}, eliminate="other")
except TypeError as error:
    print(error)
```

The refusal explains itself. What this fit reports is the cause-specific cumulative incidence with
the competing causes left alone, so an administrative exit is part of the history and enters the
clever covariate's indicator. Removing it would make it an intervened node, with a further factor
per node in the denominator, and its own exchangeability and positivity assumptions to state.

The refusal is also the scientifically right answer. A group termination is not something a rounding
program intervenes on, so a counterfactual world without them is not a world the program could
create.

## How far to trust this

The stage reports are tabular rather than prose, because each carries one row per regimen and per
node.

```python
print(result.diagnostics.run_all().summary())
```

```python
print(result.diagnostics.stagewise().to_frame())
```

The stagewise table is where cumulative positivity becomes visible. Read three of its columns
together.

| column | what it says |
| --- | --- |
| `n_followed` | how many members were still following the plan at that node |
| `effective_n` | the sample size the weights actually deliver, after the cumulative product |
| `share_truncated` | how much of the clever covariate the bound had to hold back |

An `effective_n` far below `n_followed` says the estimate rests on few members, whatever the row
count is. That is the longitudinal form of a positivity problem, and it grows with the number of
nodes.

```python
print(result.diagnostics.support().to_frame())
print(result.diagnostics.score_equations().to_frame())
print(result.validate().summary())
```

| layer | establishes | does not establish |
| --- | --- | --- |
| the stagewise report | how many members followed each plan, and how hard the weights worked | that sequential exchangeability holds at every node |
| the score-equation report | every node's fluctuation converged | that the node regressions are correctly specified |
| the registered study | the implementation recovers known two-node truths, matches R `ltmle`, and witnesses the targeting step itself | survival, competing risks, MSM, weights, clustering, or flexible learning. Those are outside its declared scope |

Read that last cell carefully against this page. The registered study covers the **end-of-study**
analysis only. The retention curve and the two-cause section on this page run machinery the study
does not reach, and their evidence is the exact-law and Gateaux work in the
[evidence manifest](../technical-reference/evidence.md#longitudinal-estimands-outside-the-target-registry)
rather than a repeated-sampling row.

The evidence row for the first part is
[end-of-study longitudinal TMLE](../technical-reference/method-evidence.md#end-of-study-longitudinal-tmle).
Positivity is comfortable throughout that study, so it does not speak to a fit whose stagewise
report shows a small effective sample size.

Two variants of this method have no longitudinal derivation.
[Collaborative TMLE](collaborative-tmle.md) and [DR-TMLE](dr-tmle.md) both refuse a longitudinal
design, and `available_methods()` says so before any model is fitted.

## Where to go next

This page reported one parameter per plan, and then one per plan per horizon. A program comparing
many rounding plans wants a summary instead. That is
[MSM projections](msm-projections.md), and the same projection works over regimens and horizons.
