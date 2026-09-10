# Longitudinal TMLE: navigation at two decisions

Navigation assigned twice is not one assignment with extra columns. This test of change shows why,
by running the two analyses side by side on the same data. The point-treatment analysis is wrong in
both of the ways it can be wrong, and neither version can be fixed by adding or removing a term.

Loss from outcome tracking is a nuisance here, and this page models it as censoring.
[Time-to-event outcomes](longitudinal-survival.md) makes leaving the plan the outcome
instead.

Read [Longitudinal TMLE](../technical-reference/longitudinal-tmle.md) for the sequential regression,
the cumulative clever covariate, and the event-process extensions.

## The applied question

The plan makes two navigation decisions for each eligible discharge. The first occurs at discharge.
The second occurs on day seven. Between them, the program records unresolved medication,
appointment, and equipment issues.

Some patients are lost from outcome tracking before day seven or day 30. The transition score still
exists for them, but the plan cannot observe it. That is censoring, and this page treats it as a
nuisance.

The program question is about a *plan*, not one offer. What share would report a top-box transition
score if every patient received navigation at discharge and day seven? Compare that with the share
if nobody received navigation at either decision.

## Why this method

The unresolved transition issues are the problem, and they are the whole problem.

| unresolved transition issues | what they do |
| --- | --- |
| respond to discharge navigation | the first contact resolves some problems. This puts unresolved issues on the causal path from the first decision to the score |
| drive day-seven navigation | a patient with unresolved issues is more likely to receive the second contact. This makes the issues a confounder of the second decision |

Those two facts are incompatible with a single regression.

| what you do with the issue count | what goes wrong |
| --- | --- |
| adjust for it | you condition on a consequence of discharge navigation and can block part of its effect |
| leave it out | day-seven navigation remains confounded by the issue count |

This is time-varying confounding. No choice of covariate set in one regression resolves it, because
the same variable must be handled differently at the two decisions. The direction of either bias
depends on the data-generating law.

Sequential regression resolves it by working backward through the nodes. Each node's regression
conditions on the history available *at that node*, and the result is averaged back over the earlier
history under the plan.

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| repeated navigation with time-varying confounders | the mean outcome under a plan, identified by the g-formula and estimated as a plug-in | one regression per node per regimen, and positivity is now a statement about a cumulative product |
| outcome tracking ends over time | observation enters the same cumulative product as treatment | an observation model per node |
| the plan depends on the history | a dynamic rule receives the history available at its node | the rule is part of the estimand. Two rules are two parameters |

## The data

The generator is `make_longitudinal`. It produces a wide frame with one row per patient and one
column per node, in time order. `cluster_size` puts patients into navigator teams. The team effect is
genuine rather than decorative.

```python
from cleverly.datasets import make_longitudinal

frame, truth = make_longitudinal(n=8_000, seed=41, cluster_size=20)
frame = frame.rename(
    columns={
        "W1": "age",
        "W2": "baseline_readiness",
        "A1": "navigation_discharge",
        "C1": "tracked_day7",
        "L2": "unresolved_transition_issues",
        "A2": "navigation_day7",
        "C2": "tracked_day30",
        "Y": "transition_top_box",
        "id": "navigator_team",
    }
)
print(frame.head())
for name, value in truth.items():
    print(f"{name:56s} {value:.4f}")
```

| column | node | role |
| --- | --- | --- |
| `age`, `baseline_readiness` | before discharge | baseline covariates |
| `navigation_discharge` | discharge | the first navigation assignment |
| `tracked_day7` | after discharge | 1 if the patient remains observable at day seven |
| `unresolved_transition_issues` | between decisions | responds to `navigation_discharge`, and drives `navigation_day7` |
| `navigation_day7` | day seven | the second navigation assignment |
| `tracked_day30` | after day seven | 1 if the transition outcome remains observable |
| `transition_top_box` | day 30 | the survey outcome |
| `navigator_team` | fixed | the cluster |

Nodes after a patient is lost from tracking are missing. That is the shape the estimator expects, and it is
why a complete-case frame would already have thrown information away.

## Design and identification

The design places every column at its node. `time_varying` has one entry per node. The empty first
entry says there is no time-varying covariate before discharge.

```python
from cleverly import CausalStudy, LongitudinalTreatment, RegimeContrast, StudyProtocol

protocol = StudyProtocol(
    target_population=(
        "Adults discharged home from a participating hospital during the enrollment period"
    ),
    eligibility=(
        "Age 18 years or older",
        "Discharged alive",
        "Discharged home from a participating hospital",
    ),
    time_zero="Hospital discharge, after baseline measurement and before first assignment",
    treatment_strategies=(
        "Offer navigation at discharge and day seven",
        "Offer no navigation at either decision",
    ),
    treatment_versions=(
        "The declared discharge and day-seven navigation contacts",
        "Usual discharge support without navigation contacts",
    ),
    outcome="Top-box patient-reported transition score",
    horizon="30 days after discharge",
    intercurrent_event_handling=(
        "Use the transition score regardless of readmission",
        "Analyze each navigation offer regardless of completed contacts",
    ),
    interference_unit="Individual patient",
    assumption_rationale=(
        "Recorded history covers the measured common causes at each decision",
        "Version records support consistency at both navigation decisions",
        "Reserved navigator capacity supports no interference between patients",
    ),
)

study = CausalStudy(
    frame,
    design=LongitudinalTreatment(
        outcome="transition_top_box",
        treatment=("navigation_discharge", "navigation_day7"),
        baseline=("age", "baseline_readiness"),
        time_varying=((), ("unresolved_transition_issues",)),
        censoring=("tracked_day7", "tracked_day30"),
        cluster="navigator_team",
    ),
    protocol=protocol,
)
plan = RegimeContrast({"always": 1, "never": 0}, reference="never")
effect = study.identify(plan)
print(effect.summary())
for assumption in effect.identification.assumptions:
    print("-", assumption)
```

The identification summary renders the stored protocol. The typed `RegimeContrast` still owns the
mathematical comparison between the two treatment plans.

The placement of `unresolved_transition_issues` is the scientific decision on this page. It is declared as
time-varying at the second node. That single statement tells the estimator to condition on it when
modeling day-seven navigation, and to average over it when propagating the discharge effect
backward.

**Loss from outcome tracking is censoring here.** The transition outcome remains conceptually
defined, but the plan cannot observe it. The `censoring=` role models observation at each node and
carries it in the same cumulative product as navigation.

The assumptions change shape from the point-treatment case.

| assumption | what it becomes here |
| --- | --- |
| exchangeability | sequential. It must hold at every node, given the recorded history at that node |
| positivity | cumulative. Every patient needs a positive probability of following the plan **and** remaining observable, through both nodes |
| consistency | each decision uses the declared protocol version, and later treatment remains defined under maintained follow-up |
| no interference | one patient's assignments do not change another patient's protocol or outcome |

Cumulative positivity is the one that bites. Two navigation nodes and two observation nodes
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
    cross_fitting=CrossFitting(n_folds=3),
    runtime=Runtime(random_state=41, n_jobs=1),
)
result = effect.estimate(method=sequential)
print(result.summary())
print("population contrast:", truth["ate_regimen[always vs never]"])
```

The result summary renders the complete protocol and its fingerprint. The method settings remain
a separate record and cannot be changed by protocol text.

Save and load the fit to verify that the complete record survives the artifact round trip:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from cleverly import load

with TemporaryDirectory() as directory:
    saved = Path(directory) / "repeated-navigation.joblib"
    result.save(saved)
    restored = load(saved)
    restored_protocol = restored.identified_effect.protocol
    assert restored_protocol is not None
    print("\n".join(restored_protocol.summary_lines()))
    assert restored_protocol.fingerprint == restored.provenance.protocol_fingerprint
```

Use a maintained path instead of a temporary directory for a real audit artifact. Load only
joblib files you trust, and keep the dependency versions compatible.

Every reported parameter carries a structured key rather than only a display label.

```python
for alias, key in result.parameter_keys.items():
    print(alias, "|", key.value, "vs", key.reference, "| horizon:", key.horizon)
```

Use those keys rather than parsing the alias string. A regimen name is chosen by the analyst, and
the alias is built from it.

## The failure mode: a point-treatment analysis of the same data

Now do it the wrong way, twice. Keep patients observed through day 30, keep those whose two
assignments agreed, and treat "received navigation at both decisions" as one exposure.

```python
observed = frame[(frame["tracked_day7"] == 1) & (frame["tracked_day30"] == 1)]
consistent = observed[observed["navigation_discharge"] == observed["navigation_day7"]].copy()
consistent = consistent.rename(columns={"navigation_discharge": "navigation_throughout"})
print("patients kept:", len(consistent), "of", len(frame))
```

That subsetting is already a loss. Patients with missing follow-up are discarded rather than modeled.
Patients whose assignments differed are discarded because a point-treatment analysis has nowhere to
put them.

```python
from cleverly import ATE, PointTreatment

target = truth["ate_regimen[always vs never]"]


def naive(adjustment, label):
    naive_study = CausalStudy(
        consistent,
        design=PointTreatment(
            outcome="transition_top_box",
            treatment="navigation_throughout",
            adjustment=adjustment,
            cluster="navigator_team",
        ),
    )
    point = naive_study.estimate(
        ATE(reference=0),
        outcome_learner=LogisticRegression(max_iter=1000),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=3,
        random_state=41,
    )["ate"]
    low, high = point.ci
    print(
        f"{label:34s} psi={point.psi:6.3f}  CI=({low:.3f}, {high:.3f})  "
        f"covers={low <= target <= high}"
    )


naive(("age", "baseline_readiness", "unresolved_transition_issues"), "adjusting for open issues")
naive(("age", "baseline_readiness"), "baseline only")

point = result["ate_regimen[always vs never]"]
low, high = point.ci
print(
    f"{'sequential regression':34s} psi={point.psi:6.3f}  "
    f"CI=({low:.3f}, {high:.3f})  covers={low <= target <= high}"
)
print("population contrast:", target)
```

At the documented sample size the two shortcuts miss the population value in opposite directions.
Neither interval covers it.

| analysis | result on this draw | structural problem |
| --- | --- | --- |
| adjusting for open issues | too small | the regression conditions on a post-discharge variable and cannot represent both treatment decisions |
| baseline only | too large | the regression omits a cause of day-seven navigation and cannot represent both treatment decisions |
| sequential regression | covers the population value | each node conditions on its own history, and earlier history is averaged under the plan |

The shortcut also conditions on agreement between the two assignments. Agreement depends on the
time-varying history, so this selected sample is not the original target population. The two point
fits therefore do not isolate a pure mediator-adjustment bias from a pure confounding bias.

The shortcut then drops patients lost from follow-up. Dropping them is harmless only under
restrictive observation conditions. The longitudinal fit keeps the target population, models
observation at each node, and includes observation in the cumulative product.

## A rule instead of a plan

A dynamic rule reads the history available at its node. This one assigns navigation at discharge,
then assigns day-seven navigation only to patients with unresolved issues.

A plan is one entry per node. An entry is either an arm for everybody, or a callable handed that
node's history frame. Mixing them is the ordinary case.

```python
from cleverly.datasets import RULE_LABEL

rule_effect = study.identify(
    RegimeContrast(
        {
            "always": 1,
            "never": 0,
            RULE_LABEL: (
                1,
                lambda history: (history["unresolved_transition_issues"] > 0).astype(float),
            ),
        },
        reference="never",
    )
)
rule_result = rule_effect.estimate(method=sequential)
print(rule_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
print("population contrast:", truth[f"ate_regimen[{RULE_LABEL} vs never]"])
```

The first node is the constant `1`, so every patient receives discharge navigation. The second node
is a rule, so the issue count decides who receives day-seven navigation. `RULE_LABEL` is the name
the generator publishes a truth under. That is why it is imported rather than typed.

This is a targeted-navigation policy a program would actually consider, because navigator time is
scarce. No static plan expresses it, and no reweighting of the always-versus-never contrast produces
its value. The rule is part of the estimand, and two rules are two parameters.

## How far to trust this

Start with the diagnostic report. It records completed checks and the diagnostics that this fitted
family cannot run. This page calls `.diagnostics.run_all()` rather than `.assess()`, because every
sensitivity operation that `.assess()` adds is unavailable for a longitudinal fit.

```python
diagnostics = result.diagnostics.run_all()
print(diagnostics.summary())

support = diagnostics.report("support")
scores = diagnostics.report("score_equations")
nuisances = diagnostics.report("nuisance_models")

print(support.to_frame())
print(scores.to_frame())
print(nuisances.to_frame())
```

The report marks corrections as `not_applicable`. Longitudinal targeting does not use the
point-treatment correction system. It marks the truncation curve and refutation as `unavailable`.
No cost flag can supply the missing longitudinal implementations.

The support table is where cumulative positivity becomes visible. Read three of its columns
together.

| column | what it says |
| --- | --- |
| `n_followed` | how many patients were still following the plan at that node |
| `effective_n` | the sample size the weights actually deliver, after the cumulative product |
| `share_truncated` | how much of the clever covariate the bound had to hold back |

An `effective_n` far below `n_followed` says the estimate rests on few patients, whatever the row
count is. That is the longitudinal form of a positivity problem, and it grows with the number of
nodes. The direct `stagewise()` method remains an alias for this support report.

The score table has two rows per fitted stage because this fit uses cross-fitting. A `solver` row
checks the equations fitted outside each reporting fold. A `stitching` row checks the pooled
out-of-fold residual against its sampling scale. The stitched residual need not equal zero.

The nuisance table identifies each fitted model by role and node. Treatment and censoring models
appear once per node because all regimens share them. Outcome and pseudo-outcome models also carry
their regimen identity.

Treatment and censoring rows report weighted negative log likelihood. Outcome rows report weighted
Brier loss for a binary target, or mean squared error otherwise. Pseudo-outcome rows report weighted
mean squared error. The `evaluation` column is `out_of_fold` for this three-split fit.

A `completed` status means the retained losses are available. It is not a verdict that any
nuisance model is correct.

| layer | establishes | does not establish |
| --- | --- | --- |
| the support report | how many patients followed each plan, and how hard the weights worked | that sequential exchangeability holds at every node |
| the score-equation report | each fold solved its equation, and the stitched residual is compatible with sampling | that the node regressions are correctly specified |
| the nuisance report | retained loss and calibration for each fitted nuisance role | that any nuisance model is correct, or that causal identification holds |
| the registered studies | end-of-study fits recover known two-node truths and witness targeting, recursion, and held-out prediction | MSM, weights, clustering, simultaneous bands, or broad learner-library selection |

Read that last cell carefully against this page. Two registered rows cover the end-of-study
construction, the
[ordinary](../technical-reference/method-evidence/ordinary-end-of-study-longitudinal-tmle.md) and
the
[cross-fitted](../technical-reference/method-evidence/cross-fitted-end-of-study-longitudinal-tmle.md)
study. Positivity is comfortable in both. Neither row speaks to a fit whose support report shows
a small effective sample size.

Two variants of this method have no longitudinal derivation.
[Collaborative TMLE](collaborative-tmle.md) and [DR-TMLE](dr-tmle.md) both refuse a longitudinal
design, and `available_methods()` says so before any model is fitted.

## Where to go next

This page reported one parameter per plan, with the transition score as the outcome and loss to
tracking as a nuisance. Make leaving the plan the outcome instead, and the outcome becomes an event
that can happen at more than one time. That is
[time-to-event outcomes](longitudinal-survival.md), which reports a cumulative risk per horizon and
then splits it by cause.
