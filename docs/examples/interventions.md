# Intervention axes: three navigation policies, three estimands

"Offer navigation to everyone" is not the only policy a program can use. Programs target limited
navigator time, change assigned support intensity, or alter a documented assignment probability.

`cleverly` runs all three through the point-treatment engine. They are not the same parameter, and
their result tables look almost identical. This test of change estimates all three and shows what
separates them.

Read the variation sections of
[Point-treatment TMLE](../technical-reference/point-treatment-tmle.md#variations) for the clever
covariates each axis produces.

## The applied question

The program office is planning next year's navigation standard. Three proposals are on the table.

| proposal | what it changes | which axis |
| --- | --- | --- |
| offer navigation only when a baseline discharge-risk screen flags | who gets the standard offer, as a function of recorded baseline information | a known regime |
| add assigned navigation hours, but never exceed a declared capacity per patient | how much of a continuous exposure each patient receives | a modified treatment policy |
| change the scheduling lottery so the conditional odds of an offer double | the assignment mechanism itself, not one fixed assignment | an incremental propensity-score intervention |

Each proposal is a different question about the world. None of them is the average treatment effect,
and a fourth number answering "offer to all versus offer to none" would not decide any of them.

Each policy keeps the shared eligibility, time zero, outcome window, and no-interference controls.
The risk rule uses baseline information only. The duration cap and scheduling odds are fixed before
fitting, so each intervention is well defined and reproducible.

## Why these are three estimands

| your situation | what the axis buys | what it costs |
| --- | --- | --- |
| the policy is a rule on recorded variables | a mean under a plan fixed before fitting, needing support only for the arm the rule assigns | the rule is part of the estimand. Two rules are two parameters |
| the exposure is continuous | the mean under a shift of the observed intensity, with the achievable maximum declared | positivity becomes a statement about the conditional density, and each shift has its own support |
| you can implement a stochastic assignment rule | a tilt of the observed mechanism | the target is defined through that mechanism, so its inference leans on estimating the mechanism well |

The common mistake is to read the three result tables as three estimates of one quantity. They
differ because the counterfactual worlds differ, not because the estimators disagree.

## A known regime: screen, then offer

```python
from cleverly.datasets import make_nonlinear_ate

frame, truth = make_nonlinear_ate(n=3_000, seed=31)
frame = frame.rename(
    columns={
        "Y": "transition_score",
        "A": "transition_navigation",
        "W1": "discharge_risk",
        "W2": "prior_utilization",
        "W3": "medication_burden",
        "W4": "age",
    }
)
print("population ATE:", truth["ate"])
```

The screening rule is written as a function of the covariates. It is declared before fitting, so it
is part of the question rather than a result of it.

```python
from cleverly import CausalStudy, PointTreatment, RegimeContrast
from cleverly.interventions import Rule, Static

study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="transition_score",
        treatment="transition_navigation",
        adjustment=("discharge_risk", "prior_utilization", "medication_burden", "age"),
    ),
)
plans = (
    Static(0, name="offer to none"),
    Static(1, name="offer to all"),
    Rule(lambda data: (data["discharge_risk"] > 0).astype(float), name="screen on risk"),
)
regimes = study.identify(RegimeContrast(plans, reference="offer to none"))
print(regimes.summary())
```

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CrossFitting, ModelSpec, Runtime, TMLEMethod

method = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(n_folds=3, learner_folds=2),
    runtime=Runtime(random_state=31, n_jobs=1),
)
regime_result = regimes.estimate(method=method)
print(regime_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
```

The screening contrast is smaller than the offer-to-all contrast, because the rule reaches only part
of the eligible population. That is the number the office needs when it budgets navigator hours. No
rescaling of the average treatment effect produces it.

A rule needs positivity only where it assigns. Lower-risk patients are never assigned navigation
under this plan, so the fit never divides by their probability of receiving an offer. A rule can
therefore be estimable where "offer to all" is not.

```python
print(regime_result.diagnostics.support().summary())
```

## A modified treatment policy: add navigation hours

Assigned navigation time over 30 days is continuous, not a switch. A continuous exposure needs a
conditional density rather than a propensity. `density_bins` controls
it.

```python
from cleverly.datasets import make_shift_dose

dose_frame, dose_truth = make_shift_dose(
    n=3_000,
    seed=32,
    shifts=(
        (0.0, None, "current practice"),
        (0.5, 5.0, "+0.5 capped at 5"),
        (0.5, None, "+0.5 uncapped"),
        (1.0, None, "+1.0 uncapped"),
    ),
)
dose_frame = dose_frame.rename(
    columns={
        "Y": "transition_score",
        "A": "navigation_hours_30d",
        "W1": "discharge_risk",
        "W2": "navigator_caseload",
        "W3": "baseline_support_need",
    }
)
for name, value in dose_truth.items():
    print(f"{name:52s} {value:.4f}")
```

Four policies are declared in one fit. The first changes nothing, so its mean is the observed mean.
The other three are the same increase under different statements about what staffing can actually
deliver.

```python
from cleverly import ModifiedTreatmentPolicyEffect
from cleverly.interventions import Shift

dose_study = CausalStudy(
    dose_frame,
    design=PointTreatment(
        outcome="transition_score",
        treatment="navigation_hours_30d",
        adjustment=("discharge_risk", "navigator_caseload", "baseline_support_need"),
        treatment_kind="continuous",
    ),
)
policies = (
    Shift(0.0, cap=None),
    Shift(0.5, cap=5.0, name="+0.5 capped at 5"),
    Shift(0.5, cap=None, name="+0.5 uncapped"),
    Shift(1.0, cap=None, name="+1.0 uncapped"),
)
shift_result = dose_study.estimate(
    ModifiedTreatmentPolicyEffect(policies),
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    density_bins=40,
    n_folds=3,
    learner_folds=2,
    random_state=32,
)
print(shift_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
```

## The failure mode: the cap is part of the question

Compare the two `+0.5` rows against the printed truths. They apply the same increase and they have
different population values, because `cap=5.0` holds back assignments whose new duration would
exceed five hours. Those patients keep their current intensity under one policy and not under
the other.

The two estimates sit close together here, because only a few per cent of rows are capped. That does
not make the cap a detail. It makes the two policies nearly the same in this program, which is a
fact about the data rather than about the estimand. A staffing ceiling inside the bulk of the
distribution would separate them.

`cap` has no default, and that is deliberate. Estimating the achievable maximum from the data would
make the parameter data-dependent. The reported standard error would then condition on a fitted
ceiling, and every bootstrap replicate would target a slightly different policy. The navigation office
states what is achievable, because that is a question about the world.

Now read the support report, which is published per policy rather than once for the fit.

```python
for policy, report in shift_result.diagnostics.support().items():
    print(report.summary())
    print()
```

Each declared shift moves the intensity into a different region of the conditional support, so each
one has its own positivity problem.

| policy | what the report shows | how to read it |
| --- | --- | --- |
| current practice | every ratio is one, and the effective sample size is the full sample | nothing was moved, so nothing was extrapolated |
| `+0.5 capped at 5` | a small share of rows capped, and the effective sample size well below the row count | the density ratio is doing real work, and the interval is wider than the row count suggests |
| `+1.0 uncapped` | the effective sample size collapses to a small fraction of the rows | the ratio is now concentrated on a few patients, and the estimate rests on them |

Every uncapped shift also raises a positivity warning during the fit. The warning names the share of
rows assigned an intensity above the largest one observed, because the outcome regression
extrapolates for those rows and identification needs the shifted intensity to be supported.

The last row of the table is the important one. The fit returns a number for `+1.0 uncapped`, and at
the documented sample size that number sits further from its population value than the others do.
The support report says why before you look at the estimate. A policy that pushes most of the mass
toward the edge of what the program has staffed is not made estimable by an estimator.

## An incremental intervention: change the scheduling lottery

The third proposal is not one fixed assignment. The program changes its documented scheduling
lottery so each patient's conditional odds of an offer double. This stochastic policy is the
intervention. No separate workflow change provides another path to the outcome.

```python
from cleverly import IncrementalEffect
from cleverly.interventions import Incremental

incremental_result = study.estimate(
    IncrementalEffect((Incremental(0.5), Incremental(2.0))),
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    n_folds=3,
    learner_folds=2,
    random_state=31,
)
print(incremental_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
```

This axis matches a program that controls assignment probabilities, and it carries a warning
that the other two do not.

An incremental target is defined *through* the observed assignment mechanism. Doubling the odds of
an offer means doubling odds the data has to supply. This axis therefore has a one-sided robustness
property rather than the usual two-sided one. Its inference leans on estimating the mechanism
consistently, and a good outcome regression does not rescue it.

The incremental target does not require ordinary treatment positivity. Its weights remain bounded
when the observed probability approaches zero or one. That protection does not make the assignment
mechanism optional, because the mechanism defines the estimand itself.

That is the opposite of the point-treatment situation, where the outcome regression is the safer of
the two nuisances to lean on. It is also the opposite of the [DR-TMLE](dr-tmle.md) situation, where
the mechanism was the one nobody could model. Read the
[incremental section](../technical-reference/point-treatment-tmle.md#incremental-propensity-score-interventions)
before reporting one.

## The three numbers side by side

```python
print("screen on risk vs offer to none:")
print(regime_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
print()
print("more navigation hours vs current practice:")
print(shift_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
print()
print("doubled assignment odds:")
print(incremental_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
```

Three tables, three column headings that read the same, three different parameters. The estimand
name in the first column is what distinguishes them, and it is the part to carry into the program
report.

One composition is refused rather than approximated. A fit cannot combine axes, because one
fluctuation solves one set of score equations. A result reporting parameters from two axes would put
two of them under one heading.

## How far to trust this

```python
print(regime_result.validate().summary())
print(shift_result.validate().summary())
print(incremental_result.validate().summary())
```

| layer | establishes | does not establish |
| --- | --- | --- |
| the per-policy support reports | which declared policies the program data can carry | that a well-supported policy is worth adopting |
| the score-equation checks | each axis solved its own score equation | that the axis you chose matches the decision the office faces |
| the evidence manifest | each axis has its own exact-law, Gateaux, and remainder checks, with nonzero witnesses | a repeated-sampling study. The intervention axes have no row in the validation grid |

The evidence for these axes is in the
[evidence manifest](../technical-reference/evidence.md#the-table), under `ey_regime`, `ate_regime`,
`ey_shift`, `ate_shift`, `ey_ipsi`, and `ate_ipsi`. The incremental rows carry nonzero
treatment-score and one-sided remainder witnesses, which exist because a check at the truth alone
would be blind to a term that vanishes there.

The choice among the three axes is not a statistical question. It is a question about which change
the program office can actually make.
