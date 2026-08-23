# Intervention axes: three rounding policies, three estimands

"Round on everyone" is not a policy any ward would adopt. Wards target their scarce nursing time.
Prescribed rounding intensity changes by a step rather than switching on. And a program office
usually cannot mandate anything, so it nudges uptake instead.

`cleverly` runs all three through the point-treatment engine. They are not the same parameter, and
their result tables look almost identical. This test of change estimates all three and shows what
separates them.

Read the variation sections of
[Point-treatment TMLE](../technical-reference/point-treatment-tmle.md#variations) for the clever
covariates each axis produces.

## The applied question

The program office is planning next year's rounding standard. Three proposals are on the table.

| proposal | what it changes | which axis |
| --- | --- | --- |
| round hourly only on patients the acuity screen flags | who gets hourly rounding, as a function of a recorded variable | a known regime |
| raise every nurse's rounds per shift, but never past what staffing allows | how much of a continuous exposure each patient receives | a modified treatment policy |
| make the checklist easier to complete, so the odds a patient is rounded on double | the adoption mechanism itself, not any individual assignment | an incremental propensity-score intervention |

Each proposal is a different question about the world. None of them is the average treatment effect,
and a fourth number answering "round on all versus round on none" would not decide any of them.

This page analyses one hospital's patients, so patients are treated as independent.

## Why these are three estimands

| your situation | what the axis buys | what it costs |
| --- | --- | --- |
| the policy is a rule on recorded variables | a mean under a plan fixed before fitting, needing support only for the arm the rule assigns | the rule is part of the estimand. Two rules are two parameters |
| the exposure is continuous | the mean under a shift of the observed intensity, with the achievable maximum declared | positivity becomes a statement about the conditional density, and each shift has its own support |
| you can move uptake but not assign it | a tilt of the observed mechanism, which is what a real program can do | the target is defined through the observed mechanism, so its inference leans on estimating that mechanism well |

The common mistake is to read the three result tables as three estimates of one quantity. They
differ because the counterfactual worlds differ, not because the estimators disagree.

## A known regime: screen, then round

```python
from cleverly.datasets import make_nonlinear_ate

frame, truth = make_nonlinear_ate(n=3_000, seed=31)
frame = frame.rename(
    columns={
        "Y": "experience_score",
        "A": "hourly_rounding",
        "W1": "acuity",
        "W2": "prior_admissions",
        "W3": "length_of_stay",
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
        outcome="experience_score",
        treatment="hourly_rounding",
        adjustment=("acuity", "prior_admissions", "length_of_stay", "age"),
    ),
)
plans = (
    Static(0, name="round on none"),
    Static(1, name="round on all"),
    Rule(lambda data: (data["acuity"] > 0).astype(float), name="screen on acuity"),
)
regimes = study.identify(RegimeContrast(plans, reference="round on none"))
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

The screening contrast is smaller than the round-on-all contrast, because the rule reaches only part
of the ward. That is the number the office needs when it is budgeting nursing hours, and no
rescaling of the average treatment effect produces it.

A rule needs positivity only where it assigns. Low-acuity patients are never assigned hourly
rounding under this plan, so the fit never divides by their probability of being rounded on. A rule
can therefore be estimable on a ward where "round on all" is not.

```python
print(regime_result.diagnostics.support().summary())
```

## A modified treatment policy: add rounds per shift

Rounds per shift is a count, not a switch. The design says the exposure is continuous, and a
continuous exposure needs a conditional density rather than a propensity. `density_bins` controls
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
        "Y": "experience_score",
        "A": "rounds_per_shift",
        "W1": "acuity",
        "W2": "unit_census",
        "W3": "nurse_experience_years",
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
        outcome="experience_score",
        treatment="rounds_per_shift",
        adjustment=("acuity", "unit_census", "nurse_experience_years"),
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
different population values, because `cap=5.0` holds back the nurses whose new count would exceed
what the ward can staff. Those patients keep their current intensity under one policy and not under
the other.

The two estimates sit close together here, because only a few per cent of rows are capped. That does
not make the cap a detail. It makes the two policies nearly the same policy on this ward, which is a
fact about the data rather than about the estimand. A staffing ceiling inside the bulk of the
distribution would separate them.

`cap` has no default, and that is deliberate. Estimating the achievable maximum from the data would
make the parameter data-dependent. The reported standard error would then condition on a fitted
ceiling, and every bootstrap replicate would target a slightly different policy. The staffing office
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
toward the edge of what the ward has ever staffed is not made estimable by an estimator.

## An incremental intervention: make the checklist easier

The third proposal cannot be written as an assignment at all. The program office cannot decide which
patients get rounded on. It can simplify the checklist and put it in the workflow, which raises the
odds of rounding for everyone.

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

This is the axis that matches what a quality program actually controls, and it carries a warning
that the other two do not.

An incremental target is defined *through* the observed adoption mechanism. Doubling the odds of
rounding means doubling odds the data has to supply. This axis therefore has a one-sided robustness
property rather than the usual two-sided one. Its inference leans on estimating the mechanism
consistently, and a good outcome regression does not rescue it.

That is the opposite of the point-treatment situation, where the outcome regression is the safer of
the two nuisances to lean on. It is also the opposite of the [DR-TMLE](dr-tmle.md) situation, where
the mechanism was the one nobody could model. Read the
[incremental section](../technical-reference/point-treatment-tmle.md#incremental-propensity-score-interventions)
before reporting one.

## The three numbers side by side

```python
print("screen on acuity vs round on none:")
print(regime_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
print()
print("more rounds per shift vs current practice:")
print(shift_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
print()
print("easier checklist:")
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
| the per-policy support reports | which declared policies the ward's own data can carry | that a well-supported policy is the one worth adopting |
| the score-equation checks | each axis solved its own score equation | that the axis you chose matches the decision the office faces |
| the evidence manifest | each axis has its own exact-law, Gateaux, and remainder checks, with nonzero witnesses | a repeated-sampling study. The intervention axes have no row in the validation grid |

The evidence for these axes is in the
[evidence manifest](../technical-reference/evidence.md#the-table), under `ey_regime`, `ate_regime`,
`ey_shift`, `ate_shift`, `ey_ipsi`, and `ate_ipsi`. The incremental rows carry nonzero
treatment-score and one-sided remainder witnesses, which exist because a check at the truth alone
would be blind to a term that vanishes there.

The choice among the three axes is not a statistical question. It is a question about which change
the program office can actually make.
