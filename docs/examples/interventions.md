# Intervention axes: three navigation policies, three estimands

A program can target navigator time, raise assigned intensity, or change assignment odds. Each
policy defines its own estimand, and the result tables look alike. The
[variations](../technical-reference/point-treatment-tmle.md#variations) give each clever covariate.

## The applied question

The program office has three proposals for next year's navigation standard.

| proposal | what it changes | which axis |
| --- | --- | --- |
| offer navigation only when a baseline discharge-risk screen flags the patient | who gets the standard offer, as a function of recorded baseline information | a known regime |
| raise the navigation intensity assigned at discharge, but never above a declared capacity | how much of a continuous exposure each patient receives | a modified treatment policy |
| change each patient's assignment probability from its current odds to twice those odds | the assignment mechanism itself, not one fixed assignment | an incremental propensity-score intervention |

None of them is the average treatment effect. The regime and the incremental policy keep the
[shared study design](index.md#the-shared-study-design). The intensity policy uses its own
continuous synthetic law. The office fixes the rule, the cap, and the odds multiplier before
fitting.

## Why these are three estimands

| your situation | what the axis buys | what it costs |
| --- | --- | --- |
| the policy is a rule on recorded variables | a mean under a plan fixed before fitting, needing support only for the arm the rule assigns | the rule is part of the estimand. Two rules are two parameters |
| the exposure is continuous | the mean under a shift of the observed intensity, with the achievable maximum declared | positivity becomes a statement about the conditional density, and each shift has its own support |
| you can implement a stochastic assignment rule | a tilt of the observed mechanism | the target is defined through that mechanism, so its inference leans on estimating the mechanism well |

The three results differ because the counterfactual worlds differ, not because the estimators
disagree.

## A known regime: screen, then offer

```python
from cleverly.datasets import make_nonlinear_ate, nonlinear_dgp

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
law = nonlinear_dgp()


def effect_of_offer(latent):
    return law.outcome_mean(latent, 1.0, None) - law.outcome_mean(latent, 0.0, None)


screen_truth = law.expectation(lambda latent: effect_of_offer(latent) * (latent[:, 0] > 0))
print("population offer-to-all contrast:", truth["ate"])
print("population screening contrast:", screen_truth)
```

The baseline covariates are standardized (mean 0, SD 1). The screening rule offers navigation when
`discharge_risk` is above average. `law.expectation` integrates the known law, which a real
program cannot do.

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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from cleverly import CrossFitting, ModelSpec, Runtime, TMLEMethod

method = TMLEMethod(
    models=ModelSpec(
        outcome_learner=HistGradientBoostingRegressor(random_state=31),
        treatment_learner=HistGradientBoostingClassifier(random_state=31),
    ),
    cross_fitting=CrossFitting(n_folds=3),
    runtime=Runtime(random_state=31, n_jobs=1),
)
regime_result = regimes.estimate(method=method)
print(regime_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
regime_assessment = regime_result.assess()
```

In this law the effect grows with `discharge_risk`, so the screen keeps most of the benefit. Its
population contrast is still smaller than the offer-to-all contrast. That is the number the office
needs when it budgets navigator hours. No rescaling of the average treatment effect produces it.

A rule needs positivity only where it assigns. Lower-risk patients receive usual support under this
plan, so the rule never divides by their probability of an offer. This fit also estimates "offer to
all" against "offer to none", and those two plans need support in both arms.

```python
regime_support = regime_assessment.report("support")
print(regime_support.summary())
print(regime_support.regimes["screen on risk"].score_load)
```

The policy row separates assigned-arm support from fitted score load. The first describes the
estimated treatment law where the rule assigns. The second describes concentration of the exact
absolute residual multipliers used by that policy's targeting equation.

## A modified treatment policy: raise assigned intensity

The office sets a navigation intensity for each patient at discharge. A continuous exposure needs a
conditional density rather than a propensity, and `density_bins` sets its resolution.

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
        "A": "assigned_navigation_intensity",
        "W1": "discharge_risk",
        "W2": "navigator_caseload",
        "W3": "caregiver_support",
    }
)
for name, value in dose_truth.items():
    print(f"{name:52s} {value:.4f}")
```

This law has its own synthetic units, centered on a program baseline, so some intensities are
negative. Intensity is normal with SD 1 given the covariates.

Four policies are declared in one fit. The first changes nothing, so its mean is the observed mean.
The other three apply an increase under different statements about what staffing can deliver.

```python
import warnings

from cleverly import ModifiedTreatmentPolicyEffect, PositivityWarning
from cleverly.interventions import Shift

dose_study = CausalStudy(
    dose_frame,
    design=PointTreatment(
        outcome="transition_score",
        treatment="assigned_navigation_intensity",
        adjustment=("discharge_risk", "navigator_caseload", "caregiver_support"),
        treatment_kind="continuous",
    ),
)
policies = (
    Shift(0.0, cap=None, name="current practice"),
    Shift(0.5, cap=5.0, name="+0.5 capped at 5"),
    Shift(0.5, cap=None, name="+0.5 uncapped"),
    Shift(1.0, cap=None, name="+1.0 uncapped"),
)
shift_effect = dose_study.identify(ModifiedTreatmentPolicyEffect(policies))
print(shift_effect.summary())
shift_method = TMLEMethod(
    models=ModelSpec(
        outcome_learner=HistGradientBoostingRegressor(random_state=32),
        treatment_learner=HistGradientBoostingClassifier(random_state=32),
        density_bins=40,
    ),
    cross_fitting=CrossFitting(n_folds=3),
    runtime=Runtime(random_state=32, n_jobs=1),
)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", PositivityWarning)
    shift_result = shift_effect.estimate(method=shift_method)
positivity_warnings = [str(w.message) for w in caught if w.category is PositivityWarning]
print("\n".join(positivity_warnings))
print(shift_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]].to_string(index=False))
shift_assessment = shift_result.assess()
```

## The failure mode: the cap is part of the question

Compare the two `+0.5` rows against the printed truths. They apply the same increase and have
different population values. `cap=5.0` holds back each patient whose new intensity would exceed 5.
Those patients keep their current intensity under one policy and not under the other.

The two estimates sit close together here, because only a few percent of rows are capped. That
makes the two policies nearly the same in this program. It is a fact about the data, not about the
estimand. A staffing ceiling inside the bulk of the distribution would separate them.

`cap` has no default. The office states what is achievable, because that is a question about the
world. The [shift section](../technical-reference/point-treatment-tmle.md#modified-treatment-policies)
explains why a cap estimated from the data would change the parameter.

Now read the support report, which is published per policy rather than once for the fit.

```python
for policy, report in shift_assessment.report("support").items():
    print(report.summary())
    print(report.score_load)
    print()
```

The density ratio omits observation weights. `score_load` includes the exact weights and score mask
used by the fit. Read both quantities, because either one can be the more concentrated.

| policy | what the report shows on this draw | how to read it |
| --- | --- | --- |
| current practice | every ratio is one, and the effective sample size is the full sample | nothing was moved, so nothing was extrapolated |
| `+0.5 capped at 5` | about 2.5% of rows capped, and an effective sample size of about 45% of the rows | the density ratio is doing real work, and the interval is wider than the row count suggests |
| `+1.0 uncapped` | an effective sample size of a few percent of the rows | the estimated ratio is concentrated on a few patients, and the estimate rests on them |

The fit printed a `PositivityWarning` for each nonzero uncapped shift. On this draw, a handful of
rows receive an intensity above the observed maximum, and the outcome regression extrapolates there.

Intensity is normal with SD 1 given the covariates, so every shift is identified in the population.
With the true density, the Kish fraction of a shift of size δ is exp(-δ²), about 37% for `+1.0`.
The estimated binned density concentrates the ratio far more. That is practical positivity strain
in a finite sample, and it grows with the size of the shift.

On this draw, the `+1.0 uncapped` interval is more than three times as wide as either `+0.5`
interval. The wider interval is consistent with the strain. The response is quadratic in intensity,
so the `+1.0` shift also has a larger effect, and it extrapolates for some rows. Under strain, the
influence-curve standard error can itself be too small. A larger sample, a smaller shift, or a
declared cap reduces the strain.

## An incremental intervention: change assignment odds

The third proposal is not one fixed assignment. The program multiplies each patient's current
conditional odds of an offer by two. This stochastic policy is the intervention. The fit reuses the
`method` object from the regime section, so only the estimand changes.

```python
from cleverly import IncrementalEffect
from cleverly.interventions import Incremental

incremental_truth = law.incremental_truth((1.0, 2.0))
print("population contrast:", incremental_truth["ate_ipsi[odds x2 vs natural course]"])

incremental_effect = study.identify(
    IncrementalEffect(
        (
            Incremental(1.0, name="current odds"),
            Incremental(2.0, name="double odds"),
        )
    )
)
print(incremental_effect.summary())
incremental_result = incremental_effect.estimate(method=method)
print(incremental_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
incremental_assessment = incremental_result.assess()
print("needs attention:", tuple(item.name for item in incremental_assessment.attention))
```

```python
for policy, report in incremental_assessment.report("support").items():
    print(policy, report.score_load)
```

This load describes the outcome targeting equation for each odds multiplier. The incremental fit
also targets the treatment mechanism. That second equation has no row-level load in this report.

The identification summary states the trade. The target needs no positivity assumption, because
its weights stay between one half and two for this multiplier. The mechanism defines the estimand,
so the estimate is not double robust. A good outcome regression cannot rescue an inconsistent
mechanism estimate. Kennedy (2019) derives both properties.

On this draw, `needs attention` names `nuisance_models`, because the boosted propensity is poorly
calibrated. A miscalibrated mechanism is the error this axis cannot absorb through the outcome
model. Compare the estimate with the printed population value, but one draw is not a coverage
result. Read the
[incremental section](../technical-reference/point-treatment-tmle.md#incremental-propensity-score-interventions)
before reporting one.

## Three tables that look alike

Each `to_frame()` table above has the same columns. The `estimand` column is what distinguishes
them: `ate_regime`, `ate_shift`, and `ate_ipsi`. Carry that name into the program report.

The library fits one intervention axis at a time. It has no joint targeting or joint influence-curve
covariance for two axes in one fit, so it refuses a combined request before fitting. Here the
intensity policy also uses a different treatment column. Fit each policy question separately.

## How far to trust this

```python
for axis, report in (
    ("known regime", regime_assessment),
    ("modified treatment policy", shift_assessment),
    ("incremental intervention", incremental_assessment),
):
    print(axis)
    print(report.summary())
    print()
```

The combined assessments keep the three policy questions separate. Read the support report for
each axis before comparing its estimates. An assessment records an omitted operation instead of
silently skipping it.

| layer | establishes | does not establish |
| --- | --- | --- |
| the combined assessments | which checks passed, warned, completed, or were omitted for each axis | that one axis answers another axis's policy question |
| the per-policy support reports | how concentrated each declared policy's weights are on these data | that a well-supported policy is worth adopting |
| the score-equation checks | each axis solved its own score equation | that the axis you chose matches the decision the office faces |

These fits do not validate the estimators. The
[validation grid](../technical-reference/method-evidence/validation-grid.md) has one parametric,
non-cross-fitted row per axis, and none covers the boosted learners or binned density used here.

Choose the axis that represents a change the program can implement. Then use its support report to
decide whether these data can estimate that policy well enough to report.

## Where to go next

Every axis on this page acts at one decision time. Read
[longitudinal TMLE](longitudinal-tmle.md) for a rule that assigns navigation again at day seven.
Read [MSM projections](msm-projections.md) for the working model that summarizes many plans as one
trend. The contrast these three axes replace is the average treatment effect, and
[point-treatment TMLE](point-treatment-tmle.md) estimates it.

The [examples index](index.md#the-program) lists every tutorial in the program.
