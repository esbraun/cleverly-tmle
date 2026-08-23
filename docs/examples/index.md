# Examples

These tutorials follow one program. A regional health plan contracts a network of hospitals, and its
patient-experience team runs tests of change on **nurse rounding practice**. Each page is one test
of change, answered by one method.

There is one tutorial for each method entry in the
[technical reference](../technical-reference/index.md), plus one for the intervention axes and one
for the survey mechanism. A reader who has decided which method fits their problem can open the
matching page and watch that method work end to end.

## Read this first

**None of these rollouts was randomized.** A stepped-wedge trial, in which wards adopt in a random
order, would answer every question on these pages better and with fewer assumptions. Ask for one
when you can still influence the design.

These tutorials are for the ordinary case, where the change spread because managers chose it. That
is exactly the situation confounding adjustment exists for, and it is also the situation in which
the adjustment can silently fail. Every page ends by saying which of its conclusions rest on the
data and which rest on an argument about the rollout.

## The program

| tutorial | the test of change | method entry | the failure mode it demonstrates |
| --- | --- | --- | --- |
| [Point-treatment TMLE](point-treatment-tmle.md) | wards adopt scripted hourly rounding | [entry](../technical-reference/point-treatment-tmle.md) | one wrong nuisance is survivable, and two are not |
| [CV-TMLE](cross-fitting.md) | the same change, with flexible learners and patients nested in wards | [entry](../technical-reference/cv-tmle.md) | an in-sample interval, and an undeclared ward, are each too narrow |
| [Collaborative TMLE](collaborative-tmle.md) | which ward characteristics belong in the adoption model | [entry](../technical-reference/collaborative-tmle.md) | a training-cohort slot predicts adoption and confounds nothing |
| [DR-TMLE](dr-tmle.md) | wards adopted on manager judgement nobody recorded | [entry](../technical-reference/dr-tmle.md) | double robustness covers consistency, and ordinary inference is only singly robust |
| [Intervention axes](interventions.md) | screen before rounding, add rounds per shift, or nudge uptake | [entry](../technical-reference/point-treatment-tmle.md#variations) | three policies produce three estimands whose result tables look alike |
| [Survey non-response](survey-nonresponse.md) | most patients never returned the HCAHPS survey | [entry](../technical-reference/point-treatment-tmle.md#missing-outcomes-and-controlled-direct-effects) | a complete-case analysis answers a question about respondents |
| [Longitudinal TMLE](longitudinal-tmle.md) | rounding at two admissions, then members who leave the plan | [entry](../technical-reference/longitudinal-tmle.md) | one regression cannot both adjust for a time-varying confounder and leave the causal path open |
| [MSM projections](msm-projections.md) | three rounding cadences read as one trend | [entry](../technical-reference/msm-projections.md) | a working model that does not fit still defines the parameter it reports |

Every page follows the same path. It states an applied question, says why the method suits it,
builds a synthetic population with a known truth, declares a design and an estimand, fits, and then
demonstrates the failure mode the method exists to handle. Each one closes by separating what its
diagnostics established from what they cannot establish.

## Three outcomes, and where each one lives

| outcome | what it is | which pages use it |
| --- | --- | --- |
| HCAHPS | a post-discharge survey, counted if it returns inside a 49-day window | the experience score on most pages, and the whole subject of [survey non-response](survey-nonresponse.md) |
| service recovery | issues a patient raises, and whether they are still open | the time-varying confounder in [longitudinal TMLE](longitudinal-tmle.md) |
| health plan churn | a member leaving the plan | censoring, then a retention curve, then two competing causes, all in [longitudinal TMLE](longitudinal-tmle.md) |

## Real data

The [TWINS analysis](twins-causal-inference.ipynb) sits outside the program. It is the only example
on real data, it compares several estimators on one question, and it stores its executed outputs.

## What the code gate does and does not check

Documentation code is syntax-checked and executed. The corresponding numerical behavior is enforced
by the ordinary fast and named slow tests listed in the
[evidence manifest](../technical-reference/evidence.md). An example that runs is a much weaker claim
than an example that is right, and only the weaker one is made by the documentation gate.

```{toctree}
:maxdepth: 1

point-treatment-tmle
cross-fitting
collaborative-tmle
dr-tmle
interventions
survey-nonresponse
longitudinal-tmle
msm-projections
twins-causal-inference
```
