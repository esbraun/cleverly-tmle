# Examples

These tutorials follow one program. A regional health plan offers **care-transition navigation**
to adults who leave participating hospitals for home. Each page studies one decision in that
program with one method.

There is one tutorial for each method entry in the
[technical reference](../technical-reference/index.md). Two more tutorials cover intervention
policies and survey non-response.

## Read this first

The tutorials use observational assignment. A randomized offer would identify the assignment
effect with fewer assumptions. Use one when the program can still control assignment.

The synthetic laws make the causal assumptions true by construction. A real analysis must defend
them from its protocol and operational records. The relevant assumptions are consistency,
exchangeability, positivity, and no interference. The
[Oxford causal assumptions chapter](https://www.stats.ox.ac.uk/~evans/APTS/causassmp.html) defines
each assumption and explains why consistency requires a well-defined intervention.

## The shared study design

| design element | program definition |
| --- | --- |
| target population | adults discharged home from a participating hospital during the declared enrollment period |
| time zero | after eligibility and baseline measurement, but before navigation assignment |
| binary treatment | assignment to a standardized transition-navigation offer, not completed participation |
| standard offer | a bedside transition plan and two scheduled navigator contacts within 30 days |
| comparison | usual discharge support without access to the navigation protocol |
| primary outcome | a 30-day patient-reported transition score, defined for every eligible discharge |
| baseline adjustment | discharge risk, prior utilization, medication burden, age, and applicable site or calendar factors measured before assignment |
| interference control | reserved navigator capacity and access controls prevent one assignment from changing another patient's protocol |
| dependence | patients can share a navigator team. The cluster tutorial keeps each team intact in fitting and inference |

Only baseline information enters a point-treatment adjustment set. In particular, actual length of
stay and completed contacts occur after assignment and cannot serve as baseline confounders.

The intervention is the **offer**. This choice fixes treatment versions despite different patient
uptake. The program records contamination, protocol changes, and capacity breaches because any of
them can weaken consistency or no interference.

## The program

| tutorial | the test of change | method entry | the failure mode it demonstrates |
| --- | --- | --- | --- |
| [Point-treatment TMLE](point-treatment-tmle.md) | assignment to the standard navigation offer | [entry](../technical-reference/point-treatment-tmle.md) | one wrong nuisance is survivable, and two are not |
| [CV-TMLE](cross-fitting.md) | the same offer, with flexible learners and patients nested in navigator teams | [entry](../technical-reference/cv-tmle.md) | an in-sample interval and an undeclared team are each too narrow |
| [Collaborative TMLE](collaborative-tmle.md) | which approved baseline variables belong in the assignment model | [entry](../technical-reference/collaborative-tmle.md) | a queue lottery predicts assignment and confounds nothing |
| [DR-TMLE](dr-tmle.md) | a recorded assignment rule that is difficult to model | [entry](../technical-reference/dr-tmle.md) | nuisance misspecification is not unmeasured confounding |
| [Intervention axes](interventions.md) | target by risk, add navigation hours, or change assignment odds | [entry](../technical-reference/point-treatment-tmle.md#variations) | three policies define three estimands whose result tables look alike |
| [Survey non-response](survey-nonresponse.md) | many patients do not return the 30-day transition survey | [entry](../technical-reference/point-treatment-tmle.md#missing-outcomes-and-controlled-direct-effects) | respondent selection does not define a common causal population |
| [Longitudinal TMLE](longitudinal-tmle.md) | navigation at discharge and day seven, followed by plan exits | [entry](../technical-reference/longitudinal-tmle.md) | one regression cannot adjust for a time-varying confounder and preserve the causal path |
| [MSM projections](msm-projections.md) | three navigation cadences summarized as one trend | [entry](../technical-reference/msm-projections.md) | a working model that does not fit still defines the parameter it reports |

Every page states an applied question, builds a synthetic population, declares the design and
estimand, fits the method, and demonstrates its failure mode. Each page also separates diagnostics
from causal assumptions that data cannot verify.

## Three outcome families

| outcome | what it is | which pages use it |
| --- | --- | --- |
| transition experience | a 30-day patient-reported score | most point-treatment pages and the survey non-response tutorial |
| unresolved transition issues | medication, appointment, or equipment problems that remain open | the time-varying confounder in longitudinal TMLE |
| health plan exit | voluntary plan switching or administrative eligibility loss | the survival and competing-risk sections of longitudinal TMLE |

## Real data

The [TWINS analysis](twins-causal-inference.ipynb) sits outside the program. It is the only example
on real data. It compares several estimators on one question and stores its executed outputs.

## What the code gate checks

Documentation code is syntax-checked and executed. Numerical behavior is enforced by the ordinary
fast and named slow tests in the [evidence manifest](../technical-reference/evidence.md). A runnable
example is not evidence that a real study satisfies its causal assumptions.

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
