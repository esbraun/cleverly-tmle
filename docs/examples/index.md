# Examples

These tutorials follow one program. A regional health plan offers **care-transition navigation**
to adults who leave participating hospitals for home. Each tutorial answers one evaluation
question in that program with one method, and the table below lists them.

## The program

Start with [Point-treatment TMLE](point-treatment-tmle.md).

| tutorial | the evaluation question | method entry | the failure mode it demonstrates |
| --- | --- | --- | --- |
| [Point-treatment TMLE](point-treatment-tmle.md) | assignment to the standard navigation offer | [entry](../technical-reference/point-treatment-tmle.md) | one consistent nuisance gives a consistent point estimate. Two inconsistent nuisances have no general guarantee |
| [CV-TMLE](cross-fitting.md) | the same offer, with flexible learners and patients nested in navigator teams | [entry](../technical-reference/cv-tmle.md) | in-sample nuisance evaluation and ignored teams can understate uncertainty |
| [Collaborative TMLE](collaborative-tmle.md) | which approved baseline variables belong in the assignment model | [entry](../technical-reference/collaborative-tmle.md) | a queue lottery predicts assignment and confounds nothing |
| [DR-TMLE](dr-tmle.md) | a recorded assignment rule that is difficult to model | [entry](../technical-reference/dr-tmle/index.md) | solved score equations do not show that the nuisance or reduced regressions converge |
| [Intervention axes](interventions.md) | target by risk, raise assigned navigation intensity, or change assignment odds | [entry](../technical-reference/point-treatment-tmle.md#variations) | three policies define three estimands. The intensity policy uses its own continuous law |
| [Survey non-response](survey-nonresponse.md) | many patients do not return the 30-day transition survey | [entry](../technical-reference/point-treatment-tmle.md#missing-outcomes-and-controlled-direct-effects) | a complete-case fit estimates the effect standardized to the respondents' covariate distribution |
| [Longitudinal TMLE](longitudinal-tmle.md) | navigation at discharge and day seven | [entry](../technical-reference/longitudinal-tmle.md) | one regression cannot adjust for a time-varying confounder and preserve the causal path |
| [Time-to-event outcomes](longitudinal-survival.md) | plan exit, then readmission and death under repeated navigation | [entry](../technical-reference/longitudinal-tmle.md#survival-and-competing-risks) | coding death as censoring targets a controlled direct effect, not the reported total effect |
| [MSM projections](msm-projections.md) | three navigation cadences summarized as one trend | [entry](../technical-reference/msm-projections.md) | a linear working model that misses the arm means still defines the parameter it reports |

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
| treatment strategy | assignment to a standardized transition-navigation offer, not completed participation |
| variations | three cadences, an assigned navigation intensity, or two decisions, each on its own tutorial |
| standard offer | a bedside transition plan and two scheduled navigator contacts within 30 days |
| comparison | usual discharge support without access to the navigation protocol |
| primary outcome | a 30-day patient-reported transition score. The protocol scores death before day 30 as the worst transition score (composite strategy) |
| unreturned surveys | the composite score counts as observed. Only living patients who do not respond are missing |
| baseline adjustment | discharge risk, prior utilization, medication burden, age, and applicable site or calendar factors measured before assignment |
| interference control | reserved navigator capacity and access controls prevent one assignment from changing another patient's protocol |
| dependence | patients can share a navigator team. The [CV-TMLE tutorial](cross-fitting.md) keeps each team intact in fitting and inference |

Only baseline information enters a point-treatment adjustment set. In particular, actual length of
stay and completed contacts occur after assignment and cannot serve as baseline confounders.

The synthetic laws standardize their baseline covariates to mean 0 and SD 1. Outcome scores are in
synthetic units. The longitudinal tutorial uses a binary top-box outcome, and its protocol scores
death before day 30 as not top box.

The intervention is the **offer**. This choice fixes treatment versions despite different patient
uptake. The program records contamination, protocol changes, and capacity breaches because any of
them can weaken consistency or no interference.

## Two outcome families

| outcome | what it is | which pages use it |
| --- | --- | --- |
| transition experience | a 30-day patient-reported score, or its top-box indicator | every tutorial except time-to-event outcomes |
| time-to-event outcomes | plan exit, readmission, or death | [time-to-event outcomes](longitudinal-survival.md) |

The day-seven engagement score in [longitudinal TMLE](longitudinal-tmle.md) is a time-varying
confounder, not an outcome.

## Real data

The [TWINS analysis](twins-causal-inference.ipynb) sits outside the program and is the only
real-data example. Its outcome is first-year infant mortality. It compares several estimators on
one question, then adds a semi-synthetic longitudinal section with a known truth.

```{toctree}
:maxdepth: 1

point-treatment-tmle
cross-fitting
collaborative-tmle
dr-tmle
interventions
survey-nonresponse
longitudinal-tmle
longitudinal-survival
msm-projections
twins-causal-inference
```
