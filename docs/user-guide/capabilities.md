# What cleverly can estimate

`cleverly` estimates causal effects with targeted maximum likelihood estimation (TMLE) and
related targeted estimators. It covers one treatment decision, treatments that repeat over time,
outcomes that are a time to an event, and a randomized trial pooled with outside rows. Find your
question in the tables below, then read the guide page in the last column. The
[implementation matrix](../technical-reference/index.md) in the technical reference is the full
list, with the theory and the evidence behind each row.

Each row names the kind of data you describe to `cleverly` (the design, such as `PointTreatment`)
and the quantity you ask for (the estimand, such as `ATE`). Whether a fit reports an interval, and
whether its result reads `supported` or `experimental`, depends on the exact setup: the guide page
and the linked technical section say which.

## One treatment decision

These rows take a `PointTreatment` design: one treatment, chosen once, and an outcome measured
afterwards. The trend and learned-rule rows also name their forms for treatments over time.

| question you ask | what to ask for | where to read |
| --- | --- | --- |
| What would the average outcome be if everyone got each option, and how do the options compare? | `CounterfactualMean`, `ATE`, `RiskRatio`, `OddsRatio`, and contrasts between any two of three or more options | [estimands](estimands.md), "Arm-indexed questions"; [theory](../technical-reference/point-treatment-tmle.md#counterfactual-means-and-arm-contrasts) |
| What was the effect among the people who got the treatment, or among those who did not? | `ATT`, `ATC` | [estimands](estimands.md), "Arm-indexed questions"; [theory](../technical-reference/point-treatment-tmle.md#conditional-population-effects) |
| The chance of treatment was set by design, as in a randomized trial. Can the analysis use it? | declare `treatment_probabilities=` on the design | [describe the data](data-design.md), "Point treatment"; [theory](../technical-reference/point-treatment-tmle.md#known-treatment-mechanism) |
| How much of the outcome would go away if no one were exposed? | `NaturalCourseMean`, `PopulationAttributableRisk`, `PopulationAttributableFraction` | [estimands](estimands.md), "Arm-indexed questions"; [theory](../technical-reference/point-treatment-tmle.md#population-interventions) |
| Some outcomes are missing, or a variable after treatment should be held fixed | `missingness=` on the design; `ControlledDirectEffect` with `intermediate=` | [describe the data](data-design.md), "Missing outcomes and missing treatments"; [estimands](estimands.md), "Controlled direct effects"; [theory](../technical-reference/point-treatment-tmle.md#missing-outcomes-and-controlled-direct-effects) |
| What if everyone followed a rule, such as "treat when the score is above 5"? | `RegimeMean`, `RegimeContrast` with a `Rule` or `Static` plan | [estimands](estimands.md), "Known regimes"; [theory](../technical-reference/point-treatment-tmle.md#known-regimes) |
| What if treatment were assigned at random with chances I choose? | `RegimeMean`, `RegimeContrast` with a `Stochastic` plan | [estimands](estimands.md), "Known regimes"; [theory](../technical-reference/point-treatment-tmle.md#known-regimes) |
| How does the average outcome trend across options, or across plans over time, summarized by a line or curve? | `MSMProjection` | [estimands](estimands.md), "Marginal structural models"; [analyses over time](longitudinal.md), "Longitudinal MSM projections"; [theory](../technical-reference/msm-projections.md) |
| How well would a treatment rule learned from these data work, and how does it compare with a fixed plan? | `LearnedRuleValue`, `OptimalRuleValue`, `LearnedRuleContrast`, `OptimalRuleContrast`; for a rule learned over two treatment times, `LTMLE(regimens=LearnedRegimen())` from `cleverly.longitudinal`, called directly | [estimands](estimands.md), "Learned treatment rules"; [analyses over time](longitudinal.md), "Declare a dynamic rule"; [theory](../technical-reference/point-treatment-tmle.md#learned-rules) |

## Changing a dose or the odds of treatment

| question you ask | what to ask for | where to read |
| --- | --- | --- |
| What if everyone's dose were raised by a fixed amount, or capped? | `ModifiedTreatmentPolicy`, `ModifiedTreatmentPolicyEffect` | [estimands](estimands.md), "Modified treatment policies"; [theory](../technical-reference/point-treatment-tmle.md#modified-treatment-policies) |
| The treatment has several continuous parts, such as a price and a discount | a modified treatment policy on a vector treatment | [estimands](estimands.md), "Modified treatment policies"; [theory](../technical-reference/point-treatment-tmle.md#vector-treatments) |
| The dose change depends on declared limits, or on inputs learned or tuned from the data | a modified treatment policy at declared boundaries | [estimands](estimands.md), "Modified treatment policies"; [theory](../technical-reference/longitudinal-tmle.md#modified-treatment-policies-at-a-node) |
| What if everyone's odds of treatment were doubled? | `IncrementalMean`, `IncrementalEffect` | [estimands](estimands.md), "Incremental propensity-score interventions"; [theory](../technical-reference/point-treatment-tmle.md#incremental-propensity-score-interventions) |
| The same, for a treatment with three or more levels | `Incremental` with one odds factor per level | [estimands](estimands.md), "Incremental propensity-score interventions"; [theory](../technical-reference/point-treatment-tmle.md#incremental-propensity-score-interventions) |
| How does the average outcome change across a range of doses? | `DoseResponseTMLE` from `cleverly.estimators`, called directly; the study interface does not offer it yet | [estimands](estimands.md), "A curve over a continuous dose"; [theory](../technical-reference/dose-response.md) |

## Who benefits

| question you ask | what to ask for | where to read |
| --- | --- | --- |
| How does the effect change with a baseline column? | `ConditionalContrast` on a declared basis, with a curve, a band and a test that the effect varies | [conditional effects](conditional-effects.md); [theory](../technical-reference/conditional-effects.md) |
| The same, without choosing a basis | `ConditionalContrast` with `EPMethod`, which reports no interval | [conditional effects](conditional-effects.md), "Fit the effect without choosing a basis"; [theory](../technical-reference/conditional-effects.md) |
| Does a priority score put the people who benefit most first? | `effect.targeting_curve` | [conditional effects](conditional-effects.md), "Check how well a score ranks who to treat first"; [theory](../technical-reference/conditional-effects.md#targeting-curves) |
| Does a score built elsewhere predict who benefits, and is one score better than another? | `effect.assess_score`, with `compare=` for two scores | [conditional effects](conditional-effects.md), "Check a score fitted elsewhere"; [theory](../technical-reference/conditional-effects.md#external-scores) |

## Treatments over time

These rows take a `LongitudinalTreatment` or `LongPanel` design: a treatment decided at several
times, with covariates measured between them.

| question you ask | what to ask for | where to read |
| --- | --- | --- |
| What would the end-of-study outcome be under each treatment plan? | `RegimeMean`, `RegimeContrast`; the default method | [analyses over time](longitudinal.md), "Declare temporal roles"; [theory](../technical-reference/longitudinal-tmle.md#end-of-study-regimen-means) |
| The same, with an estimator that stays on target when the models at each step are only partly right | `SDRMethod` | [analyses over time](longitudinal.md), "Fit the sequentially doubly robust estimator"; [theory](../technical-reference/longitudinal-tmle.md#the-sequentially-doubly-robust-estimator) |
| What do the simpler reference estimators say? | `GComputationMethod`, `IPWMethod` | [analyses over time](longitudinal.md), "Report the sequential baselines"; [theory](../technical-reference/sequential-baselines.md) |
| What is the effect of each treatment time, and does it vary with a covariate? | `StructuralNestedMean` from `cleverly.longitudinal`, called directly; the study interface does not offer it yet | [analyses over time](longitudinal.md), "Estimate structural blips"; [theory](../technical-reference/structural-dynamic-effects.md) |
| What does a fully modelled forward simulation say, or a plan that fixes a continuous dose? | `ParametricGFormulaMethod` | [analyses over time](longitudinal.md), "Simulate the parametric g-formula"; [theory](../technical-reference/parametric-gformula.md) |
| What if the odds of treatment were changed at every time? | `IncrementalMean`, `IncrementalEffect` with `IPSIMethod` | [analyses over time](longitudinal.md), "Estimate an incremental intervention over time"; [theory](../technical-reference/longitudinal-tmle.md#incremental-interventions-over-time) |
| The outcome is measured at every time, not only at the end | a design with repeated outcomes; the default method | [analyses over time](longitudinal.md), "Repeated outcomes"; [theory](../technical-reference/longitudinal-tmle.md#repeated-outcomes) |
| The same, with the sequentially doubly robust estimator | repeated outcomes with `SDRMethod` | [analyses over time](longitudinal.md), "Repeated outcomes"; [theory](../technical-reference/longitudinal-tmle.md#sdr-on-a-repeated-outcome) |
| The same, under a change in the odds of treatment | repeated outcomes with `IncrementalMean` and `IPSIMethod` | [analyses over time](longitudinal.md), "Repeated outcomes"; [theory](../technical-reference/longitudinal-tmle.md#ipsi-on-a-repeated-outcome) |

## Time to an event

These rows take a `TimeToEvent` design, or a longitudinal design with an event outcome. A grid
of check times, such as days 30, 60 and 90, is the usual form; `grid="continuous"` uses the exact
event times.

| question you ask | what to ask for | where to read |
| --- | --- | --- |
| What is the risk of the event by each check time under each treatment? | `RegimeContrast` with `horizons=` on a `TimeToEvent` design | [time-to-event outcomes](survival.md), "One treatment at the start and a time to an event"; [theory](../technical-reference/longitudinal-tmle.md#one-baseline-treatment-held-over-the-nodes) |
| The same, when treatment changes over follow-up, or when another event can come first | a longitudinal design with an event outcome; for competing events, `causes=` on `TimeToEvent` or a mapping of causes in `outcome=` on a longitudinal design | [time-to-event outcomes](survival.md), "Treatments over time with an event outcome" and "When another event can come first"; [theory](../technical-reference/longitudinal-tmle.md#survival-and-competing-risks) |
| What would the risk be had the competing event been prevented, or through which event does the treatment act? | `eliminate=` on the design; a `Separable` plan | [time-to-event outcomes](survival.md), "Ask what the risk would be had the other event been prevented" and "Separate a treatment's effect on two events"; [theory](../technical-reference/longitudinal-tmle.md#survival-and-competing-risks) |
| Can one correction step fit the whole risk curve at once, through the event rate at each time? | `survival_model="hazard"` with `targeting="one_step"` | [time-to-event outcomes](survival.md), "Keep the risk curve from dipping"; [theory](../technical-reference/longitudinal-tmle.md#hazard-based-survival-targeting) |
| Can the data choose the treatment and censoring models? | `CollaborativeTMLEMethod` on a `TimeToEvent` design, which reports point estimates only | [time-to-event outcomes](survival.md), "Let the data choose the treatment and censoring models"; [theory](../technical-reference/collaborative-tmle.md#time-to-event-outcomes) |
| Can the risk interval stay valid when one set of models is wrong? | `DRTMLEMethod` on a `TimeToEvent` design | [time-to-event outcomes](survival.md), "Keep the interval when one set of models is wrong"; [theory](../technical-reference/dr-tmle/theorem.md#time-to-event-outcomes) |
| Can I use the exact event times instead of a grid? | `TimeToEvent(grid="continuous")`, which takes the default method only | [time-to-event outcomes](survival.md), "Exact event times, without a grid"; [theory](../technical-reference/continuous-time-survival.md) |

## A trial with outside data

| question you ask | what to ask for | where to read |
| --- | --- | --- |
| A small randomized trial has outside rows from a registry or an earlier study. Can they sharpen the trial's effect without biasing it? | `TrialWithExternalData`, `TrialATE` and `AdaptiveTMLEMethod`, the only method on this design | [a small trial with external data](trial-with-external-data.md); [theory](../technical-reference/adaptive-tmle.md) |

## Ranking candidate exposures

| question you ask | what to ask for | where to read |
| --- | --- | --- |
| Which of several candidate exposures moves the outcome most, with p-values adjusted for testing many? | `variable_importance`; `variable_importance_plan` counts its fits first | [Python API](../api/results-assessment.md); [method](../technical-reference/validation-methods.md#variable-importance) |

## Choosing how to fit

The rows above fix the question. These choose the estimator, its settings, and the checks after
the fit.

| question you ask | what to ask for | where to read |
| --- | --- | --- |
| Can the data choose which covariates the treatment model uses? | `CollaborativeTMLEMethod` | [methods and learners](methods-learners.md), "Method presets"; [theory](../technical-reference/collaborative-tmle.md) |
| Can the interval stay valid when one of the two models is wrong? | `DRTMLEMethod` | [methods and learners](methods-learners.md), "Method presets"; [theory](../technical-reference/dr-tmle/index.md) |
| Can I weight by the treatment model alone, adapted to the outcome? | `AdaptiveIPTWMethod` | [methods and learners](methods-learners.md), "Method presets"; [theory](../technical-reference/adaptive-iptw.md) |
| How are the models trained out of fold, and when are the folds reused? | `CrossFitting` | [methods and learners](methods-learners.md), "Two fold layers"; [theory](../technical-reference/cv-tmle.md) |
| How does the correction step stop, and which other fitting options exist? | `Targeting`, `ModelSpec` | [methods and learners](methods-learners.md), "Targeting and bounds"; [theory](../technical-reference/point-treatment-tmle.md#stopping-rules-step-bound-and-best-iterate) |
| How are intervals, bands, diagnostics and sensitivity analyses computed? | `Inference`; `result.diagnostics`, `result.sensitivity` | [results and assessment](results-assessment.md); [theory](../technical-reference/inference.md) |
| How far does the evidence for analyses over time reach, in number of treatment times? | the default method, `SDRMethod` | [analyses over time](longitudinal.md); [theory](../technical-reference/longitudinal-tmle.md) |

## Why a request can be refused

An error at identification or method selection is often a scientific boundary, not merely an API
limitation. Every `CleverlyError` carries `error.refusal`, and `error.refusal.reason` holds one of
five codes.

| `error.refusal.reason` | meaning |
| --- | --- |
| `invalid_declaration` | the requested analysis cannot run on the declared sample or design |
| `different_estimand` | the requested option changes the causal question |
| `implementation_gap` | the method is well posed, and this package has no implementation yet |
| `inference_gap` | the estimate exists, and the requested uncertainty calculation is not derived |
| `api_redirection` | the feature exists through another argument or object |

`effect.available_methods()` carries the same record for each method before any fit. Read
[how to read a refusal](../technical-reference/scope-and-refusals.md#how-to-read-a-refusal) for the
action that each code asks of you.
Do not work around a refusal by switching to a similarly named estimand unless that estimand is
genuinely the scientific question.

## What is not implemented

Separable direct and indirect effects on an event and controlled direct effects ship, in the rows
above. Natural and interventional mediation effects do not; roadmap item
[X5](../roadmap.md) tracks them. Learning the weighting function of an estimate directly (the
Riesz representer), in place of a treatment model, is roadmap items R1 and R2.
`effect.available_methods()` lists a `riesz_tmle` method as unavailable on every design; where
direct Riesz learning is the missing piece, its reason names R1. Effects identified from a causal
graph beyond an adjustment set, front-door adjustment, instrumental-variable effects, and carrying
an effect over to another population have no roadmap item and are outside the current scope.

The EP learner (`EPMethod`) estimates a `ConditionalContrast` and reports no interval. The same
learner can also set the rule of a learned treatment rule, through `LearnedRule(loss="ep")`, and
the value of that rule keeps its interval. EP learning of any other target is not implemented
([conditional effects](../technical-reference/conditional-effects.md)). The
[roadmap](../roadmap.md) states the evidence required before a proposed method can become a
release claim.

## Extending the package

The public protocols are `Estimand`, `IdentificationProvider`, `EstimationMethod`, and
`CausalResult`.

- A custom provider returns an `IdentifiedEffect` only when it can name the observed-data
  functional and its assumptions.
- A custom method declares which functionals it supports and returns a result with stable
  parameter keys.
- A new registered analytic target needs the oracle, Gateaux, remainder, and mutation evidence
  that the [evidence manifest](../technical-reference/evidence.md) describes. A typed wrapper
  certifies nothing.
