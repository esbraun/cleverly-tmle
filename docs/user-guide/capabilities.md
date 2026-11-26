# What cleverly can estimate

`cleverly` estimates causal effects with targeted maximum likelihood estimation (TMLE) and related
estimators. The main ones fit a model of the outcome and a model of who gets the treatment, then
runs a correction step that uses the treatment model to remove the bias the outcome model leaves.
For the most common question, whether a yes/no program changed an outcome on average, ask for
`ATE`. The [full list of methods](../technical-reference/index.md) gives the theory and evidence
behind each row. Most rows run through `CausalStudy`, the main entry point; a few call an estimator
directly.

Each row names the design (the data you describe, such as `PointTreatment`) and the estimand (the
quantity you ask for, such as `ATE`). Most answers are causal only if the data record every common
cause of the treatment and the outcome (and of dropout), and every kind of person had a real chance
of each option compared. Some questions, such as changing the odds of treatment, relax the second
condition; each row's guide page states what its answer needs. `cleverly` marks a result
experimental when no validation study covers its setup, and some setups report no interval; each
row's links say which.

## One treatment decision

These rows take a `PointTreatment` design: one treatment, chosen once, and an outcome measured
afterwards. The trend and learned-rule rows also apply to a treatment that repeats.

| question you ask | what to ask for | where to read |
| --- | --- | --- |
| What would the average outcome be if everyone got each option (each arm), and how do the options compare? | `CounterfactualMean`, `ATE`, `RiskRatio`, `OddsRatio`, and multi-arm contrasts | [estimands](estimands.md), "Arm-indexed questions"; [theory](../technical-reference/point-treatment-tmle.md#counterfactual-means-and-arm-contrasts) |
| What was the effect among the people who got the treatment, or among those who did not? | `ATT`, `ATC` | [estimands](estimands.md), "Arm-indexed questions"; [theory](../technical-reference/point-treatment-tmle.md#conditional-population-effects) |
| The chance of treatment was set by design, as in a randomized trial. Can the analysis use it? | declare `treatment_probabilities=` on the design | [describe the data](data-design.md), "Point treatment"; [theory](../technical-reference/point-treatment-tmle.md#known-treatment-mechanism) |
| How much of the outcome would go away if no one were exposed? | `NaturalCourseMean`, `PopulationAttributableRisk`, `PopulationAttributableFraction` | [estimands](estimands.md), "Arm-indexed questions"; [theory](../technical-reference/point-treatment-tmle.md#population-interventions) |
| Some outcomes are missing, or you want the effect with a later variable held at one value (a controlled direct effect) | `missingness=` on the design; `ControlledDirectEffect` with `intermediate=` | [describe the data](data-design.md), "Missing outcomes and missing treatments"; [estimands](estimands.md), "Controlled direct effects"; [theory](../technical-reference/point-treatment-tmle.md#missing-outcomes-and-controlled-direct-effects) |
| What if everyone followed a rule, such as "treat when the score is above 5"? | `RegimeMean`, `RegimeContrast` with a plan: `Static` (treat everyone, or no one) or `Rule` | [estimands](estimands.md), "Known regimes"; [theory](../technical-reference/point-treatment-tmle.md#known-regimes) |
| What if treatment were assigned at random with chances I choose? | `RegimeMean`, `RegimeContrast` with a `Stochastic` plan | [estimands](estimands.md), "Known regimes"; [theory](../technical-reference/point-treatment-tmle.md#known-regimes) |
| How does the average outcome trend across options, or across plans over time, summarized by a line or curve? | `MSMProjection`, a marginal structural model (a line or curve across the options) | [estimands](estimands.md), "Marginal structural models"; [analyses over time](longitudinal.md), "Longitudinal MSM projections"; [theory](../technical-reference/msm-projections.md) |
| How well would a treatment rule learned from these data work, and how does it compare with a fixed plan? | `LearnedRuleValue`, `OptimalRuleValue` (the best possible rule), `LearnedRuleContrast`, `OptimalRuleContrast`; for a rule learned over two treatment times, `LTMLE(regimens=LearnedRegimen())`, called directly | [estimands](estimands.md), "Learned treatment rules"; [analyses over time](longitudinal.md), "Declare a dynamic rule"; [theory](../technical-reference/point-treatment-tmle.md#learned-rules) |

## Changing a dose or the odds of treatment

| question you ask | what to ask for | where to read |
| --- | --- | --- |
| What if everyone's dose were raised by a fixed amount, or capped? | `ModifiedTreatmentPolicy`, `ModifiedTreatmentPolicyEffect` | [estimands](estimands.md), "Modified treatment policies"; [theory](../technical-reference/point-treatment-tmle.md#modified-treatment-policies) |
| The treatment has several continuous parts, such as a price and a discount | `ModifiedTreatmentPolicy` on a treatment with several columns | [estimands](estimands.md), "Modified treatment policies"; [theory](../technical-reference/point-treatment-tmle.md#vector-treatments) |
| The changed dose must stay inside limits you declare, or uses a cap learned from the data | `ModifiedTreatmentPolicy` with `treatment_support=` on the design, or a cap tuned on the data | [estimands](estimands.md), "Modified treatment policies"; [theory](../technical-reference/longitudinal-tmle.md#modified-treatment-policies-at-a-node) |
| What if everyone's odds of treatment were doubled? | `IncrementalMean`, `IncrementalEffect` | [estimands](estimands.md), "Incremental propensity-score interventions"; [theory](../technical-reference/point-treatment-tmle.md#incremental-propensity-score-interventions) |
| The same, for a treatment with three or more levels | `Incremental` with one odds factor per level | [estimands](estimands.md), "Incremental propensity-score interventions"; [theory](../technical-reference/point-treatment-tmle.md#incremental-propensity-score-interventions) |
| How does the average outcome change across a range of doses? | `DoseResponseCurve`, with intervals and one band over the range | [estimands](estimands.md), "A curve over a continuous dose"; [theory](../technical-reference/dose-response.md) |

## Who benefits

| question you ask | what to ask for | where to read |
| --- | --- | --- |
| How does the effect change with a baseline column? | `ConditionalContrast` on columns you choose (the basis), with a curve, a band over the whole curve, and a test that the effect varies | [conditional effects](conditional-effects.md); [theory](../technical-reference/conditional-effects.md) |
| The same, letting the data choose the shape of the curve | `ConditionalContrast` with the efficient plug-in (EP) learner `EPMethod`, which reports no interval | [conditional effects](conditional-effects.md), "Fit the effect without choosing a basis"; [theory](../technical-reference/conditional-effects.md) |
| Does a priority score put the people who benefit most first? | `effect.targeting_curve`, the gain over random choice | [conditional effects](conditional-effects.md), "Check how well a score ranks who to treat first"; [theory](../technical-reference/conditional-effects.md#targeting-curves) |
| Does a score built elsewhere predict who benefits, and is one score better than another? | `effect.assess_score`, with `compare=` for two scores | [conditional effects](conditional-effects.md), "Check a score fitted elsewhere"; [theory](../technical-reference/conditional-effects.md#external-scores) |

## Treatments over time

These rows take a `LongitudinalTreatment` design (one row per person) or a `LongPanel` design (one
row per person and time): a treatment decided at several
times, with covariates measured between them.

| question you ask | what to ask for | where to read |
| --- | --- | --- |
| What would the end-of-study outcome be under each treatment plan? | `RegimeMean`, `RegimeContrast`; the default method, longitudinal TMLE | [analyses over time](longitudinal.md), "Declare temporal roles"; [theory](../technical-reference/longitudinal-tmle.md#end-of-study-regimen-means) |
| The same, with an estimator that stays on target when, at each time, either the outcome model or the treatment model is right | `SDRMethod` | [analyses over time](longitudinal.md), "Fit the sequentially doubly robust estimator"; [theory](../technical-reference/longitudinal-tmle.md#the-sequentially-doubly-robust-estimator) |
| What do simpler estimators say, beside the targeted one? | `GComputationMethod` (outcome models alone), `IPWMethod` (weights alone) | [analyses over time](longitudinal.md), "Report the sequential baselines"; [theory](../technical-reference/sequential-baselines.md) |
| What is the effect of each treatment time, and does it vary with a covariate? | `StructuralNestedMean`, a linear model of each time's effect, called directly | [analyses over time](longitudinal.md), "Estimate structural blips"; [theory](../technical-reference/structural-dynamic-effects.md) |
| Can I simulate forward from models of every covariate, or fix a continuous dose at each time? | `ParametricGFormulaMethod` | [analyses over time](longitudinal.md), "Simulate the parametric g-formula"; [theory](../technical-reference/parametric-gformula.md) |
| What if the odds of treatment were changed at every time? | `IncrementalMean`, `IncrementalEffect` with `IPSIMethod` | [analyses over time](longitudinal.md), "Estimate an incremental intervention over time"; [theory](../technical-reference/longitudinal-tmle.md#incremental-interventions-over-time) |
| The outcome is measured at every time, not only at the end | a design with repeated outcomes; the default method | [analyses over time](longitudinal.md), "Repeated outcomes"; [theory](../technical-reference/longitudinal-tmle.md#repeated-outcomes) |
| The same, with the sequentially doubly robust estimator | repeated outcomes with `SDRMethod` | [analyses over time](longitudinal.md), "Repeated outcomes"; [theory](../technical-reference/longitudinal-tmle.md#sdr-on-a-repeated-outcome) |
| The same, under a change in the odds of treatment | repeated outcomes with `IncrementalMean` and `IPSIMethod` | [analyses over time](longitudinal.md), "Repeated outcomes"; [theory](../technical-reference/longitudinal-tmle.md#ipsi-on-a-repeated-outcome) |

## Time to an event

These rows take a `TimeToEvent` design, or a longitudinal design with an event outcome. Risks are
read at check times, such as days 30, 60 and 90, or at the exact event times. A person is
censored when follow-up ends before the event; a competing event, such as death before
readmission, prevents it.

| question you ask | what to ask for | where to read |
| --- | --- | --- |
| What is the risk of the event by each check time under each treatment? | `RegimeContrast` of two `Static` plans with `horizons=` on a `TimeToEvent` design, giving risk differences | [time-to-event outcomes](survival.md), "One treatment at the start and a time to an event"; [theory](../technical-reference/longitudinal-tmle.md#one-baseline-treatment-held-over-the-nodes) |
| The same, when treatment changes over follow-up, or when another event can come first | a longitudinal design with an event outcome; for competing events, `causes=` on `TimeToEvent` or a mapping of causes in `outcome=` on a longitudinal design | [time-to-event outcomes](survival.md), "Treatments over time with an event outcome" and "When another event can come first"; [theory](../technical-reference/longitudinal-tmle.md#survival-and-competing-risks) |
| What would the risk be had the competing event been prevented, or how much of the effect runs through the competing event? | `eliminate=` on the design; a `Separable` plan, which splits the treatment's effect on the two events | [time-to-event outcomes](survival.md), "Ask what the risk would be had the other event been prevented" and "Separate a treatment's effect on two events"; [theory](../technical-reference/longitudinal-tmle.md#survival-and-competing-risks) |
| Can the risk curve never dip, with one correction step for every check time at once? | `survival_model="hazard"` with `targeting="one_step"` | [time-to-event outcomes](survival.md), "Keep the risk curve from dipping"; [theory](../technical-reference/longitudinal-tmle.md#hazard-based-survival-targeting) |
| Can the data choose the treatment and censoring models? | `CollaborativeTMLEMethod` on a `TimeToEvent` design, which reports point estimates only | [time-to-event outcomes](survival.md), "Let the data choose the treatment and censoring models"; [theory](../technical-reference/collaborative-tmle.md#time-to-event-outcomes) |
| Can the risk interval stay valid when the event model, or the treatment and censoring models, are wrong? | `DRTMLEMethod` on a `TimeToEvent` design | [time-to-event outcomes](survival.md), "Keep the interval when one set of models is wrong"; [theory](../technical-reference/dr-tmle/theorem.md#time-to-event-outcomes) |
| Can I use the exact event times instead of a grid? | `TimeToEvent(grid="continuous")`, which takes the default method only | [time-to-event outcomes](survival.md), "Exact event times, without a grid"; [theory](../technical-reference/continuous-time-survival.md) |

## A trial with outside data

| question you ask | what to ask for | where to read |
| --- | --- | --- |
| Can outside rows, such as a registry, sharpen a small trial's estimate, correcting for how they differ? | `TrialWithExternalData`, `TrialATE` and `AdaptiveTMLEMethod`, the only method on this design | [a small trial with external data](trial-with-external-data.md); [theory](../technical-reference/adaptive-tmle.md) |

## Ranking candidate exposures

| question you ask | what to ask for | where to read |
| --- | --- | --- |
| Which of several candidate exposures has an effect on the outcome you can detect, with p-values adjusted for testing many? | `variable_importance`; `variable_importance_plan` counts the model fits first | [rank candidate exposures](variable-importance.md); [method](../technical-reference/validation-methods.md#variable-importance) |

## Plotting a result

| question you ask | what to ask for | where to read |
| --- | --- | --- |
| Which figure shows the size of an effect, the overlap of the groups, the weights or the sensitivity analyses? | `cleverly.plotting`: `forest`, `overlap`, `weights`, `sensitivity` and ten more | [which plot answers which question](plots.md); [plotting reference](../api/plotting.md) |

## Choosing how to fit

| question you ask | what to ask for | where to read |
| --- | --- | --- |
| Can the data choose which covariates the treatment model uses? | `CollaborativeTMLEMethod` | [methods and learners](methods-learners.md), "Method presets"; [theory](../technical-reference/collaborative-tmle.md) |
| The default interval needs both models close to right. Can it hold when one is wrong? | `DRTMLEMethod` | [methods and learners](methods-learners.md), "Method presets"; [theory](../technical-reference/dr-tmle/index.md) |
| Can I estimate with treatment weights alone (IPTW), with no outcome model? | `AdaptiveIPTWMethod` | [methods and learners](methods-learners.md), "Method presets"; [theory](../technical-reference/adaptive-iptw.md) |
| How do I use my own scikit-learn models? | `outcome_learner=`, `treatment_learner=` and other `ModelSpec` settings | [methods and learners](methods-learners.md), "Learner choices"; [theory](../technical-reference/nuisance-learners.md) |
| How are the models trained out of fold, and when are the folds reused? | `CrossFitting` | [methods and learners](methods-learners.md), "Two fold layers"; [theory](../technical-reference/cv-tmle.md) |
| How does the correction step stop, and which other fitting options exist? | `Targeting`, `ModelSpec` | [settings reference](settings.md); [theory](../technical-reference/point-treatment-tmle.md#stopping-rules-step-bound-and-best-iterate) |
| How are intervals, bands, diagnostics and sensitivity analyses computed? | `Inference`; `result.diagnostics`, `result.sensitivity` | [results](results-assessment.md), [checks](assessment.md) and [sensitivity](sensitivity.md); [theory](../technical-reference/inference.md) |
| How long will a fit take, and how do I make it faster or fit in memory? | `planned_fits`, `n_jobs`, `set_thread_limit`, `max_memory_bytes`, `RefutationCheckpoint` | [run time and memory](run-time-and-memory.md) |
| How far does the evidence for analyses over time reach, in number of treatment times? | studies cover three to five times; a fit no study covers is marked experimental | [analyses over time](longitudinal.md); [theory](../technical-reference/longitudinal-tmle.md) |

## Why a request can be refused

A refusal often marks a scientific boundary rather than a missing feature. Every `CleverlyError` carries `error.refusal`, and `error.refusal.reason` holds one of
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
Do not switch to a similarly named estimand to get around a refusal unless it is your question.

## What is not implemented

Natural and interventional mediation effects, the part of an effect that runs through a later
variable, are roadmap item [X5](../roadmap.md); controlled direct and separable effects ship.
Learning an estimate's weights directly, without a treatment model (the Riesz representer), is
roadmap items R1 and R2; `effect.available_methods()` lists `riesz_tmle` as unavailable. Effects
found from a causal graph alone, front-door adjustment, instrumental variables (a variable that
moves the treatment and nothing else), and carrying an effect over to another population have no
roadmap item and are outside the current scope. The EP learner fits no target other than a
conditional effect and a learned rule ([conditional
effects](../technical-reference/conditional-effects.md)).

## Extending the package

:::{dropdown} Details: the protocols a contributor implements

The public protocols are `Estimand`, `IdentificationProvider`, `EstimationMethod`, and
`CausalResult`.

- A custom provider returns an `IdentifiedEffect` only when it can name the observed-data
  functional and its assumptions.
- A custom method declares which functionals it supports and returns a result with stable
  parameter keys.
- A new registered analytic target needs the oracle, Gateaux, remainder, and mutation evidence
  that the [evidence manifest](../technical-reference/evidence.md) describes. A typed wrapper
  certifies nothing.

:::
