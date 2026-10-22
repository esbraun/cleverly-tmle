# Capabilities and refusals

## Shipped analysis families

The [technical implementation matrix](../technical-reference/index.md) is the authoritative
reader-facing inventory. In brief, the package supports point and longitudinal TMLE, the sequentially doubly
robust longitudinal estimator, arm and regimen contrasts, modified treatment policies, incremental interventions, MSM projections, controlled
direct effects, C-TMLE, DR-TMLE, cross-fitting, influence-curve inference, sensitivity analysis,
and post-fit diagnostics in the combinations documented there. The longitudinal
family also includes Kennedy's incremental estimator `IPSI`. An incremental intervention takes
one odds factor for each level of a treatment with three or more levels, at a point treatment and
at a node of `IPSI`. The registered study [categorical incremental policies](../technical-reference/method-evidence/categorical-incremental.md) covers the support rows `T35` at a point treatment and `G43` over time. Any other such fit, for example with weights, clusters or a `delta=` argument, is labelled `experimental`.

A point-treatment functional also has the adaptive-propensity IPTW estimator, `AdaptiveIPTW`.
It reads the treatment and missingness mechanisms and a reduced regression of the outcome on the
fitted propensity, and it fits no outcome regression. It reports a point estimate for every
supported fit. The arm means and the targets built from them report an interval only when the
treatment probabilities are declared and the data are complete. A fit with an estimated mechanism
takes the status `estimated_mechanism_plugin`.

[Adaptive-propensity IPTW](../technical-reference/adaptive-iptw.md) states the contract. Its
evidence label is "Python property evidence only, with the registered study pending". Every estimate
reads `experimental` until a registered study covers its configuration.

| longitudinal estimator | reads | interval |
| --- | --- | --- |
| `LTMLE` | outcome regressions and mechanisms, with a fluctuation | the influence curve |
| `SDR` | the same, with the sequentially doubly robust construction | the influence curve |
| `IPSI` | the treatment mechanism | the influence curve of the incremental estimator |
| `SequentialGComputation` | the outcome and pseudo-outcome regressions | the stacked curve when every regression is a `GLM`. Otherwise the point only |
| `SequentialIPW` | the treatment and censoring mechanisms | the stacked curve when every mechanism is a `GLM`. Otherwise the point only |
| `StructuralNestedMean` | the outcome and treatment regressions on the full history, with a linear blip model | the influence curve, valid under the linear structural nested mean model with the declared blip features |
| `ParametricGFormula` | a model of each time-varying covariate, and the outcome or hazard models | the stacked curve with the simulation error when every model is a `GLM`. Otherwise the point only |

The two sequential baselines are reference estimators. They refuse `n_folds > 1` and `msm=`.
[Sequential g-computation and IPW](../technical-reference/sequential-baselines.md#refusals) lists
every refusal. `ParametricGFormula` refuses the same two requests, and it accepts a fixed
or rule-set continuous dose with a stated model reliance. [Parametric g-formula](../technical-reference/parametric-gformula.md#refusals)
lists its refusals.

`StructuralNestedMean` reports blip coefficients and static regimen contrasts. It refuses a rule, a
policy, censoring, survival and repeated outcomes. It refuses a static contrast when a feature
reads a time-varying covariate.
[Structural dynamic effects](../technical-reference/structural-dynamic-effects.md#refusals) lists
every refusal. Its evidence label is "comparator-backed", against EconML 0.17.0 `DynamicDML`. A cross-fitted
fit that a `G42` row covers reads `supported`, and any other fit reads `experimental`
([support label](../technical-reference/structural-dynamic-effects.md#support-label)).

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

A capability row carries the same record before any call. Read
[how to read a refusal](../technical-reference/scope-and-refusals.md#how-to-read-a-refusal) for the
action that each code asks of you.
Do not work around a refusal by switching to a similarly named estimand unless that estimand is
genuinely the scientific question.

## Explicitly unimplemented

Graph identification, front-door adjustment, instrumental-variable effects, mediation
decompositions, transport, direct Riesz learning, and EP learning are proposals or outside the
current scope. The [roadmap](../roadmap.md) states the evidence required before a proposed method
can become a release claim.

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
