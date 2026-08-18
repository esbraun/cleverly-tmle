# Capabilities and refusals

## Shipped analysis families

The [technical implementation matrix](../technical-reference/index.md) is the authoritative
reader-facing inventory. In brief, the package supports point and longitudinal TMLE, arm and
regimen contrasts, continuous shifts, incremental interventions, MSM projections, controlled
direct effects, C-TMLE, DR-TMLE, cross-fitting, influence-curve inference, sensitivity analysis,
and post-fit diagnostics in the combinations documented there.

## Why a request can be refused

An error at identification or method selection is often a scientific boundary, not merely an API
limitation. Refusals fall into three groups:

1. **Not implemented:** the parameter is well posed, but its identification map, influence
   function, targeting step, or evidence gate is missing.
2. **Different question:** the supplied design and estimand imply another causal parameter, such
   as requesting an arm ATE for a continuous modified treatment policy.
3. **Invalid construction:** the requested object is undefined, nonidentified under the declared
   design, or cannot produce valid inference with the supplied information.

`CapabilityError` and `MethodConfigurationError` explain the boundary before nuisance fitting.
Do not work around one by switching to a similarly named estimand unless that estimand is genuinely
the scientific question.

## Explicitly unimplemented

Graph identification, front-door adjustment, instrumental-variable effects, mediation
decompositions, transport, direct Riesz learning, and EP learning are proposals or outside the
current scope. The [roadmap](../roadmap.md) states the evidence required before a proposed method
can become a release claim.

## Migration

The clean alpha API removed old one-call constructors. Use the [migration guide](../migration.md)
for the complete argument map and the static audit tool.
