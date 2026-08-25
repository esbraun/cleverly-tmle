# User guide

The user guide explains the choices that sit between the quickstart and a defensible applied
analysis. Read it by task. Use the [Python API](../api/index.md) for signatures and the
[technical reference](../technical-reference/index.md) for derivations and evidence.

The public workflow has four visible stages:

```text
CausalStudy -> identify(typed estimand) -> IdentifiedEffect -> estimate(method) -> CausalResult
```

The study design owns the column roles. The estimand owns the causal question. The method owns
learning, targeting, inference, and runtime settings. That separation is what lets `cleverly`
refuse an unsupported combination before it fits any nuisance model.

The examples here are explanatory. The behavioural guarantees live in the test tiers, not in an
example.

```{toctree}
:maxdepth: 2

data-design
estimands
methods-learners
results-assessment
longitudinal
capabilities
```
