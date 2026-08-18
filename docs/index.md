# cleverly

```{raw} html
<div class="cleverly-hero">
  <p class="lead">Targeted maximum likelihood estimation for Python, organized around the
  causal question before the estimation method.</p>
</div>
```

`cleverly` provides typed causal estimands, targeted maximum likelihood estimation, influence-
curve inference, diagnostics, and sensitivity analysis for point and longitudinal treatment
settings. The workflow keeps the scientific question separate from the estimation machinery:

```text
study design -> typed estimand -> identified effect -> estimation method -> causal result
```

:::{warning}
`cleverly` is alpha software. Unsupported combinations fail before fitting, but the public API may
still change. Pin a commit for reproducible work and read the identification statement returned by
your analysis.
:::

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Getting started
:link: getting-started/index
:link-type: doc

Install the package, fit one average treatment effect, and learn how to read the result.
:::

:::{grid-item-card} Workflow
:link: workflow
:link-type: doc

Move from a causal question and observed-data design to diagnostics and reporting.
:::

:::{grid-item-card} User guide
:link: user-guide/index
:link-type: doc

Choose estimands, interventions, learners, methods, inference, and assessments.
:::

:::{grid-item-card} Technical reference
:link: technical-reference/index
:link-type: doc

Trace every implementation to theory, citations, source, provenance, and evidence.
:::

:::{grid-item-card} Examples
:link: examples/index
:link-type: doc

Follow complete point-treatment, intervention, longitudinal, and assessment workflows.
:::

:::{grid-item-card} Python API
:link: api/index
:link-type: doc

Look up supported objects, signatures, attributes, methods, and return types.
:::

::::

## What is implemented

- Point-treatment TMLE for binary and multi-valued treatments, missing outcomes, observation
  weights, clustering, and strata.
- Static, dynamic, and stochastic regimes; continuous modified treatment policies; incremental
  propensity-score interventions; and marginal structural model projections.
- Longitudinal TMLE for end-of-study, survival, and competing-risk outcomes.
- Cross-fitting, CV-TMLE, collaborative TMLE, DR-TMLE, influence-curve inference, simultaneous
  intervals, and bootstrap procedures.
- Post-fit validation, positivity and nuisance diagnostics, sensitivity analysis, variable
  importance, and safe result persistence.

Graph discovery, front-door adjustment, instrumental-variable effects, mediation decompositions,
and transport are outside the current scope: there is no object to construct, so nothing
approximates them. Direct Riesz learning and EP learning go further and are declared refusals —
`available_methods()` reports each as unavailable with the reason it is missing.

```{toctree}
:hidden:
:maxdepth: 2

getting-started/index
workflow
user-guide/index
technical-reference/index
examples/index
api/index
development/index
```
