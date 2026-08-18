# cleverly

```{raw} html
<div class="cleverly-hero">
  <span class="cleverly-eyebrow">Causal inference for Python · Alpha</span>
  <p class="cleverly-hero-title">Start with the causal question.<br>Then choose the estimator.</p>
  <p class="cleverly-hero-copy">Targeted maximum likelihood estimation for point and
  longitudinal treatment settings, with typed estimands, influence-curve inference,
  diagnostics, and sensitivity analysis.</p>
</div>
```

::::{grid} 1 2 2 2
:gutter: 2
:class-container: cleverly-actions

:::{grid-item}
```{button-ref} getting-started/index
:color: primary
:expand:

Get started
```
:::

:::{grid-item}
```{button-ref} api/index
:color: secondary
:outline:
:expand:

Browse the Python API
```
:::

::::

:::{warning}
`cleverly` is alpha software. Unsupported combinations fail before fitting, but the public API may
still change. Pin a commit for reproducible work and read the identification statement returned by
your analysis.
:::

## Install from GitHub

```bash
python -m pip install "git+https://github.com/esbraun/cleverly-tmle.git"
```

Python 3.11 or newer is required. The [installation guide](getting-started/installation.md) covers
optional backends, development environments, and reproducible commit-pinned installs.

## One workflow, explicit decisions

```{raw} html
<ol class="cleverly-workflow" aria-label="The cleverly analysis workflow">
  <li><strong>Design</strong><span>Declare the observed data</span></li>
  <li><strong>Estimand</strong><span>Name the causal question</span></li>
  <li><strong>Identify</strong><span>Inspect assumptions</span></li>
  <li><strong>Estimate</strong><span>Configure and fit</span></li>
  <li><strong>Result</strong><span>Assess and report</span></li>
</ol>
```

The separation is deliberate: changing an estimation method does not silently change the
scientific question. Follow the [analysis workflow](workflow.md) from formulation through
diagnostics and reporting.

## Choose your path

::::{grid} 1 2 3 3
:gutter: 3
:class-container: cleverly-card-grid

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

## Roadmap

- Build the general nested Riesz engine and its initial evidence-gated catalog, including analytic
  and direct representers, nested composition, diagnostics, and persistence.
- Add optional DoWhy integration for graph-based identification and backdoor translation while
  keeping the core package standalone.
- Add EP learning for heterogeneous effects, beginning with conditional average treatment effects
  and conditional relative risks.
- Expand the estimand catalog target by target, with a separate derivation, evidence record,
  refusal contract, and statistical study for each family.

These are accepted directions, not implemented release claims. The detailed evidence and source
requirements are in the [roadmap](roadmap.md),
[public API redesign](public-api-redesign.md), and
[nested Riesz implementation plan](riesz-implementation-plan.md).

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
