# cleverly

```{raw} html
<div class="cleverly-hero">
  <span class="cleverly-eyebrow">Causal studies with TMLE · Alpha</span>
  <p class="cleverly-hero-title">Estimate. Validate. Stress-test.</p>
  <p class="cleverly-hero-copy">Build a causal study on TMLE. Name the question, inspect the
  identification argument, fit, and then probe the result with validation, diagnostics, and
  sensitivity analyses.</p>
</div>
```

*It is named for TMLE's clever covariate. Even the jokes in this toolbox are targeted.*

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
`cleverly` is alpha software. Unsupported combinations fail before fitting. The public API may
still change, so pin a commit for reproducible work. Read the identification statement your
analysis returns.
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

The separation is deliberate. Changing an estimation method does not silently change the
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

## What is implemented, and what is planned

The [implementation matrix](technical-reference/index.md#implementation-matrix) lists every
implementation family the package ships. Each row names the theory, the source module, the
external provenance, and the correctness evidence.

The [roadmap](roadmap.md) lists every proposed direction in one table, with its readiness, its
governing sources, and the evidence gate it has to clear. A roadmap entry is an accepted direction
and not a release claim.

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
