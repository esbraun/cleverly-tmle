---
orphan: true
---

# Documentation source map

This directory holds the documentation for `cleverly`, the Python toolbox for causal studies built
on targeted maximum likelihood estimation (TMLE). This page is for contributors. Readers start at
[`index.md`](index.md), the Sphinx and MyST site root.

Build the warning-as-error site locally with `nox -s docs`.

The API object pages are generated from docstrings during the build. Every narrative source stays
ordinary Markdown that is readable on GitHub.

## The examples contract

[Examples](examples/index.md) carries sixteen tutorials and the TWINS notebook. Every method entry
in the technical reference has at least one tutorial, and each entry and its tutorials link to each
other in both directions. A new method entry needs a new tutorial, or a new section in a tutorial
that fits its question.

| method entry | tutorials |
| --- | --- |
| [point-treatment TMLE](technical-reference/point-treatment-tmle.md) | point-treatment-tmle, survey-nonresponse, interventions, population-impact, randomized-pilot, treatment-rules, schedules-and-intensity |
| [CV-TMLE](technical-reference/cv-tmle.md) | cross-fitting |
| [collaborative TMLE](technical-reference/collaborative-tmle.md) | collaborative-tmle |
| [MSM projections](technical-reference/msm-projections.md) | msm-projections |
| [longitudinal TMLE](technical-reference/longitudinal-tmle.md) | sequential-pricing, longitudinal-survival, sequential-offers, weekly-outcomes, discharge-to-readmission |
| [sequential g-computation and IPW](technical-reference/sequential-baselines.md) | sequential-offers |
| [parametric g-formula](technical-reference/parametric-gformula.md) | model-based-over-time |
| [structural dynamic effects](technical-reference/structural-dynamic-effects.md) | model-based-over-time |
| [DR-TMLE](technical-reference/dr-tmle/index.md) | randomized-pilot |
| [adaptive-propensity IPTW](technical-reference/adaptive-iptw.md) | randomized-pilot |

The tutorials share two narratives. Thirteen follow one health program: a health plan's hospital
network evaluates care-transition navigation. A new tutorial for a method joins that program rather
than inventing a scenario, unless the method needs a law that the program lacks. The pricing
tutorial is that case, because the health program has no continuous price. Three tutorials follow a
second program, a subscription retailer that sends weekly offers, counts weekly orders, and sets
monthly renewal prices.

[Worked examples by method family](development/example-notebooks.md#worked-examples-by-method-family)
maps each family of the implementation matrix to the tutorial that shows it.

## The record documents

These are the documents no reader path introduces on its own.

| document | what is in it |
| --- | --- |
| [Sensitivity and validation methods](technical-reference/validation-methods.md) | every instrument for reviewing an implemented method: why you use it, what it tells you, and how it tells you |
| [Scope and refusals](technical-reference/scope-and-refusals.md) | how to read a refusal, and where a multi-valued treatment is supported |
| [Roadmap](roadmap.md) | a remediation queue, then open work ordered by beta milestone, then hard-stopped future investigations, with implementation contracts, refusals, and evidence gates |
| [Architecture invariants](architecture-invariants.md) | cross-module constraints and standing decisions, each with the condition that would reopen it |
| [Evidence](technical-reference/evidence.md) | per registered estimand: which oracle, Gateaux, remainder, identity, and variant instruments exist, plus their blind spots |
| [Implementation validation grid](technical-reference/method-evidence/validation-grid.md) | every registered validation study in one table: the method, the canonical implementation compared, the counts, and the declared limits |
| [Implementation validation studies](technical-reference/method-evidence/index.md) | one row per committed test: what it checked, what its own endpoints required, and the verdict |
| [The red-cell ledger](technical-reference/method-evidence/red-cells.md) | every red verdict the registered studies publish, generated from the committed results, with the roadmap ask that owns each one |
| [Method benchmarking strategy](development/method-benchmarking.md) | how R comparisons and independent statistical-property studies are designed, registered, and accepted |
| [Fast tests and validation studies](development/testing-strategy.md) | which checks a change needs, and when to regenerate an affected study |
| [Writing for readers](development/writing-for-readers.md) | the reader of each documentation area, the writing rules, the library words to replace, and the reader review that checks a rewritten page |
| [Contributing](development/contributing.md) | how to set up the project, which checks a change needs, and where each working rule is written down |
| [Pull requests](development/pull-requests.md) | the commit style, the pull request body, what each CI job checks, and what none of them checks |
| [Releases](development/releases.md) | the alpha version rule, package gates, tag process, and Trusted Publishing setup |
| [Release notes template](development/release-notes-template.md) | the reviewed GitHub release structure, writing rules, and source precedents |
| [DR-TMLE](technical-reference/dr-tmle/index.md) | the doubly-robust variant's production contract: the supported estimands and refusals, what Theorem 1 covers, the targeting and cross-fitting choices, the nuisance conditions the interval is conditional on, and the diagnostics to inspect |
| [References](references.md) | every paper a derivation is read off, with the locators the prose cites |
