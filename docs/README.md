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

[Examples](examples/index.md) carries nine tutorials and the TWINS notebook. Six mirror the six
method entries in the technical reference. The other three cover sections inside an entry: the
intervention axes and survey non-response inside the point-treatment entry, and longitudinal event
outcomes inside the longitudinal entry. A new method entry needs a new tutorial, and the two link
to each other in both directions.

All nine share one narrative. A health plan's hospital network evaluates care-transition
navigation. A new tutorial joins that program rather than inventing a scenario.

## The record documents

These are the documents no reader path introduces on its own.

| document | what is in it |
| --- | --- |
| [Sensitivity and validation methods](technical-reference/validation-methods.md) | every instrument for reviewing an implemented method: why you use it, what it tells you, and how it tells you |
| [Scope and refusals](technical-reference/scope-and-refusals.md) | how to read a refusal, and where a multi-valued treatment is supported |
| [Roadmap](roadmap.md) | source-backed work in one binding sequence, followed by hard-stopped future investigations, with implementation contracts, refusals, and evidence gates |
| [Architecture invariants](architecture-invariants.md) | cross-module constraints and standing decisions, each with the condition that would reopen it |
| [Evidence](technical-reference/evidence.md) | per registered estimand: which oracle, Gateaux, remainder, identity, and variant instruments exist, plus their blind spots |
| [Implementation validation grid](technical-reference/method-evidence/validation-grid.md) | every registered study in one table: the method, the canonical implementation compared, the counts, and the declared limits |
| [Implementation validation studies](technical-reference/method-evidence/index.md) | one row per committed test: what it checked, what its own endpoints required, and the verdict |
| [Method benchmarking strategy](development/method-benchmarking.md) | how R comparisons and independent statistical-property studies are designed, registered, and accepted |
| [Test tiers and gates](development/testing-strategy.md) | which tier a change has to satisfy, and which deprecated studies no longer run |
| [Contributing](development/contributing.md) | how to set up the project, which checks a change needs, and where each working rule is written down |
| [Pull requests](development/pull-requests.md) | the commit style, the pull request body, what each CI job checks, and what none of them checks |
| [DR-TMLE](technical-reference/dr-tmle/index.md) | the doubly-robust variant's production contract: the supported estimands and refusals, what Theorem 1 covers, the targeting and cross-fitting choices, the nuisance conditions the interval is conditional on, and the diagnostics to inspect |
| [References](references.md) | every paper a derivation is read off, with the locators the prose cites |
