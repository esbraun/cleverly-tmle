# Documentation source map

This directory contains the documentation for `cleverly`, the Python toolbox for causal studies
built on targeted maximum likelihood estimation (TMLE).

[`index.md`](index.md) is the Sphinx/MyST site root. The main reader paths are
[Getting started](getting-started/index.md), [Workflow](workflow.md),
[User guide](user-guide/index.md), [Technical reference](technical-reference/index.md),
[Examples](examples/index.md), and [Python API](api/index.md). The API object pages are generated
from docstrings during the build; the narrative source remains ordinary Markdown that is readable
on GitHub.

[Examples](examples/index.md) carries eight tutorials and the TWINS notebook. Six mirror the six
method entries in the technical reference. The other two cover the intervention axes and survey
non-response, both of which are sections inside the point-treatment entry. A new method entry needs
a new tutorial, and the two are cross-linked in both directions.

All eight share one narrative: a health plan's hospital network evaluating care-transition
navigation. A new tutorial joins that program rather than inventing a scenario.

Build the warning-as-error site locally with `nox -s docs`.

The documents below form the scientific, user, and engineering record. The
[development reference](development/index.md) is deliberately limited to the roadmap,
architecture invariants, testing strategy, and method-benchmarking strategy.

| document | what is in it |
| --- | --- |
| [Recipe compendium](user-guide.md) | one worked recipe per capability retained alongside the task-oriented user guide |
| [Migration guide](migration.md) | runnable old-to-new workflows, complete constructor-argument maps, changed defaults, the static audit tool, and the final-old-API tag |
| [Sensitivity and validation methods](technical-reference/validation-methods.md) | every instrument for reviewing an implemented method: why you use it, what it tells you, and how it tells you |
| [Scope and refusals](technical-reference/scope-and-refusals.md) | how to read a refusal, and where a multi-valued treatment is supported |
| [Roadmap](roadmap.md) | all proposed work in parallel priority tracks, with governing sources, planned interfaces, implementation contracts, refusals, and evidence gates |
| [Architecture invariants](architecture-invariants.md) | cross-module constraints and standing decisions, each with the condition that would reopen it |
| [Evidence](technical-reference/evidence.md) | per registered estimand: which oracle, Gateaux, remainder, identity, and variant instruments exist, plus their blind spots |
| [Implementation validation grid](technical-reference/index.md#implementation-validation-grid) | every registered study in one table: the method, the canonical implementation compared, the counts, and the declared limits |
| [Implementation validation studies](technical-reference/method-evidence.md) | one row per committed test: what it checked, what its own endpoints required, and the verdict |
| [Method benchmarking strategy](development/method-benchmarking.md) | how R comparisons and independent statistical-property studies are designed, registered, and accepted |
| [DR-TMLE](drtmle.md) | the doubly-robust variant's production contract: supported estimands and refusals, what Theorem 1 covers, the targeting and cross-fitting choices, the nuisance conditions the interval is conditional on, and the diagnostics to inspect |
| [References](references.md) | every paper a derivation is read off, with the locators the prose cites |

## Where engineering constraints live

Cross-module constraints, standing decisions, and the conditions that would reopen them belong in
[architecture invariants](architecture-invariants.md). Proposed work belongs in the
[roadmap](roadmap.md). Machine-specific exploration remains in Git history rather than being
published as a durable fact about the package.
