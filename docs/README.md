# Documentation source map

This directory contains the documentation for `cleverly`, the Python toolbox for causal studies
built on targeted maximum likelihood estimation (TMLE).

[`index.md`](index.md) is the Sphinx/MyST site root. The main reader paths are
[Getting started](getting-started/index.md), [Workflow](workflow.md),
[User guide](user-guide/index.md), [Technical reference](technical-reference/index.md),
[Examples](examples/index.md), and [Python API](api/index.md). The API object pages are generated
from docstrings during the build; the narrative source remains ordinary Markdown that is readable
on GitHub.

Build the warning-as-error site locally with `nox -s docs`.

The documents below form the scientific, user, and engineering record. The
[development reference](development/index.md) is deliberately limited to the roadmap,
architecture invariants, and method-benchmarking strategy.

| document | what is in it |
| --- | --- |
| [Recipe compendium](user-guide.md) | one worked recipe per capability retained alongside the task-oriented user guide |
| [Migration guide](migration.md) | runnable old-to-new workflows, complete constructor-argument maps, changed defaults, the static audit tool, and the final-old-API tag |
| [Technical appendix](methodology.md) | per algorithm: the estimand, efficient influence curve, second-order remainder, failure witness, and method evidence grid |
| [Roadmap](roadmap.md) | all proposed work in parallel priority tracks, with governing sources, planned interfaces, implementation contracts, refusals, and evidence gates |
| [Architecture invariants](architecture-invariants.md) | cross-module constraints and standing decisions, each with the condition that would reopen it |
| [Evidence](technical-reference/evidence.md) | per registered estimand: which oracle, Gateaux, remainder, identity, and variant instruments exist — and which mistakes none of them would see |
| [Method evidence studies](technical-reference/method-evidence.md) | dedicated test-by-test result pages for registered same-DGP comparisons and statistical-property studies |
| [Method benchmarking strategy](development/method-benchmarking.md) | how R comparisons and independent statistical-property studies are designed, registered, and accepted |
| [DR-TMLE](drtmle.md) | the doubly-robust variant's production contract: supported estimands and refusals, what Theorem 1 covers, the targeting and cross-fitting choices, the nuisance conditions the interval is conditional on, and the diagnostics to inspect |
| [References](references.md) | every paper a derivation is read off, with the locators the prose cites |

## Where engineering constraints live

Cross-module constraints, standing decisions, and the conditions that would reopen them belong in
[architecture invariants](architecture-invariants.md). Proposed work belongs in the
[roadmap](roadmap.md). Machine-specific exploration remains in Git history rather than being
published as a durable fact about the package.
