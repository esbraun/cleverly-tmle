# Documentation source map

This directory contains the documentation for `cleverly`, the Python toolbox for causal studies
built on targeted maximum likelihood estimation (TMLE).

[`index.md`](index.md) is the Sphinx/MyST site root. The main reader paths are
[Getting started](getting-started/index.md), [Workflow](workflow.md),
[User guide](user-guide/index.md), [Technical reference](technical-reference/index.md),
[Examples](examples/index.md), and [Python API](api/index.md). The API object pages are generated
from docstrings during the build; the narrative source remains ordinary Markdown that is readable
on GitHub.

Build the warning-as-error site locally with `nox -s docs`. Read the
[production documentation plan](documentation-plan.md) for the reference review, information
architecture, technical-coverage contract, publishing design, and acceptance gates.

The documents below are the scientific and engineering record routed through the site's
[development reference](development/index.md).

| document | what is in it |
| --- | --- |
| [Recipe compendium](user-guide.md) | one worked recipe per capability retained alongside the task-oriented user guide |
| [Migration guide](migration.md) | runnable old-to-new workflows, complete constructor-argument maps, changed defaults, the static audit tool, and the final-old-API tag |
| [Technical appendix](methodology.md) | per algorithm: the estimand, its efficient influence curve, the second-order remainder, and the test that fails when it is built wrong |
| [Roadmap](roadmap.md) | proposed methods in priority order, with the publication and canonical-source evidence required before implementation |
| [Public API redesign](public-api-redesign.md) | the accepted design gate for the study/identify/estimate object model, Riesz and EP integration, ordered work packages, and exit evidence — work package 1 is implemented; later packages remain proposals |
| [Nested Riesz implementation plan](riesz-implementation-plan.md) | the source-audited, review-gated plan for work package 3: typed stages and representers, stage order, direct loss, catalog boundary, persistence, mutations, and implementation commits |
| [Architecture invariants](architecture-invariants.md) | cross-module constraints and standing decisions, each with the condition that would reopen it |
| [Evidence](evidence.md) | per registered estimand: which instruments check its influence curve — oracle law, Gateaux comparison, remainder rate, exact identity — and which mistakes none of them would see |
| [DR-TMLE](drtmle.md) | the doubly-robust variant's production contract: supported estimands and refusals, what Theorem 1 covers, the targeting and cross-fitting choices, the nuisance conditions the interval is conditional on, and the diagnostics to inspect |
| [Benchmarks](benchmarks/README.md) | where a fit's time goes, and what compiling or parallelising the package's own arithmetic would buy — which, measured properly, is not enough for a dependency |
| [References](references.md) | every paper a derivation is read off, with the locators the prose cites |

## Where engineering constraints live

Cross-module constraints, the standing decisions that go with them, and the conditions that would
reopen each one all belong in [architecture invariants](architecture-invariants.md). Detailed
performance evidence belongs in the [benchmark reports](benchmarks/README.md). The roadmap is reserved for
proposed work.

## Where the reports do not live

`benchmarks/results/` is output. A benchmark run writes there, it is git-ignored, and
nothing in it is committed — a `results.jsonl` from a four-core container would read as a
fact about the package rather than about that box. The write-ups that interpret such runs
are in [`docs/benchmarks/`](benchmarks/README.md).
