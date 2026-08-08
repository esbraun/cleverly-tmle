# The documentation set

Six things a reader might want, and where each of them is. Nothing here is generated —
every file is hand-written and is expected to be read.

| document | what is in it |
| --- | --- |
| [User guide](user-guide.md) | one runnable recipe per capability: multi-arm treatments, dynamic and stochastic regimes, continuous doses, incremental interventions, marginal structural models, longitudinal and survival fits, C-TMLE, cross-fitting, weights |
| [Technical appendix](methodology.md) | per algorithm: the estimand, its efficient influence curve, the second-order remainder, and the test that fails when it is built wrong |
| [Roadmap](roadmap.md) | what ships, its current limitations, candidate features, and the evidence-based [standing decisions](roadmap.md#standing-decisions) that guide future work |
| [DR-TMLE](drtmle.md) | the doubly-robust variant's production contract: supported estimands and refusals, what Theorem 1 covers, the targeting and cross-fitting choices, the nuisance conditions the interval is conditional on, and the diagnostics to inspect |
| [Benchmarks](benchmarks/) | where a fit's time goes, and what compiling or parallelising the package's own arithmetic would buy — which, measured properly, is not enough for a dependency |
| [References](references.md) | every paper a derivation is read off, with the locators the prose cites |

## Where a decision lives

**Read [the standing-decisions table](roadmap.md#standing-decisions) before reopening an
engineering choice.** Each row states the current evidence and the condition that would justify a
different design. The reports are the evidence; the roadmap is the current verdict.

## Where the reports do not live

`benchmarks/results/` is output. A benchmark run writes there, it is git-ignored, and
nothing in it is committed — a `results.jsonl` from a four-core container would read as a
fact about the package rather than about that box. The write-ups that interpret such runs
are in [`docs/benchmarks/`](benchmarks/).
