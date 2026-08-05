# The documentation set

Six things a reader might want, and where each of them is. Nothing here is generated —
every file is hand-written and is expected to be read.

| document | what is in it |
| --- | --- |
| [User guide](user-guide.md) | one runnable recipe per capability: multi-arm treatments, dynamic and stochastic regimes, continuous doses, incremental interventions, marginal structural models, longitudinal and survival fits, C-TMLE, cross-fitting, weights |
| [Technical appendix](methodology.md) | per algorithm: the estimand, its efficient influence curve, the second-order remainder, and the test that fails when it is built wrong |
| [Roadmap](roadmap.md) | what has landed, what is open, and — at the top — the [standing decisions](roadmap.md#standing-decisions) this package has taken and will not re-litigate |
| [`DRTMLE`](drtmle/) | the one variant still in progress: the theorem concordance, the validation plan, the [coverage study's design](drtmle/coverage-study.md) and [its evidence manifest](drtmle/study-manifest.md), the investigation log, and the external review that started it |
| [Benchmarks](benchmarks/) | where a fit's time goes, and what compiling or parallelising the package's own arithmetic would buy — which, measured properly, is not enough for a dependency |
| [References](references.md) | every paper a derivation is read off, with the locators the prose cites |

## Where a decision lives

**Read [the standing-decisions table](roadmap.md#standing-decisions) before reading a
report.** It is one screen, it links to the evidence for each row, and it exists so that
"why is there no `numba` dependency" is answered without opening seven measurement
write-ups. The reports are the *evidence*; the roadmap is the *verdict*.

## Where the reports do not live

`benchmarks/results/` is output. A benchmark run writes there, it is git-ignored, and
nothing in it is committed — a `results.jsonl` from a four-core container would read as a
fact about the package rather than about that box. The write-ups that interpret such runs
are in [`docs/benchmarks/`](benchmarks/).
