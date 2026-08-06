# F4's preregistration

[`prereg.json`](prereg.json) is the frozen design of [piece
F4](../../docs/roadmap.md#f-localize-the-shortfall-before-changing-anything) — the construction
contrasts — committed **before the first decision fit**, which is what that row asks for and
what makes the verdicts it is read against verdicts rather than choices.

[`docs/drtmle/construction-contrasts.md`](../../docs/drtmle/construction-contrasts.md) is the
record in prose; [`docs/drtmle/validation-plan.md`](../../docs/drtmle/validation-plan.md)'s
*The contrast rule, frozen before the dispatch* is the rule in prose. This file is what a run
reads.

## What each part is, and what a run does with it

| part | what a run does with it |
| --- | --- |
| `rule` | every constant a verdict is read against, including the sizing arithmetic. `validate_prereg` refuses a run whose live constants differ from these — a threshold that moved after the freeze is not the threshold the verdict is read against |
| `configuration` | what the instrument was built at: the arms, the cells, the sizes, the tier, the reduced learner, the quadrature rule. The keys in `PINNED_CONFIGURATION` are refused if they differ |
| `cohorts` | the `(data_seed, fold_seed)` pairs of the **selection** and **audit** halves, from `SeedSequence(20250801).spawn(3)`'s first two children. Disjoint, and checked again at the run on the **data** seed |
| `contrasts` | the five paired cohort contrasts, one factor each, **and the order they are read in** |

## The order, and it is one pass

```
--phase prereg   ->   commit prereg.json here   ->   --phase run, per (cell, size, cohort)
```

`--phase prereg` fits nothing: it is seeds and constants. `--phase run` calls
`validate_prereg` **before** anything is fitted and exits non-zero on any complaint, so a
dispatch against a moved rule fails rather than reports.

## Two things about this freeze that are not E2R's

**There is no data-dependent selection to freeze, so the manifest is produced where the study
is dispatched from.** E2R's `selection.json` had to come back out of a job log, because a rung
chosen by a first dispatch sits between two runs of the same session. Nothing here is chosen by
a run — the design is chosen by a derivation — so `--phase prereg` is a local command and the
commit of its output is the freeze.

**The two sizes carry different draw counts, and the manifest says what each can answer.**
`rule.sizing` records, per size, the worst measured paired spread, the count that spread
implies at the declared half-width, the count committed, and — where the two differ — that the
size is powered for `moved` and not for `flat`. At `n = 2,400` the study resolves both; at
`n = 600` the same arithmetic asks for 1,755 draws, which is out of budget by an order of
magnitude, so 24 are committed and the limit is **declared here rather than discovered in the
result**. Every contrast row also carries its realized minimum detectable effect.

The spreads those counts come from were measured on a **third** child of the same seed
sequence, disjoint from both cohorts and spent — `SeedSequence.spawn` gives child `i` the same
state whatever `n` is, so reserving it leaves the two cohorts byte-identical to what `spawn(2)`
produced, and `tests/unit/test_drtmle_construction.py` checks that rather than asserting it. A
study sized on the draws it then reads would be sizing on its own outcome.

## What is not here

**The sixth factor.** F4's matrix has six contrasts and this manifest declares five. The
truncation convention is read by `benchmarks.drtmle_construction.truncation_reading` on the two
frozen trace fixtures, exactly and deterministically, because a cohort of tier-2 draws cannot
answer it: that law's initial mechanism has a clip share of `0.0000` even at a bound of
`(0.15, 0.85)`, so the two arms are bit-identical and a cohort of them would report a null on a
contrast that could not have been non-null. The record says what the reading found.

**Final coverage.** It is in neither diagnostic — that is F8's and only F8's.
