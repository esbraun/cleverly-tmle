# F5's preregistration, its two cohorts, and the nomination

[`prereg.json`](prereg.json) is the frozen design of [piece
F5](../../docs/roadmap.md#f5-the-terminal-experiment) — the terminal experiment — committed
**before the first inferential fit**, and it carries **both** phases: the confirmation's seeds,
replicate counts, margins, exclusions and decision rules are frozen before phase 1 begins, so the
only value that crosses between the phases is the identity of the nomination.

[`docs/drtmle/terminal-experiment.md`](../../docs/drtmle/terminal-experiment.md) is the record in
prose. This directory is what a run reads and what it wrote.

## What each part is, and what a run does with it

| part | what a run does with it |
| --- | --- |
| `prereg.json` → `rule` | every constant a verdict is read against, generated from the same `Column` tuple `verdict()` reads. `validate_prereg` refuses a run whose live constants differ — a threshold that moved after the freeze is not the threshold the verdict is read against |
| `prereg.json` → `phase1` | the arms, the three drops with a reason apiece, the cells, the sizes, the per-size draw counts, and `nominable` / `not_nominable` with a reason apiece so a reader can check what a nomination was *permitted* to select |
| `prereg.json` → `phase2` | the confirmation, frozen before phase 1: four arms, three cells, three sizes, `500` replicates × two batches, the Monte Carlo rule, the exclusions, and clause 3 recorded **not feasible** with its three citations |
| `prereg.json` → `disjointness` | what is checked against what — the two phase-1 cohorts against each other and against F4's, the confirmation batches against each other and against both cohorts |
| `*-selection.jsonl`, `*-audit.jsonl` | one row per `(cohort, cell, n, data_seed, arm, estimand)`. Three rows per fit — `ate`, `ey1`, `ey0` — of which `contrast_rows` reads the `ate` rows alone |
| `*-contrasts.jsonl` | every non-baseline arm against `glm-pooled`, on all 18 declared columns, in every `(cell, size)`, with the band and the four-way verdict |
| `nomination.json` | the instrument's verdict under the frozen rule, with a per-arm reason for every rejection |

## The order, and it is one pass

```
--phase prereg  ->  commit prereg.json here  ->  --phase select --cohort selection
                                              ->  --phase select --cohort audit
                                              ->  --phase nominate
```

`--phase prereg` fits nothing: it is seeds and constants. `--phase select` calls
`validate_prereg` **before** anything is fitted and exits non-zero on any complaint.
`--phase nominate` calls `refuse_dead_gates()` before a verdict is taken and **raises** if any
gating column has no finite reading anywhere — a veto that cannot fire reads in a table exactly
like one that fired and found nothing.

Reproducing it from this directory:

```bash
python -m benchmarks.drtmle_f5 --phase select --cohort selection
python -m benchmarks.drtmle_f5 --phase select --cohort audit
python -m benchmarks.drtmle_f5 --phase nominate --out evidence/f5-terminal
```

The first two write to `benchmarks/results/drtmle-f5/`, which is git-ignored; the artefacts were
copied here under their original timestamped names, which are their provenance. `--phase nominate`
globs `*.jsonl` in `--out`, excluding `-contrasts` and `-pilot`, so it reads the two cohort files
and re-derives the contrasts rather than trusting the committed ones.

## F5's committed rows

Digests and byte counts are of the **LF** bytes git stores, not of a `core.autocrlf` working copy
— which is the convention F4's table already uses, and `--phase nominate`'s emitted
`NOMINATION-SHA256` is the same quantity.

| file | rows | bytes | `sha256` |
| --- | --- | --- | --- |
| `prereg.json` | 1 | 21,702 | `85386d75bf69ce38…` |
| `20260807T152658-selection.jsonl` | 3,120 | 2,707,384 | `dee3927abaa5c73f…` |
| `20260807T171525-audit.jsonl` | 3,120 | 2,694,980 | `6560e1bb9cf45410…` |
| `20260807T152658-selection-contrasts.jsonl` | 288 | 92,566 | `f950b10b558a3338…` |
| `20260807T171525-audit-contrasts.jsonl` | 288 | 91,521 | `0a3627871c2d321c…` |
| `nomination.json` | 1 | 1,765 | `4bddcc8fcd69cc3e…` |

`3,120` rows is `1,040` fits per cohort — 208 draws × 5 arms — and `208` draws is
`(24 + 80) × 2` cells. Zero errors and zero identity failures on every arm of both cohorts.

## Two things about this freeze that are not F4's

**The manifest carries a second phase that has not run.** F4's `prereg.json` froze one pass;
this one freezes the confirmation as well, because F8 is retired into F5's phase 2 and its clause
1 asks for the confirmation's every input to be frozen in a commit *before dispatch*. So a reader
of this file can check phase 2's design against what phase 2 eventually does, which is the point.

**Clause 5's statistic was corrected before the cohort was read, and the artefact shows it.** The
nomination clause that asks the flexible candidate to carry real weight originally read
`flex_weight_min`; measured on 41 partial draws that sat at `0.0000` at every quantile including
the maximum, so **no arm could have passed** and the study would have returned *no nomination* as
an artefact of its own predicate. It now reads the candidate's **mean** weight, with
`flex_weight_min` retained beside it as a diagnostic no clause reads. `0.05` and `0.90` are
unchanged — only the quantity they apply to moved.

## What is not here

**Phase 2.** Nothing of the confirmation has run: `--phase confirm`, `--phase readout`,
`--phase verify` and `--phase cost` are unbuilt, and so is `ceiling_adequacy()`.

**The timing pilot's rows.** They carry `cohort="sizing"` and measure cost and nothing else;
`contrast_rows()` and `nominate()` **raise** on them rather than merely documenting that they
should not be read. Sizing comes from F4's committed `PILOT_PAIRED_SPREAD` and nothing else.

**The 41 partial selection draws that `boost-nested` was withdrawn from.** They are not the study
— the study is the two complete cohorts above — and no reading taken on that arm is carried
forward, reported as a result, or used to support any conclusion. The withdrawal was on cost, and
[the terminal experiment's record](../../docs/drtmle/terminal-experiment.md) states it, what it
cost, and why it is worded differently from the two drops that were taken before any fit.

**Final coverage.** It is not in either cohort — phase 1 cannot resolve a coverage move smaller
than about `0.10` at 24 and 80 draws, which is why `abs_coverage_gap` is a **veto only** here and
`unresolved` fires no veto. Coverage is settled in phase 2, at 500 replicates per batch.
