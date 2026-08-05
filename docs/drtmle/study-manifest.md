# The C3c study manifest

Everything needed to identify, retrieve and check the per-replicate evidence behind
[the coverage study](coverage-study.md#what-the-study-measured), the
[gate readout](coverage-study.md#the-gates-read-out-clause-by-clause) and every number the
[investigation log](investigation-log.md#what-the-c3c-dispatch-measured) reads off it.

**This file exists because a summary table is a transcription of the evidence and not the
evidence.** Six thousand fits produced ten summary tables; the tables are what the roadmap
argues from, and until this manifest landed the only copy of the rows behind them was a CI
artefact with a ninety-day retention. That is the [standing
decision](../roadmap.md#standing-decisions) this file is the evidence for.

## What was run

| | |
| --- | --- |
| workflow | `.github/workflows/drtmle-coverage.yml`, `workflow_dispatch`, one job per cell |
| code | `main` at `0033c824e9e7b9cc0049001c19ed7ab45eb0eb5c` — **no code change before or between the batches** |
| tier | 2 — both nuisances injected at prescribed rates, reductions fitted |
| cells | `q-drift`, `g-drift` |
| sizes | `600`, `1200`, `2400` |
| replicates | 250 per cell per size |
| estimators | `TMLE` and `DRTMLE` on the same draw, paired |
| estimands | `ate`, `ey1`, `ey0` |
| seeds | batch A `20250801`, batch B `20250802`; batch B dispatched after A completed |
| reduced learner | `glm` — `benchmarks/drtmle_coverage.REDUCED_LEARNER` |
| `--evaluation-n` | `2000`, from the `EVALUATION_SEED = 90_000_000` stream |
| `--jobs` | `2` |
| totals | 3,000 draws, **6,000 fits**, four artefacts |

**Batch A is not a fully fresh confirmation.** The two `SeedSequence` streams are prefix-stable,
so A shares the **pilot's data seeds** while drawing its own fold splits; batch B's stream is
fresh. Both agree on every qualitative claim. See
[the note under the gate readout](coverage-study.md#the-gates-read-out-clause-by-clause).

## The four artefacts

Digests, sizes and expiries are as reported by the GitHub Actions API on 2026-08-05.

| batch | cell | run | artefact | bytes | `sha256` |
| --- | --- | --- | --- | --- | --- |
| A | `q-drift` | `30979765029` | `8921415297` | 527,401 | `611335776f97c554f2ac209be14933f35df1b4ed57cc2739f1575812133c55f1` |
| A | `g-drift` | `30979765029` | `8922309841` | 522,473 | `88c07fe57646983c227ef5dbbbc1c3669968c8bd75d2f2cc19dd2c178410102c` |
| B | `q-drift` | `30987423687` | `8924840239` | 527,418 | `197eac8ba3f39f4419662925875da182f5527ff41d9396c9128ed610ffa52aba` |
| B | `g-drift` | `30987423687` | `8925807320` | 523,708 | `4781692d0799daf1c7499846e6d27612a40af6ee764c9387c6f7b4f0de041531` |

Both runs report `head_sha 0033c824e9e7b9cc0049001c19ed7ab45eb0eb5c`, matching the code row above.
**Retention expires 2026-11-03**, at which point the digests here are a record of what was
measured and no longer a way to obtain it.

> **The rows themselves are not in this commit, and that is a gap rather than a decision.**
> Actions artefacts are served from `*.blob.core.windows.net`, which the sandbox this manifest
> was written in cannot reach — the request is refused at the proxy, not by GitHub. So this file
> lands with the identifiers and digests verified against the API and **without** the payload.
> Retrieving it needs a machine with ordinary outbound access, before the retention date:
>
> ```bash
> gh run download 30979765029 --repo esbraun/cleverly-tmle --dir evidence/c3c/batch-a
> gh run download 30987423687 --repo esbraun/cleverly-tmle --dir evidence/c3c/batch-b
> sha256sum evidence/c3c/batch-*/*.jsonl   # against the table above, on the zip as fetched
> ```
>
> That is the first task of [E0's follow-up](../roadmap.md#e-what-c3c-handed-back), and until it
> is done this manifest names evidence it does not carry.

## The row schema

One JSON object per line, one line per `(cell, n, data_seed, estimator, estimand)`. The record is
`benchmarks.drtmle_coverage.Replicate`, which is flat and JSON-serialisable on purpose — a nested
record would need a schema before a reader could do anything with it. Fields, as of `0033c82`:

| field | meaning |
| --- | --- |
| `cell`, `n`, `data_seed`, `fold_seed` | which draw |
| `estimator`, `estimand` | `TMLE`/`DRTMLE`; `ate`/`ey1`/`ey0` |
| `truth`, `psi`, `std_error`, `lower`, `upper`, `covered` | the estimate and its interval |
| `valid`, `identity_failures`, `score_failures` | `score_check().passed` and its two causes, counted apart because gate 1 asks for them apart |
| `contract`, `initial_clip_share`, `margin`, `gr1_margin` | the per-fit truncation witness — item 25's label |
| `exit_reason`, `failure`, `rounds`, `seconds` | how the alternation ended. **This is where the 99 invalid fits are classified**, and [E3](../roadmap.md#e-what-c3c-handed-back) is the pull request that replays them |
| `r2`, `r2_targeted` | the plain remainder at the initial and at the targeted regression — the regime-entry column |
| `p0_curve`, `pn_curve` | `P₀D̂` off the evaluation draw, and `PₙD̂`, which targeting drove to zero |
| `remaining`, `root_n_remaining` | the corrected remainder Theorem 1 assumes negligible, and its `√n` scaling — **item 13's column** |
| `branch_q`, `branch_g`, `branch_error` | the two appendix branches' second-order halves and their binning error. The `M` terms are refused rather than approximated, so these are not the full theorem terms |

**A reader deriving a rate from a single row's `√n R_remaining` is reading the quadrature, not the
remainder**: at `m = 2,000` the companion's own error is of order `1.0/√m ≈ 0.023` per replicate,
which `√n` multiplies. Only the replicate mean with its Monte Carlo error means anything.

**[E1](../roadmap.md#what-e1-landed-and-what-e1b-withdrew) has since put that error in a column**, and the schema above is
therefore the schema **of these four artefacts** rather than of the harness as it now stands. A
`0033c82` row has no `companion_se`, `companion_replicate_se`, `companion_rule`, `companion_rows` or
`companion_scramble` in
it, and a reader joining these files against a later run's has to know that: the four fields are
absent rather than null. What they would have said, had they existed, is measured — at the i.i.d.
rule these runs used, the evaluation draw accounts for a large share of each cell's across-draw
spread, so the `± 0.09` above is not the estimator's alone.

## The second artefact: score rows

**Runs from [E1](../roadmap.md#what-e1-landed-and-what-e1b-withdrew) onwards write two files**, `<stamp>.jsonl` and
`<stamp>-scores.jsonl`, from one timestamp so they join. The four artefacts above predate it and
have only the first.

One JSON object per line, one line per `(cell, n, data_seed, estimator, score row)` — the grain is
the **fit** rather than the estimand, which is why it is a second file: `Replicate` is per-estimand
and nesting a per-fit fact inside it would store every row three times and break the flatness this
schema rests on. The record is `benchmarks.drtmle_coverage.ScoreRow` and it carries every field of
the library's `ScoreCheckRow`, which `tests/unit/test_drtmle_coverage.py` pins structurally so a
field added there cannot silently stop being carried.

| field | meaning |
| --- | --- |
| `cell`, `n`, `data_seed`, `fold_seed`, `estimator` | the fit, and the join key against the file above |
| `tolerance`, `corrected`, `passed_overall` | the check's own context, repeated per row so a line means something on its own |
| `name`, `kind` | which equation, and which of `correction` / `identity` / `diagnostic` |
| `score`, `threshold`, `std_error`, `ratio` | the number, the bar it is read against, and their quotient |
| `passed`, `converged`, `n_iter`, `method`, `failure` | what the solver did and why it stopped |
| `score_initial`, `reduction` | the same score before targeting moved anything, and the factor. **This is the field a count cannot replace**: a score that started near zero had nothing to do, which is a different situation from one driven down, and only the second is evidence targeting worked |
| `hessian_condition` | how well `epsilon` is identified, which can be poor where the score looks solved |
| `folds_converged`, `folds_total` | the pair off `ScoreCheckRow.folds_converged`, split in two because a tuple in JSON is a list whose order a reader has to know |

**Nothing is filtered.** Writing only the failing rows would make the file unable to answer the
`score_initial` question above, and it is the file [E3](../roadmap.md#e-what-c3c-handed-back)
classifies the 99 invalid fits from — `valid`, `identity_failures` and `score_failures` on the
replicate rows say *how many* and say nothing about *which*.

## Regenerating the tables

The summary tables are produced by the same module that writes the rows, so a regeneration is a
re-read rather than a re-fit:

```bash
python -m benchmarks.drtmle_coverage --tier 2 --cells q-drift g-drift \
    --sizes 600 1200 2400 --replicates 250 --seed 20250801 \
    --evaluation-n 2000 --jobs 2 --rows
```

**Do not run that in a small container.** It is 250 replicates a cell at 2.7s–4.7s a fit —
77 to 112 minutes per cell on one GitHub runner, and the whole study is four such jobs. Dispatch
the workflow and read the tables out of the job log, exactly as the study did.

## Numbers checkable against this manifest

Spot checks that a reader can do against the summary tables without the rows, and that a
regeneration must reproduce:

- **99 invalid fits of 3,000**, all `DRTMLE`, all `score` failures and no `identity` failures —
  the per-cell rates `0.028/0.008/0.012` and `0.032/0.028/0.008` in `q-drift` and
  `0.072/0.052/0.032` and `0.060/0.036/0.028` in `g-drift` sum to `12 + 17 + 39 + 31`.
- **Bound-active draws** `6/5/4` and `15/8/3` in `q-drift`, `11/18/11` and `22/13/17` in
  `g-drift`, out of 250 — the `1.2%`–`8.8%` of clause 1.0.
- **The paired gap** at `n = 2,400` in `q-drift`: `+0.312 ± 0.031` and `+0.376 ± 0.033`.
- **The `se ratio`** `0.903` in `q-drift` at the largest size, in both batches.
