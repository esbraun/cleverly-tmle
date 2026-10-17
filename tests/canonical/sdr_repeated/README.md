# Canonical evidence for repeated-outcome longitudinal SDR

This directory holds the study of the sequentially doubly robust estimator (`SDR`) on a repeated
outcome. `cleverly` fits each panel once with `outcome=Repeated(["Y1", "Y2", "Y3"])`. Each
horizon `h` is the end-of-study estimate for the panel cut after node `h` (Díaz, Williams,
Hoffman and Schenck 2023, Equation (4), Lemmas 3 and 4 and Theorem 4).

The truths come from the exact law in `tests/studies/ltmle_repeated_law.py`.

## State: registered, comparator pending

The study declares `pending_reference="lmtp"`. No reference process runs in this change. The
change that registers the R `lmtp` 1.5.4 rows adds them as the TRACKER Study queue item of
plan X32b, PR C. This change writes the pairing columns beside each replicate, so that change
can pair on the same draws:

| column | content |
| --- | --- |
| `ratio__<plan>__<node>` | the per-node ratio at the observed arm, zero where the unit leaves the plan |
| `shift__<plan>__<node>__1` | the arm each unit takes at the node under the plan |
| `fold` | the fold assignment of the fit |

This change builds the pairing columns and checks them in every replicate. It commits none of
them, because 2,400 replicates of 12,000 rows hold about 29 GB, and the container has 15 GB.
The R side of PR C must read samples sharded by scenario and replicate, one file per fit, and
not one CSV of about 12 GB. `pairing_sample` in `canonical_sdr_repeated.py` builds one
replicate's frame. The `shift__` columns are `int8` arm codes.

Pilots P1, P2 and P3 are recorded in `tests/studies/sdr_repeated_properties.py`. The marker
`PILOT_STATE` in `tests/studies/canonical_sdr_repeated.py` reads `"recorded"`. The declared
Python run has finished. Its artifacts are committed beside this file, and the study is
registered with the comparator pending.

## Finding of pilot P2: the in-sample n doubles

Pilot P2 at `n = 8,000` gave these SE ratios:

| arm | cell | SE ratio |
| --- | --- | --- |
| C1 in sample | `rule_t3` | 0.920 |
| C1 in sample | `rule_vs_never_t3` | 0.949 |
| C6 continuous | `rule_t3` | 0.897 |
| C6 continuous | `rule_vs_never_t3` | 0.911 |
| C3 cross-fit, C5 booster | all gated labels | 0.98 to 1.00 |

The cause is a finite-sample shortfall of the in-sample variance estimate. The outcome
regressions are saturated, full-history `CellMeansBackoff` fits. The in-sample residuals of a
cell with `k` units lose variance by the factor `(k - 1) / k`. The rare high-weight cells of the
rule at node 3 dominate the estimate. The deficit decays as `1 / n`:

| n | 2,000 | 4,000 | 8,000 | 16,000 | 32,000 |
| --- | --- | --- | --- | --- | --- |
| in-sample SE times root n, over EIF standard deviation, `rule_t3` | 0.857 | 0.907 | 0.944 | 0.966 | 0.984 |

This is not an estimator defect. `LTMLE` with the same learners gives the same standard error
to 1e-6. A GLM outcome regression gives ratios of 0.98 to 1.00 at every `n`. The declared route
for a finite-sample shortfall is the `n` that a pilot shows removes it, re-run on fresh seeds. A
design change would drop the saturated regression that the in-sample cells need. So
`IN_SAMPLE_N` and `CONTINUOUS_N` each double twice, to 32,000. `CALIBRATION_N` stays at 8,000 for
C3 and C5. The C4 joint band reads the C1 rows, so it moves with `IN_SAMPLE_N`.

The re-pilot at 32,000 ran. These are its SE ratios:

| arm | cell | SE ratio |
| --- | --- | --- |
| C1 in sample | `rule_t3` | 0.9707 |
| C1 in sample | `rule_vs_never_t3` | 0.9796 |
| C6 continuous | `rule_t3` | 0.9691 |
| C6 continuous | `rule_vs_never_t3` | 0.9689 |
| C1 and C6 | all other cells | 0.966 to 1.018 |

The C4 joint coverage is 0.9486. The C6 `n` rule does not fire, so both sizes stay at 32,000.
The calibration rule gives `CALIBRATION_REPLICATES = 38,000` on the pinned values. This is under
the cap of 50,000.

Pilot P3 also shows ratios below 1 under a wrong mechanism: 0.763 for `double_robustness`
`rule_t3__outcome_correct` and 0.772 for `full_history_necessity` `rule_t3__full_history`. The
Wald standard error is inconsistent under a wrong mechanism, so the study reports it with no
calibration claim. Those families gate bias only. The ratios are expected, not a shortfall.

To record the pilots:

1. Run P1, P2 and P3 with the commands below.
2. Run `assemble` for each pilot. It prints the tables and the sizes the rules give.
3. Paste the tables into `sdr_repeated_properties.py` and set the size constants to the printed
   values.
4. Run the declared study (the last section).

## Pilots

The driver is `tests/canonical/sdr_repeated/pilot.py`. A pilot has units. A unit is one fit
set that the driver can run, re-run or override alone. The driver splits the tasks of the
chosen units into chunks. A chunk writes one part file to `--output`. A relaunch skips the
parts that exist, and a part that exists is never replaced.

| pilot | units | fit | fits per unit |
| --- | --- | --- | --- |
| P1 | scenarios `repeated_folds_1` and `repeated_folds_5` | `n = PRIMARY_N` | 2,000 |
| P2 | arms `in_sample`, `crossfit`, `continuous`, `crossfit_boosted` | `IN_SAMPLE_N` for `in_sample`, `CONTINUOUS_N` for `continuous`, `CALIBRATION_N` for the others | 8,000 |
| P3 | the six property families | family `n` | 2,000 |

P2 also yields the C4 joint coverage. The pilots draw from streams that no declared run uses.
The sample size is part of each stream, so a pilot at a new `n` draws fresh seeds.

`plan.json` in each directory records the arguments, the resolved sample size of each unit,
the git HEAD, and the hash of each study file and estimator source file. `assemble` prints a
WARNING and lists the files that differ when a directory's hashes differ from the current tree. A relaunch with other arguments or other
module bytes stops. Use a new `--output` then.

Run the pilots from the repository root. Run P2 per arm if you want to split it.

```powershell
uv run python -m tests.canonical.sdr_repeated.pilot run P1 --output .tmp/sdr-p1 --jobs 8
uv run python -m tests.canonical.sdr_repeated.pilot run P2 --output .tmp/sdr-p2 --jobs 8
uv run python -m tests.canonical.sdr_repeated.pilot run P3 --output .tmp/sdr-p3 --jobs 8
```

Each command prints one progress line for each chunk. Use `--chunk` to change the chunk size.
Use `--replicates` for a smaller run. Use `--units` to run some units only, for example
`--units continuous`. Use `--n N` to set the size of every chosen unit and `--n-for UNIT=N`
to set one unit. `--n-for` wins over `--n`. Both overrides appear in `plan.json`.

To split a pilot between two launches, give one launch `--from 0 --to 4000` and the other
`--from 4000`. `--down-to 4000` runs the chunks from the end down to task 4000, last chunk
first. Two launches over the same output directory then meet in the middle. Task indices
count from zero, unit by unit.

### Assemble

`assemble` reads one or more directories. It prints, for every cell, the draw count, the covered
count, the SE ratio, the standardized bias and the spread of the estimate. It then prints the
tables to paste into `sdr_repeated_properties.py`, and the sizes the rules give.

```powershell
uv run python -m tests.canonical.sdr_repeated.pilot assemble P1 --output .tmp/sdr-p1
uv run python -m tests.canonical.sdr_repeated.pilot assemble P2 --output .tmp/sdr-p2
uv run python -m tests.canonical.sdr_repeated.pilot assemble P3 --output .tmp/sdr-p3
```

`assemble` reads a unit only from a directory whose recorded size equals the size the study
declares now. It checks the `n` column of the rows too. A unit with no such directory stops the
assembly. When two directories supply a unit, the last `--output` wins. `assemble` stops when
a part is missing. It refuses a table that no size up to 50,000 replications can satisfy.

### After an `n` rule fires

The `n` rules change one constant at a time. Each has a doubling count beside its base size:
`PRIMARY_N_DOUBLINGS`, `IN_SAMPLE_N_DOUBLINGS`, `CONTINUOUS_N_DOUBLINGS` and
`ABSORBING_N_DOUBLINGS`.
Take N1 as the example. Its rule doubles `ABSORBING_N` to 16,000.

1. Set `ABSORBING_N_DOUBLINGS = 1` in `sdr_repeated_properties.py`.
2. Run the family again. The default size is the new declared size.
3. Assemble with both directories. The old P3 directory still supplies the other families
   at their sizes, and the new directory supplies N1. The `tau` reading pools all six families.

```powershell
uv run python -m tests.canonical.sdr_repeated.pilot run P3 `
  --units repeated_outcome_necessity --output .tmp/sdr-p3-n1 --jobs 8
uv run python -m tests.canonical.sdr_repeated.pilot assemble P3 `
  --output .tmp/sdr-p3 --output .tmp/sdr-p3-n1
```

The primary rule changes `PRIMARY_N`. The expected number of unseen-cell fits then still holds,
because the count falls as `n` grows. The C6 rule doubles `CONTINUOUS_N` alone. The in-sample
route doubles `IN_SAMPLE_N` alone. The C3 and C5 arms keep `CALIBRATION_N`.

### Re-pilot of P2 at the doubled size

`IN_SAMPLE_N_DOUBLINGS = 2` and `CONTINUOUS_N_DOUBLINGS = 2` set both sizes to 32,000. The
re-pilot ran with the commands below. It wrote the units into a new directory, and `assemble`
read that directory with the old one. The old directory supplied `crossfit` and
`crossfit_boosted` at 8,000. `assemble` skips the `in_sample` and `continuous` rows drawn at
8,000.

```powershell
uv run python -m tests.canonical.sdr_repeated.pilot run P2 `
  --units in_sample continuous --output .tmp/sdr-p2-32k --jobs 8
uv run python -m tests.canonical.sdr_repeated.pilot assemble P2 `
  --output .tmp/sdr-p2 --output .tmp/sdr-p2-32k
```

## What the sampled rows do not show

The pairing plug-in misses the truth by 0.15 EIF-SE for `ey_regimen[rule @ t=2]` and by 0.24
for `t=3` at `n = 12,000`. Both are below the 0.25 margin. No sampled row shows that the
correction is needed for those two cells. Tests E2, E3 and N8' carry that claim exactly. The
evidence page quotes them for the rule's means.

## What the in-sample arms do not test

In sample, a saturated outcome learner reproduces every seen cell. The correction is then
exactly zero on seen cells. The in-sample cells test the plug-in and the mechanism terms. The
`double_robustness` and `correction_necessity` families use a regression on `W` alone, and their
correction is nonzero. Tests E2, E3 and N8' cover the correction exactly.

## Smoke study

Run a disposable smoke study from the repository root. It uses explicit sizes and skips the
property study, so the declared-run guard does not apply. It checks the wiring only.

```powershell
uv run python -m tests.canonical.sdr_repeated.regenerate `
  --replicates 20 --n 1000 --skip-properties --output .tmp/sdr-smoke
uv run python -m tests.canonical.sdr_repeated.pilot run P2 `
  --output .tmp/sdr-smoke-p2 --replicates 4 --n 1000 --chunk 4 --jobs 2
```

## Declared study

After the pilots are recorded, run the declared study and refresh its documentation:

```powershell
uv run python -m tests.canonical.sdr_repeated.regenerate
uv run python -m tests.studies.evidence.document --slug longitudinal-repeated-sdr
```

The manifest records every result-determining module. The evidence page is
`docs/technical-reference/method-evidence/repeated-outcome-sdr.md`, and the validation grid has
a row for the study.
