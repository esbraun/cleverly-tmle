# Canonical evidence for repeated-outcome longitudinal IPSI

This directory holds the study of incremental propensity score interventions (IPSI) on a
repeated outcome. `cleverly` fits each panel once with `outcome=Repeated(["Y1", "Y2", "Y3"])`.
Each horizon `h` is the end-of-study estimate for the panel cut after node `h`
(Kim, Kennedy and Naimi 2021, Theorems 3.1, 4.1 and 5.1).

The study has no comparator. No R package implements IPSI for a repeated outcome. The truths
come from the exact law in `tests/studies/ltmle_repeated_law.py`. The study declares no
`reference` and no `pending_reference`.

## State: placeholder pilots

The pilot tables in `tests/studies/ipsi_repeated_properties.py` are placeholders. The marker
`PILOT_STATE` in `tests/studies/canonical_ipsi_repeated.py` reads `"placeholder"`. The declared
sizes follow the rules in `tests/studies/repeated_sizing.py` applied to those placeholders, so
they are not final. The declared run refuses to start until the pilots are recorded.

To record the pilots:

1. Run P4, P5 and P6 with the commands below.
2. Run `assemble` for each pilot. It prints the tables and the sizes the rules give.
3. Paste the tables into `ipsi_repeated_properties.py` and set the size constants to the printed
   values. Set `PILOT_STATE` to `"recorded"`.
4. Run the declared study (the last section).

## Pilots

The driver is `tests/canonical/ipsi_repeated/pilot.py`. A pilot has units. A unit is one fit
set that the driver can run, re-run or override alone. The driver splits the tasks of the
chosen units into chunks. A chunk writes one part file to `--output`. A relaunch skips the
parts that exist, and a part that exists is never replaced.

| pilot | units | fit | fits per unit |
| --- | --- | --- | --- |
| P4 | scenarios `repeated_folds_5` and `repeated_folds_1` | `n = PRIMARY_N` | 2,000 |
| P5 | arms `in_sample`, `crossfit`, `continuous` | `CALIBRATION_N`, `CONTINUOUS_N` for `continuous` | 8,000 |
| P6 | `single_robustness`, `repeated_outcome_necessity`, `outcome_history_necessity` | family `n` | 2,000 |

P5 also yields the C4 joint coverage and the C6 fallback mass. P4 and P5 draw from streams that
no declared run uses. The sample size is part of each stream, so a pilot at a new `n` draws
fresh seeds.

`plan.json` in each directory records the arguments, the resolved sample size of each unit,
the git HEAD, and the hash of each study module. A relaunch with other arguments or other
module bytes stops. Use a new `--output` then.

Run the pilots from the repository root. Run P5 per arm if you want to split it.

```powershell
uv run python -m tests.canonical.ipsi_repeated.pilot run P4 --output .tmp/ipsi-p4 --jobs 8
uv run python -m tests.canonical.ipsi_repeated.pilot run P5 --output .tmp/ipsi-p5 --jobs 8
uv run python -m tests.canonical.ipsi_repeated.pilot run P6 --output .tmp/ipsi-p6 --jobs 8
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
tables to paste into `ipsi_repeated_properties.py`, and the sizes the rules give.

```powershell
uv run python -m tests.canonical.ipsi_repeated.pilot assemble P4 --output .tmp/ipsi-p4
uv run python -m tests.canonical.ipsi_repeated.pilot assemble P5 --output .tmp/ipsi-p5
uv run python -m tests.canonical.ipsi_repeated.pilot assemble P6 --output .tmp/ipsi-p6
```

`assemble` reads a unit only from a directory whose recorded size equals the size the study
declares now. It checks the `n` column of the rows too. A unit with no such directory stops the
assembly. When two directories supply a unit, the last `--output` wins. `assemble` stops when
a part is missing. It refuses a table that no size up to 50,000 replications can satisfy.

### After an `n` rule fires

The `n` rules change one constant at a time: `PRIMARY_N`, `CALIBRATION_N`, `CONTINUOUS_N`,
`ABSORBING_N` or `HISTORY_FREE_N`. Take N2 as the example. Its rule doubles `HISTORY_FREE_N`
to 16,000.

1. Set `HISTORY_FREE_N = 16_000` in `ipsi_repeated_properties.py`.
2. Run the family again. The default size is the new declared size.
3. Assemble with both directories. The old P6 directory still supplies the other two families
   at their sizes, and the new directory supplies N2. The `tau` reading pools all three families.

```powershell
uv run python -m tests.canonical.ipsi_repeated.pilot run P6 `
  --units outcome_history_necessity --output .tmp/ipsi-p6-n2 --jobs 8
uv run python -m tests.canonical.ipsi_repeated.pilot assemble P6 `
  --output .tmp/ipsi-p6 --output .tmp/ipsi-p6-n2
```

The C6 rule doubles `CONTINUOUS_N` alone, and the C1 and C3 arms keep `CALIBRATION_N`. Run
`--units continuous` in the same way.

### Directories from the first driver

The first version of the driver wrote `"n": null` in `plan.json` and no `n` in the P4 rows.
Run `stamp` once on such a directory. `stamp` refuses a plan that records an `--n` override. It
records the sizes that the study declares at the commit you run it from. Run it from the commit
that wrote the directory, or from one that declares the same sizes.

```powershell
uv run python -m tests.canonical.ipsi_repeated.pilot stamp P4 --output .tmp/ipsi-p4
uv run python -m tests.canonical.ipsi_repeated.pilot assemble P4 --output .tmp/ipsi-p4
```

## What the in-sample arms do not test

In sample, the outcome learner reproduces every seen cell. The outcome-residual correction is
then exactly zero on seen cells. The in-sample cells test the plug-in and the mechanism terms.
Only the cross-fit arms and `single_robustness` test the residual term with a nonzero residual.
Tests E5, E6, N9 and W1 cover that term exactly.

## Smoke study

Run a disposable smoke study from the repository root. It uses explicit sizes and skips the
property study, so the placeholder refusal does not apply. It checks the wiring only.

```powershell
uv run python -m tests.canonical.ipsi_repeated.regenerate `
  --replicates 20 --n 1000 --skip-properties --output .tmp/ipsi-smoke
uv run python -m tests.canonical.ipsi_repeated.pilot run P5 `
  --output .tmp/ipsi-smoke-p5 --replicates 4 --n 1000 --chunk 4 --jobs 2
```

## Declared study

After the pilots are recorded, run the declared study and refresh its documentation:

```powershell
uv run python -m tests.canonical.ipsi_repeated.regenerate
uv run python -m tests.studies.evidence.document --slug longitudinal-repeated-ipsi
```

The manifest records every result-determining module. No grid row or evidence page exists for
this study yet. A later change adds them with the artifacts.
