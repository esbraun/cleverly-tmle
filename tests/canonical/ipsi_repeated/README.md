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

The driver is `tests/canonical/ipsi_repeated/pilot.py`. A pilot splits its tasks into chunks.
A chunk writes one part file to `--output`. A relaunch skips the parts that exist. The file
`plan.json` in the output directory records the arguments. A relaunch with other arguments stops.

| pilot | tasks | fit | fits per cell |
| --- | --- | --- | --- |
| P4 | primary, scenarios `repeated_folds_5` and `repeated_folds_1` | `n = 8,000` | 2,000 |
| P5 | calibration arms `in_sample`, `crossfit`, `continuous` | `n = 8,000` | 8,000 |
| P6 | `single_robustness`, `repeated_outcome_necessity`, `outcome_history_necessity` | family `n` | 2,000 |

P5 also yields the C4 joint coverage and the C6 fallback mass. P4 and P5 draw from streams that
no declared run uses.

Run the pilots from the repository root:

```powershell
uv run python -m tests.canonical.ipsi_repeated.pilot run P4 --output .tmp/ipsi-p4 --jobs 8
uv run python -m tests.canonical.ipsi_repeated.pilot run P5 --output .tmp/ipsi-p5 --jobs 8
uv run python -m tests.canonical.ipsi_repeated.pilot run P6 --output .tmp/ipsi-p6 --jobs 8
```

Each command prints one progress line for each chunk. Use `--chunk` to change the chunk size.
Use `--replicates` and `--n` for a smaller run. Both arguments are part of `plan.json`.

To split a pilot between two machines, give one machine `--from 0 --to 4000` and the other
`--from 4000`. `--down-to 4000` runs the chunks from the end down to task 4000, last chunk first.
Two launches over the same output directory then meet in the middle, and neither repeats a part.
Task indices count from zero. Scenario, arm or family blocks follow each other in task order.

Summarize a finished pilot:

```powershell
uv run python -m tests.canonical.ipsi_repeated.pilot assemble P4 --output .tmp/ipsi-p4
uv run python -m tests.canonical.ipsi_repeated.pilot assemble P5 --output .tmp/ipsi-p5
uv run python -m tests.canonical.ipsi_repeated.pilot assemble P6 --output .tmp/ipsi-p6
```

`assemble` stops when a part is missing. It refuses a table that no size up to 50,000
replications can satisfy.

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
