# Pilot readings and budget of continuous-time-longitudinal

This note holds what the study read before its declaration: the law's feasibility check, the
plug-in separation, the smoke's seconds and memory per fit, the budget with its `--jobs` rule,
and the red-cell readings declared before any run. The pilot's primary draws are pending; a
calibration row that the pilot leaves below 0.90 is diagnosed here by its `cell/row` name before
the declaration (`regenerate.py`, `pilot_refusals`).

## Law readings

`python -m tests.studies.continuous_time_longitudinal_pilot --jobs 4 --kinds truth`, then
`--kinds feasibility plugin_limit --resume`, wrote the truth, feasibility and plug-in rows of
`tests/studies/continuous_time_longitudinal_pilot.csv.gz`. The truths are `TRUTH_RISKS` in
`tests/studies/continuous_time_longitudinal_law.py`, each on 10,000,000 units per law and regime.

**Feasibility** (truth-only, plan 5.7). Under the plan's law the 99th weight percentile of the
baseline followers was 41.5 for never and 25.3 for the rule, above the bound of 20. The
pre-declaration adjustment the plan allows lowered the visit intensity's base rate from 0.10 to
0.07 per day and the treatment intercept at a visit from -1.0 to -1.5; every other value is the
plan's. On 1,000,000 units of CT-day:

| regime | cause-1 events by day 10 per 1,000 | cause-2 events by day 10 per 1,000 | weight p99 |
| --- | --- | --- | --- |
| never | 60.2 | 36.8 | 19.26 |
| always | 51.8 | 56.0 | 8.81 |
| rule | 67.1 | 56.5 | 17.83 |

Each count is at least 30 and each percentile at most 20, so `feasibility_refusals` returns
nothing.

**Plug-in separation** (review S8). One draw of 200,000 CT-day units, fitted once with
`mechanism_correct`'s outcome side (2 outer folds, 416 s, 5.9 GB), gives the untargeted plug-in
limit. Its distance from the truth in true-weight IPW SDs at n = 1,000 on the same draw:

| day-30 risk | limit | truth | IPW SD | separation |
| --- | --- | --- | --- | --- |
| never, cause 1 | 0.3442 | 0.3707 | 0.0457 | 0.58 |
| never, cause 2 | 0.1885 | 0.1966 | 0.0328 | 0.25 |
| always, cause 1 | 0.2196 | 0.2077 | 0.0225 | 0.53 |
| always, cause 2 | 0.2592 | 0.2503 | 0.0253 | 0.35 |

Two day-30 risks sit at least 0.5 SD from their truths, so the plan's coefficients of $L$ in the
cause hazards and the $L$ transition stand unchanged. A first reading on 20,000 units gave at most
0.46 at day 30; the plan's reading is the one on 200,000 units.

## Smoke

Two draws of each cell at n = 1,000, each cell in a fresh process with every thread variable at
1, on the 4-core cloud machine (2026-11-25). Peak memory is the process's resident peak, which
includes about 250 MB of interpreter and imports.

| cell | draw 0 (s) | draw 1 (s) | seconds per fit | peak memory |
| --- | --- | --- | --- | --- |
| `day_cf` | 18.1 | 15.1 | 16.6 | 278 MB |
| `cont_cf` | 98.5 | 109.3 | 103.9 | 2,111 MB |
| `mechanism_correct` | 7.6 | 7.3 | 7.4 | 274 MB |
| `outcome_correct` | 13.1 | 11.0 | 12.1 | 275 MB |
| `both_wrong` | 2.6 | 2.8 | 2.7 | 276 MB |
| `ltmle_daily` (X70, 9 nodes) | 53.8 | 48.1 | 51.0 | 252 MB |

Every fit returned. `cont_cf` costs six times `day_cf`: the untied law's grid holds every distinct
event and visit time, and the continuation is held at each visit. In draw 0, X70 withheld the
interval of the four day-30 rows that involve always (`constant_node_plugin`: a node regression of the recursion
read a constant target). The pairing keeps their point estimates, leaves out their SE and
coverage, and counts them as `x70_interval_refused` in `ltmle-pairing.csv`.

## Budget

| item | fits | seconds per fit | CPU-h |
| --- | --- | --- | --- |
| `day_cf` | 2,400 | 16.6 | 11.1 |
| `cont_cf` | 2,400 | 103.9 | 69.3 |
| `mechanism_correct` | 1,600 | 7.4 | 3.3 |
| `outcome_correct` | 800 | 12.1 | 2.7 |
| `both_wrong` | 800 | 2.7 | 0.6 |
| `ltmle_daily` | 400 | 51.0 | 5.7 |
| declared run | | | 92.6 |

The projection is below the plan's hold of 200 CPU-h. The pilot's `budget` repeats it from the
pilot's own seconds once the primary draws are in, and `regenerate.py` refuses a projection above
200. A larger rung costs more: 3,200 primary and 2,400 mechanism fits project to 121 CPU-h, and
4,800 and 3,200 to 176 CPU-h.

The pilot's 400 draws of each of its four cells cost 15.6 CPU-h. If a calibration row needs the
800-draw extension, `cont_cf`'s costs 11.5 CPU-h more and `day_cf`'s 1.8.

## The `--jobs` rule

A worker keeps its largest fit's memory, so every worker can reach `cont_cf`'s 2.1 GB. Allow
2.2 GB per worker and keep 6 GB for the parent process, the operating system and the page cache.

| machine | threads | memory | `--jobs` | declared run | pilot |
| --- | --- | --- | --- | --- | --- |
| cloud | 4 cores | 15 GB | 4 | about 23 h | about 4 h |
| local | 16 threads at about 1/3 of a cloud core each | 34 GB | 12 | about 23 h (280 thread-hours) | about 4 h |

On the local machine, (34 - 6) / 2.2 allows 12 workers; 16 would need about 35 GB at the peak.

## Red-cell readings

The readings are declared in `regenerate.py`'s docstring before any run, and this section repeats
them. Every primary row of `day_cf` and `cont_cf`, every row of `mechanism_correct`, the plug-in
control, both calibration rows and both shrunk-SE controls are gates, and any red verdict fails
the run. No cell is re-sized, re-drawn or downgraded after a verdict is seen, and no law changes to
make a row pass.

- A red calibration row is a finite-sample red only when the 99% interval of its SE ratio
  contains 1, its coverage is consistent with 0.95 and the diagnosis finds no defect; it then goes
  to reporting with an owner. Any other red row is a defect suspect.
- Declared exception (review S6): with the treatment model right and the continuation approximated,
  the influence-curve variance is expected to be conservative, so a calibration row red only
  because its SE-ratio interval lies above 1.07, with its coverage interval at or above 0.95 and no
  defect found, reads conservative and goes to reporting with an owner. A red on the low side
  stays a defect suspect.
- Where to look first: red in `cont_cf` only points at the untied path (the reduced grid, the
  continuation held at many visit times); red in `day_cf` only at the tie rule; red at day 30 in
  both with day 10 green at the visit or covariate term. Red rule rows alone point at the rule's
  slot, its evaluation or the decision visits; red in `mechanism_correct` at targeting and the slot
  weight.
- A red rule row keeps only rules experimental; static regimes are labelled from their own rows.
- A plug-in control that does not fire, or a shrunk-SE control that stays in the band, is an
  instrument failure: the run is void and goes back to the plan.
- `outcome_correct`, `both_wrong` and `sparse_calibration` are reported and start no diagnosis.
  The X70 pairing gates nothing; a mean paired difference above 0.5 of the continuous engine's SE
  (`diagnose` in `ltmle-pairing.csv`) starts a diagnosis of both engines on the same draws.
- A replication that raises is never redrawn. A `DataError` is a recordable fit failure bounded by
  0.01 per cell.

## Running the pilot

```
python -m tests.studies.continuous_time_longitudinal_pilot --jobs 12 --kinds primary --resume
```

`--resume` keeps the committed truth, feasibility and plug-in rows and starts each cell at its
first missing draw; the file is rewritten after every block of 50 draws. When the primary draws are
committed, `pilot_refusals()` in `regenerate.py` lists what the declaration must still change,
and a declared run refuses until that list is empty.
