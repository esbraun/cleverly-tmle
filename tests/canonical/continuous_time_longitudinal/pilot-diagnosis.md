# Pilot readings and budget of continuous-time-longitudinal

This note holds what the study read before its declaration: the law's feasibility check, the
plug-in separation, the visit cap, the panel's truth, the smoke's seconds and memory per fit, the
budget with its `--jobs` rule, the calibration rules, and the red-cell readings declared before
any run. The pilot's primary draws are pending. A calibration row that the pilot leaves below 0.90
after the extension is recorded here by its `cell/row` name, as a chance reading or, when it
departs beyond chance, with its diagnosis (`regenerate.py`, `pilot_refusals`).

## Law readings

`python -m tests.studies.continuous_time_longitudinal_pilot --jobs 4 --kinds truth`, then
`--kinds truth_capped truth_panel feasibility plugin_limit --resume`, wrote the truth, capped,
panel, feasibility and plug-in rows of `tests/studies/continuous_time_longitudinal_pilot.csv.gz`.
The truths are `TRUTH_RISKS`, `CAPPED_RISKS` and `PANEL_RISKS` in
`tests/studies/continuous_time_longitudinal_law.py`, each on 10,000,000 units per law, regime and
cap.

**Feasibility** (truth-only, plan 5.7). Under the plan's law the 99th percentile of the true
inverse weight among the units that follow a regime to their exit was 41.5 for never and 25.3
for the rule, above the bound of 20. The
pre-declaration adjustment the plan allows lowered the visit intensity's base rate from 0.10 to
0.07 per day and the treatment intercept at a visit from -1.0 to -1.5; every other value is the
plan's. On 1,000,000 units of CT-day:

| regime | cause-1 events by day 10 per 1,000 | cause-2 events by day 10 per 1,000 | weight p99 | weight p99 at risk on day 30 |
| --- | --- | --- | --- | --- |
| never | 60.2 | 36.8 | 19.26 | 26.43 |
| always | 51.8 | 56.0 | 8.81 | 10.57 |
| rule | 67.1 | 56.5 | 17.83 | 23.62 |

The events count among the baseline followers, and the declared percentile reads the followers
to their exit. Each count is at least 30 and each declared percentile at most 20, so
`feasibility_refusals` returns nothing. The last column, among the followers still at risk on
day 30, is a report: the plan's bound does not name the population, and the law stays. The law
gives about 0.065 visits per day of follow-up, about two over 30 days in place of the plan's
three, and 1.3 visits per observed unit on CT-day.

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

## Visit cap

The fit targets $\psi^{(K^*_r)}$, the regime at baseline and at the first $K^*_r$ visits, and the
gates compare with the full risk $\psi_r$ (plan 3.5.3). The engine's own cap rule
(`visit_cap`, with its stratified folds, stopped before any learner) on 200 draws per cell at
n = 1,000 gives:

| cell | regime | $K^* = 3$ | $K^* = 4$ | $K^* = 5$ |
| --- | --- | --- | --- | --- |
| `day_cf` | never | 10 | 182 | 8 |
| `day_cf` | always | 23 | 176 | 1 |
| `day_cf` | rule | 20 | 178 | 2 |
| `cont_cf` | never | 1 | 172 | 27 |
| `cont_cf` | always | 3 | 184 | 13 |

The gap $\psi^{(K)} - \psi_r$ from `CAPPED_RISKS` (each side on 10,000,000 units, Monte Carlo SE
of a gap about 0.0002):

| law | regime | risk | $K = 3$ | $K = 4$ | $K = 5$ |
| --- | --- | --- | --- | --- | --- |
| CT-day | never | cause 1, day 30 | -0.0036 | -0.0013 | -0.0005 |
| CT-day | never | cause 2, day 30 | +0.0009 | +0.0002 | +0.0001 |
| CT-day | always | cause 1, day 30 | +0.0009 | +0.0003 | -0.0002 |
| CT | never | cause 1, day 30 | -0.0045 | -0.0017 | -0.0006 |
| CT | always | cause 1, day 30 | +0.0015 | +0.0003 | +0.0003 |

Under the measured mix of caps the expected gap is -0.0013 (`day_cf`) and -0.0015 (`cont_cf`) at
never's cause-1 day-30 risk, and +0.0017 at the cause-1 day-30 difference; every other row's is
at most 0.0005 in absolute value. Against the true-weight IPW SD of 0.0457 that is 0.03 SD, and
about 0.05 of the TMLE's SD if that SD is near 0.6 of the IPW SD: small, but it stacks on any
targeting bias of the same sign. The pilot records each fit's caps and capped truth, so its own
distribution replaces this table when the primary draws are in. The smoke's share past the cap
reached 0.093 (`day_cf`) and 0.060 (`cont_cf`), above the 0.05 of the W2 warning, while the
realised gap is about 0.0015: the share bounds the gap but is a poor estimate of it.

## The panel's truth

The X70 panel sees a visit inside a node only at the next node start, where its regime resets the
treatment, so it identifies a coarsened risk, not $\psi_r$. `PANEL_RISKS` simulates it on
10,000,000 units per regime. Against $\psi_r$ the gap is -0.0024, -0.0042 and -0.0054 at
never's cause-1 risk on days 10, 20 and 30, +0.0013, +0.0023 and +0.0031 at always's, at most
0.0020 in absolute value for cause 2, and +0.0085 at the cause-1 day-30 difference. The pairing
reads X70 against this truth and starts a diagnosis only when the mean paired difference departs
from that expected gap by more than 0.5 of the continuous engine's SE.

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

## Calibration rules

**The shrunk-SE controls** (review B1). The pilot's refusal reads a shrunk control's power at a
true SE ratio of 1, as plan 5's preface states it: 0.99996 at 2,400 fits, against the floor of
0.999. Read at a pilot's point ratio, the power falls below 0.999 above a ratio near 1.01 (0.9993
at 1.01, 0.993 at 1.02, 0.63 at 1.05), and a 400-draw ratio has a relative SD near 0.035, so that
reading would refuse a calibrated estimator by chance; it is reported here once the draws are in
(`shrunken_power_at_pilot`) and refuses nothing. In the declared run a shrunk control passes when
its SE-ratio interval lies below 0.93. When its calibration row reads conservative, with the
row's SE-ratio interval above 1 (the direction plan 5.7 expects, review S6), the control passes
when its interval lies wholly below the row's own interval, so a conservative row does not void
the run through its control.

**The extension** (review S4). A calibration row's plug-in pass at 2,400 reaches 0.90 only for a
pilot SE ratio inside about [0.985, 1.013]. A calibrated row's 400-draw ratio lands there with
probability 0.31 and an 800-draw ratio with 0.43, so a calibrated row extends with probability
0.69. Under the first declaration's rule, which asked a diagnosis of every row still below 0.90
after the extension, a calibrated row would have needed one with probability about 0.44, and one
of the two rows with about 0.68 (a simulation of the nested 400 and 800 draws). Only a confirmed
departure needs a diagnosis now: the extended pilot's 99% SE-ratio interval excludes 1 or its
coverage interval excludes 0.95 (`calibration_departure`). Any other row below 0.90 is recorded
here as a chance reading, as the `continuous-survival-bootstrap` pilot recorded 11 of its 12 rows.

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
- The visit cap: a red bias at a day-30 row of never, or at a difference against never, is first
  read against the capped truth at the fit's $K^*_r$ (`visit-cap.csv`). A row green against it is
  a cap red, which the share condition of plan 3.5.9 governs, not an engine defect suspect.
- A plug-in control that does not fire is an instrument failure: the run is void and goes back to
  the plan. A shrunk-SE control is read as [the calibration rules](#calibration-rules) state, and
  one that fails its reading is an instrument failure too.
- `outcome_correct`, `both_wrong` and `sparse_calibration` are reported and start no diagnosis.
  The X70 pairing gates nothing; a mean paired difference that departs from its expected value,
  the panel's risk minus $\psi_r$, by more than 0.5 of the continuous engine's SE (`diagnose` in
  `ltmle-pairing.csv`) starts a diagnosis of both engines on the same draws.
- A replication that raises is never redrawn. A `DataError` is a recordable fit failure bounded by
  0.01 per cell. A secondary fit records only an error `recordable` judges a fit failure, with its
  class; any other error stops the run.
- Part (e)'s support constants are read after the run from `floors.csv` and set no verdict.

## Running the pilot

```
python -m tests.studies.continuous_time_longitudinal_pilot --jobs 12 --kinds primary --resume
```

`--resume` keeps the committed truth, capped, panel, feasibility and plug-in rows and starts each cell at its
first missing draw; the file is rewritten after every block of 50 draws. When the primary draws are
committed, `pilot_refusals()` in `regenerate.py` lists what the declaration must still change,
and a declared run refuses until that list is empty.
