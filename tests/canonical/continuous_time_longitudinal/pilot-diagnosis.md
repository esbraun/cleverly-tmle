# Pilot readings and budget of continuous-time-longitudinal

This note holds what the study read before its declaration: the law's feasibility check, the
plug-in separation, the visit cap, the panel's truth, the smoke's seconds and memory per fit, the
budget with its `--jobs` rule, the calibration rules, and the red-cell readings declared before
any run. The first pilot's primary draws failed the family rule and were dropped when the default
treatment model was fixed (the last two sections); the fresh primary pilot is pending. A calibration row that the pilot leaves below 0.90
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
1, on the 4-core cloud machine, rerun on 2026-11-25 after the default treatment model gained its
baseline indicator. Peak memory is the process's resident peak, which includes about 250 MB of
interpreter and imports. The X70 row is the first smoke's, since the change does not reach X70.

| cell | draw 0 (s) | draw 1 (s) | seconds per fit | peak memory |
| --- | --- | --- | --- | --- |
| `day_cf` | 19.6 | 15.1 | 17.4 | 282 MB |
| `cont_cf` | 103.7 | 108.3 | 106.0 | 1,947 MB |
| `mechanism_correct` | 9.6 | 8.2 | 8.9 | 280 MB |
| `outcome_correct` | 13.4 | 9.8 | 11.6 | 280 MB |
| `both_wrong` | 2.4 | 2.5 | 2.4 | 281 MB |
| `ltmle_daily` (X70, 9 nodes) | 53.8 | 48.1 | 51.0 | 252 MB |

Every fit returned. The first smoke, on the pooled treatment model, read 16.6, 103.9, 7.4, 12.1
and 2.7 seconds and a `cont_cf` peak of 2,111 MB; the wider treatment design costs little.
`cont_cf` costs six times `day_cf`: the untied law's grid holds every distinct
event and visit time, and the continuation is held at each visit. In draw 0, X70 withheld the
interval of the four day-30 rows that involve always (`constant_node_plugin`: a node regression of the recursion
read a constant target). The pairing keeps their point estimates, leaves out their SE and
coverage, and counts them as `x70_interval_refused` in `ltmle-pairing.csv`.

## Budget

| item | fits | seconds per fit | CPU-h |
| --- | --- | --- | --- |
| `day_cf` | 2,400 | 17.4 | 11.6 |
| `cont_cf` | 2,400 | 106.0 | 70.7 |
| `mechanism_correct` | 1,600 | 8.9 | 4.0 |
| `outcome_correct` | 800 | 11.6 | 2.6 |
| `both_wrong` | 800 | 2.4 | 0.5 |
| `ltmle_daily` | 400 | 51.0 | 5.7 |
| declared run | | | 95.0 |

The projection is below the plan's hold of 200 CPU-h. The pilot's `budget` repeats it from the
pilot's own seconds once the primary draws are in, and `regenerate.py` refuses a projection above
200. A larger rung costs more: 3,200 primary and 2,400 mechanism fits project to 124 CPU-h, and
4,800 and 3,200 to 181 CPU-h.

The pilot's 400 draws of each of its four cells cost 16.0 CPU-h at these seconds. The first pilot
ran about 1.4 times slower per fit under four-job contention, which puts the fresh pilot near
22.6 CPU-h. If a calibration row needs the 800-draw extension, `cont_cf`'s costs 11.8 CPU-h more
and `day_cf`'s 1.9.

## The `--jobs` rule

A worker keeps its largest fit's memory, so every worker can reach `cont_cf`'s 2.1 GB (the
first smoke's peak; the rerun read 1.9 GB, and the rule keeps the larger). Allow
2.2 GB per worker and keep 6 GB for the parent process, the operating system and the page cache.

| machine | threads | memory | `--jobs` | declared run | pilot |
| --- | --- | --- | --- | --- | --- |
| cloud | 4 cores | 15 GB | 4 | about 24 h | about 4 h |
| local | 16 threads at about 1/3 of a cloud core each | 34 GB | 12 | about 24 h (285 thread-hours) | about 4 h |

On the local machine, (34 - 6) / 2.2 allows 12 workers; 16 would need about 35 GB at the peak.

## Calibration rules

**The shrunk-SE controls** (review B1). The pilot's refusal reads a shrunk control's power at a
true SE ratio of 1, as plan 5's preface states it: 0.99996 at 2,400 fits, against the floor of
0.999. Read at a pilot's point ratio, the power falls below 0.999 above a ratio near 1.01 (0.9993
at 1.01, 0.993 at 1.02, 0.63 at 1.05), and a 400-draw ratio has a relative SD near 0.035, so that
reading would refuse a calibrated estimator by chance; it is reported here once the draws are in
(`shrunken_power_at_pilot`) and refuses nothing. In the declared run a shrunk control passes when
its SE-ratio interval lies below 0.93 or wholly below its calibration row's own interval. The
second reading carries a row that reads conservative, with its SE-ratio interval above 1 (the
direction plan 5.7 expects, review S6), so such a row does not void the run through its control.
Taking either reading also closes the window the re-check found near a ratio of 1.04, where a
heavy-tailed row's interval could leave the band reading failed while its lower bound stays
below 1. On a conservative row the control checks only that the replicates resolve a 15%
shortfall from the row.

**The plug-in control's power** must reach 0.99 at the pilot's plug-in bias and
`mechanism_correct`'s count. At 1,600 fits that needs a bias near 0.32 SD; the separation of
0.5 IPW SDs predicts 0.5 or more, read by the pilot with an SE near 0.05, so the floor refuses by
chance only when the true bias lies within about 0.1 of the threshold. A floor of 0.80 would have
accepted a control that voids the declared run with probability 0.2.

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

## Pilot reading and family failure (2026-11-25)

The pilot's 400 draws of each cell are in, and none of them raised. One `day_cf` draw (31) stopped
targeting without converging, and its estimates are not outliers. `pilot_refusals()` refuses
because no pair of rungs reaches the family-wise pass of 0.95. Of the 60 gated rows, the product
of the per-row pass probabilities read at the pilot's point values is 0.109 at 2,400 primary and
1,600 mechanism fits, 0.479 at 3,200 and 2,400, and 0.808 at 4,800 and 3,200. The plan's remedy
moves rows with coverage below 0.925 to reported, but `binding_rows` is empty: the lowest pilot
coverage is 0.928. The failure is in the SE ratio and the bias gates, not in coverage.

### What binds

Seven rows pass with probability below 0.90 at 2,400 and 1,600:

| row | pass | pilot reading |
| --- | --- | --- |
| `mechanism_correct` always, cause 2 at day 20 | 0.509 | SE ratio 1.147 |
| `mechanism_correct` always vs never, cause 1 at day 30 | 0.655 | SE ratio 1.124, bias 0.120 SD |
| `day_cf` rule, cause 1 at day 20 | 0.800 | bias 0.130 SD |
| `mechanism_correct` always, cause 2 at day 30 | 0.823 | SE ratio 1.130 |
| `cont_cf` never, cause 2 at day 20 | 0.826 | SE ratio 1.142 |
| `day_cf` rule, cause 1 at day 10 | 0.856 | bias 0.126 SD |
| `mechanism_correct` never, cause 1 at day 30 | 0.880 | bias -0.106 SD |

The SE ratios lean above 1 across whole groups. A draw bootstrap with 99% intervals gives pooled
SE ratios of 1.076 (1.041 to 1.123) over the 18 `cont_cf` rows, 1.085 (1.045 to 1.137) over
`mechanism_correct`, 1.072 (1.019 to 1.134) on `day_cf` never levels, 1.065 (1.014 to 1.135) on
always levels and 1.057 (1.007 to 1.122) on the always-vs-never contrasts. The `day_cf` rule levels
read 0.956 (0.914 to 1.007) and rule minus never 1.019. Within a group the rows do not differ
beyond chance (heterogeneity p of 0.37 for `cont_cf` and 0.059 for `mechanism_correct`). So the
conservative SE on static regimes is real, not pilot noise, and it is common to every static row.

### Where the conservative SE comes from

Four candidates were measured on refits of the pilot's own draws (`day_cf` and
`mechanism_correct` 20 draws each, `cont_cf` 6), and each is too small:

- **Projection on the treatment and censoring scores.** With the outcome model wrong and the
  mechanism estimated by maximum likelihood in a correct model, the influence curve the estimator
  actually has is $D^* - \Pi(D^* \mid T_G)$, so the plug-in variance of $D^*$ is conservative
  (Moore and van der Laan 2009, sections 4.2 and 5; Petersen et al. 2014, section 3.7; plan
  3.5.7). The theory applies, but here the effect is small. Regressing each draw's influence curve
  on the unit-level treatment scores and piecewise-Poisson censoring scores gives an adjusted
  $R^2$ between -0.018 and 0.003 in `day_cf` and `mechanism_correct`, at most 0.2% of the SE.
- **Cross-fitting.** Refit without cross-fitting on the same draws, the paired ratio of SEs is
  1.015 in `day_cf` and 1.011 in `cont_cf`, and the estimates correlate at 0.995 and 0.998.
- **Residual score.** The mean of $\sqrt{n}\,P_n D^*/\hat\sigma$ at stop is -0.014 SE in `day_cf` and
  -0.016 in `mechanism_correct`, with a largest absolute value of 0.13. The one-step variance
  matches the targeted one within 0.95 to 1.0.
- **Visit cap.** Read against the capped truth $\psi^{(K)}$, the SE ratios and biases are the same.

The estimator itself is efficient. On one draw of 1,000,000 units the exact efficient influence
function of CT-day comes from a backward recursion over the Markov state (W1, W2, L, A) by day,
with the true treatment and censoring laws; its means reproduce the truths with $|z| \le 1.5$.
The pilot's Monte Carlo SD over this efficiency bound is 1.014 on `day_cf` static rows, 1.021 on
rule rows and 1.009 on `mechanism_correct`. The mean estimated SE over the bound is 1.078, 1.007
and 1.074. The error is in the estimated influence curve, not in the estimator's spread.

On the same draws the engine's influence curve has SD 1.070 times the exact one on `day_cf` static
rows and 1.082 on `mechanism_correct`. The excess is multiplicative, with positive covariance, and
is already present among baseline followers who die before any visit (1.06 for always, 0.92 for
never). Those units' curve depends only on the baseline treatment fit. The engine's default
treatment model is one main-terms logistic fit (and its L1 member) on
[W1, W2, L, previous A, k, visit time], pooled over baseline and every decision visit. The law has
$A(0) \sim \operatorname{expit}(-0.3 + 0.8 L_0 + 0.3 W_1)$ at baseline and
$A(s) \sim \operatorname{expit}(-1.5 + 1.0 L + 2.0 A(s^-) + 0.2 W_2/6)$ at a visit, which no single
main-terms fit in these columns holds. Fit on 200,000 rows the pooled model returns W1 0.164,
W2 0.019, L 0.832, previous A 1.652, k -0.219, visit time -0.016 and intercept -0.504. The ratio
of true to fitted P(A = 1) is 1.04 to 1.16 at baseline, 0.57 to 0.99 for treated units at the first
visit and 1.06 to 1.34 for untreated ones.

The fix was tried in the scratchpad without changing the engine: a learner fitting a separate
logistic model for baseline rows on [W1, W2, L] and another for visit rows on all columns. On the
same draws the influence-curve SD over the exact one goes from 1.070 to 0.989 on `day_cf` static
rows and from 1.082 to 0.998 on `mechanism_correct`. The estimated SE over the bound goes from
1.079 to 0.998 and from about 1.07 to 0.99. On the six `cont_cf` draws, the default-over-stratified
SE ratio is 1.087 (1.03 to 1.13 by row), which brings the pooled 1.076 to about 0.99. Static point
estimates move by 0.001 SD. At $n = 10{,}000$ on one draw the default's ratio to the capped exact
curve stays at 1.085 (1.01 to 1.14), while the stratified learner reads 0.988 (0.93 to 1.03), so the
excess does not shrink with $n$.

### The rule regime

Rule rows escaped the excess by compensation. Under the stratified learner the rule's influence
curve reads 0.962 of the capped exact one at $n = 1{,}000$ (0.92 to 0.98 on day-20 and day-30
levels) and 0.99 to 1.00 at $n = 10{,}000$. That deficit is finite-sample. Under the default
learner the treatment misfit adds back about 7% and the rule reads 0.996.

The rule bias is positive on all six levels: 0.126, 0.130 and 0.113 SD for cause 1 at days 10, 20
and 30, and 0.083, 0.050 and 0.058 for cause 2. Rule minus never reads 0.075 to 0.099 for cause 1
and near 0 for cause 2. The initial estimate is already biased (0.074, 0.080 and 0.114 SD for
cause 1), and targeting does not remove it. The cap, the residual score (0.02 to 0.03 SD) and the
treatment misfit do not explain it, and the stratified learner moves rule levels up by 0.02 to
0.10 SD. Over all 30 `day_cf` rows the bias pools to 0.054 SD (0.020 to 0.089, heterogeneity p of
0.73), with rule levels at 0.093 (0.028 to 0.159) and always levels at 0.080. `cont_cf` has none
(0.000, -0.038 to 0.035). A level bias in CT-day alone points at the tie or day-grid approximations
of that variant rather than at the engine, but the pilot cannot separate them, and one draw at
$n = 10{,}000$ does not settle it. At the pooled 0.093 each rule row passes the bias gate with
probability at least 0.99 at 2,400 draws, so the rule bias is a reported reading, not a demonstrated
defect.

### Family pass under other readings

The precedent in `tests/canonical/continuous_survival_bootstrap/pilot-diagnosis.md` sizes on
anchored readings. Here the anchored current engine reads each row's Monte Carlo SD as its
efficiency bound times its group's pooled SD over bound (static or rule, by cell), so its SE ratio
is the pilot's mean SE over that anchored SD. `cont_cf`, which has no exact curve, takes its pooled
SE ratio. Each row's bias is its group's pooled bias, and coverage is the normal coverage these
imply. The prediction after the fix scales each mean SE by the paired default-over-stratified ratio
on the same draws.

| reading | 2,400 / 1,600 | 3,200 / 2,400 | 4,800 / 3,200 |
| --- | --- | --- | --- |
| raw pilot points | 0.109 | 0.479 | 0.808 |
| raw, `mechanism_correct` SE gate removed | 0.324 | 0.707 | 0.946 |
| raw, `mechanism_correct` reported | 0.502 | 0.795 | 0.975 |
| raw, SE gate lower side only everywhere | 0.400 | 0.761 | 0.954 |
| group-pooled SE, bias and coverage | 0.846 | 0.990 | 1.000 |
| anchored, current engine | 0.938 | 0.997 | 1.000 |
| predicted after the fix, all gated | 0.518 | 0.774 | 0.946 |
| predicted after the fix, coverage below 0.925 reported | 0.867 | 0.987 | 0.999 |

After the fix the binding rows are rule cause 1 at day 30 and rule minus never cause 1 at day 30,
predicted at coverage about 0.918 with SE ratio about 0.89. Those are the rows the plan's coverage
remedy moves. In the anchored current engine the worst rows are `mechanism_correct` always cause 1
at day 30 (0.983) and `day_cf` rule cause 1 at day 30 (0.989). The cost of the declared run is
128.7 CPU-h at 2,400 and 1,600, 168.9 at 3,200 and 2,400 and 248.1 at 4,800 and 3,200, from the
pilot's seconds under four-job contention; the smoke's seconds give 92.6, 121.0 and 176.2.

### Diagnosis

The conservative SE is a defect of the default treatment model, together with the study premise
that the defaults hold the treatment model (`continuous_time_longitudinal_law.py` and plan 5.7).
They do not: the pooled fit cannot hold a baseline mechanism that differs from the visit
mechanism, so every influence curve is weighted by a misfit $\hat g$. As a consequence
`mechanism_correct` does not test double robustness, since its mechanism is not correct. A
one-sided SE gate justified by the projection theory would not be honest here, because the
projection accounts for at most 0.2% of the SE and the excess does not shrink with $n$. Declaring
the current engine on anchored readings would certify an influence curve built on a misfit
mechanism.

The remedy is an engine change before the declaration: fit the default treatment model separately
for baseline and visit rows (or interact a baseline indicator with every column) in both library
members, with a witness that compares the engine's influence curve with an exact one on a law
whose baseline mechanism differs, and the pooled model as its mutation control. Re-pilot `day_cf`,
`cont_cf` and `mechanism_correct` on the fixed engine: 400 draws each cost 20.1 CPU-h at the
pilot's seconds (14.2 at the smoke's). `outcome_correct` costs about 1.5 CPU-h more; its wrong
learner would then be the stratified one without L. Then read the new pilot under the declared
rule. The probe scripts (exact influence curve, capped variant, stratified learner, projection,
paired cross-fit refits) are in the session scratchpad and were not committed.

## Treatment-model fix and pilot reset (2026-11-25)

Each member of the default treatment library (the main-terms logistic model and its three L1
members) now fits on $[x_k, b, b\,x_k]$, with $b$ the baseline indicator
(`by_baseline` in `src/cleverly/survival/_longitudinal_nuisance.py`). At baseline the previous
treatment, the visit count and the visit time are zero, so the unpenalized member equals separate
baseline and visit fits, while the L1 members shrink the baseline toward the visit coefficients.
One design keeps one selection record and one cross-validated choice, which two separate fits
would split. A supplied `treatment_learner` reads $x_k$ as it is. The witness
`test_default_treatment_model_holds_a_baseline_mechanism_of_its_own` fits law PL, whose baseline
and visit mechanisms differ: the stratified fit reaches the exact efficient influence curve to
$10^{-8}$, and the pooled fit misses it by more than $10^{-3}$.

The censoring default needs no such change. Its Cox models read the segment design, and law CT's
censoring hazard $0.008 \exp(0.6 L - 0.4 A)$ is the same before and after a visit and constant in
time, so the main-terms Cox model holds it. The stratified main-terms logistic model holds both
treatment mechanisms: fit on 40,000 units of each law it returns, for CT, visit intercept -1.500,
$L$ 0.985, previous $A$ 1.987 and $W_2$ 0.030 (truth -1.5, 1.0, 2.0 and 0.033), and baseline
intercept -0.299, $L$ 0.794 and $W_1$ 0.259 (truth -0.3, 0.8 and 0.3), each within 2 SE; CT-day
reads the same. So `mechanism_correct` now holds the correct mechanism, and `outcome_correct`'s
wrong treatment model is the stratified main-terms model without $L$.

On the pilot's first 20 draws of `day_cf` and `mechanism_correct`, refitted with the fixed engine,
the influence curve's SD over the exact one is 0.996 on `day_cf` static rows (it was 1.070), 0.961
on rule rows (0.990) and 1.005 on `mechanism_correct` (1.082). The mean estimated SE over the
efficiency bound is 1.004, 0.974 and 0.998 (it was 1.079, 1.004 and 1.074). Against the first
pilot's Monte Carlo SD over the bound (1.014, 1.021 and 1.009), that is an SE ratio near 0.99 on
static rows and 0.95 on rule rows, the finite-sample deficit of the rule regime recorded above.
The estimates sit within 0.024 SD of the scratch stratified learner's.

The plug-in limit does not depend on the treatment model: refitting a 3,000-unit draw under
`mechanism_correct` with the pooled and the stratified default gives identical untargeted estimates
(difference 0.0), since under cross-fitting the untargeted pass reads only the hazards, the visit
intensity and the continuation, and the limit is fitted with two folds. The truth, capped, panel,
feasibility and plug-in limit rows stand, and the primary rows were dropped from
`tests/studies/continuous_time_longitudinal_pilot.csv.gz`. `pilot_refusals()` now refuses with "the
pilot holds no primary draws" until the fresh pilot is committed. The command under Running the
pilot reruns all four cells, at about 16 CPU-h at the smoke's seconds.
