# AdaptiveIPTW gate evidence (`adaptive-iptw`)

This study fits `AdaptiveIPTW` on exact laws. It tests the targeted and one-step forms, the
mechanism learners that the theory allows, and the cases where the theory predicts a bias. It is
Python only. The R `drtmle` comparator is declared and pending (`STUDY.pending_reference`). The
pairing study `adaptive-iptw-pairing` runs it later.

| file | role |
| --- | --- |
| `tests/studies/canonical_adaptive_iptw.py` | The record, the fits, the scenes, the readers, the truths and the primary phase. |
| `tests/studies/adaptive_iptw_properties.py` | The property families and their rules. |
| `tests/studies/adaptive_iptw_laws.py` | The laws P, PD, VL1 and VL0, their exact truths and the learners. |
| `tests/unit/test_adaptive_iptw_study_design.py` | The design test: items 1 to 8 of the plan. |
| `tests/unit/_adaptive_iptw_study_support.py` | The population engine that the design test reads. |

## State of this directory

The directory holds the driver and the committed artifacts of the declared run. The run is the
repeat after the fold-wise reduced regression replaced the pooled one. The study is registered
in `tests/studies/evidence/registry.py`, with the `reporting` publication policy and the
`pending_reference` `drtmle`. The evidence page is
`docs/technical-reference/method-evidence/adaptive-iptw.md`. The red cells have owners in
`tests/studies/evidence/red_cells.py`, and the support rows `AI.*` in
`src/cleverly/_support_rows.py` cite the cells that passed. The diagnostic of the one red
composite cell is `tests/diagnostics/adaptive_iptw_composite_l3/`.

The design test sets every count in `canonical_adaptive_iptw.py`. It fails when a constant
differs from its computation.

| constant | value | rule |
| --- | --- | --- |
| `R_CAL` | 2,000 | Smallest multiple of 200 at which a calibrated cell passes with probability 0.90 (0.935). |
| `R_BIAS` | 400 | Same rule for a bias row with `b = 0`. |
| `R_RAISED` | `p_rate` 600, `rm` 800 | Smallest multiple of 200 that passes every row of the scene. |
| `R_LADDER` | 1,000 | Every rung and the slope row pass. |
| `R_VL` | 1,200 | The spread interval excludes 1 with probability 0.95 (0.961). |

**Role rule** (item 2, rev 3.9). A row with `|b| <= 0.05` is positive. A row with `|b| >= 0.5` is
a control. A row between the two stays positive only when some declared `R` of at most 1,600
passes it with probability 0.90 and `|b(4n)| < |b(n)|`. Any other middle-band row goes to
reporting before the run (`DESIGN_ROUTED`). The set is empty now. The law P `rate` rows are in
the middle band, and their bias falls with `n`.

**Law M `rate` rows.** Law M is not H1. A shift of the logits along a covariate breaks the level
sets of each `g_a`, so the validity term is first order and `b` grows with `n`. The review of
rev 3.8 found this (`b` of `ey[low]`: -0.198 at n = 2,000 and -0.375 at 32,000). The learner
now shrinks each arm probability to uniform by `2 n^(-1/3)`. Each shrunk probability is a
function of the true one, so the level sets survive and `b` is below 1e-3 at every `n`. The
VL1 `level_set_breaking` control measures the failure itself.

**Expected red rows.** A correct run shows about 3.4 red rows by chance: 2.0 calibration rows and
1.5 bias, ladder and necessity rows (`EXPECTED_CHANCE_REDS`). The probability that all 124
verdicts are green is about 0.03. `RED_READING` in the runner module holds the reading of a
red row, and `regenerate.py` repeats it.

**Calibration sensitivity.** The sizing assumes a true SE ratio of 1.0. At `R_CAL = 2,000` a
row passes when the estimated ratio lies in about [0.971, 1.029]. A true ratio of 0.99 passes
with probability 0.88, and 0.98 with probability 0.72. A red calibration row with a ratio 1% to
2% from 1 is the expected outcome and reads by the calibration reading. `R_CAL` does not change.

**Duplicate verdicts.** A `targeting_necessity` targeted row reads the same estimates as the
`bias` row of the same key, so the evidence page counts that verdict once.

Two limits of the instruments. The necessity variance bound assumes independent arms. The
exact paired variance is smaller (`rm_d`: probability 0.999 instead of 0.961). The pairing
study declares the R comparator. Until then `pending_reason` records that the declared
reference sources are a placeholder.

## Run

Run a disposable smoke of the primary and property phases into a scratch directory:

```bash
python tests/canonical/adaptive_iptw/regenerate.py --replicates 20 --output /tmp/adaptive-iptw-smoke
```

A `--replicates` count other than 2,000 runs the primary phase only. For the property phase,
call `adaptive_iptw_properties.generate_property_rows(n_jobs=1, budget=20)`.

Run the declared study from a clean, pushed tree on one thread per library:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python tests/canonical/adaptive_iptw/regenerate.py --output /tmp/adaptive-iptw-run-1 --jobs 4
```

Resume a stopped run at the same commit with a new empty `--output`. The checkpoints sit in
`~/.cache/cleverly/runs/adaptive-iptw/<key>`. A container that loses `~/.cache` restarts the
run, which costs about 4 CPU-hours. After the run, refresh the document:

```bash
python -m tests.studies.evidence.document --slug adaptive-iptw
```
