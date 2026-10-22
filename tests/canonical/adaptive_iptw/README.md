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

The directory holds the driver only. The study is not declared, not run and not registered.
`registered()` does not list it, and the properties module is in `UNCOMMITTED_MODULES` of
`tests/unit/test_method_evidence.py`. The declaration commit and the full run come later.

The design test sets every count in `canonical_adaptive_iptw.py`. It fails when a constant
differs from its computation.

| constant | value | rule |
| --- | --- | --- |
| `R_CAL` | 2,000 | Smallest multiple of 200 at which a calibrated cell passes with probability 0.90 (0.935). |
| `R_BIAS` | 400 | Same rule for a bias row with `b = 0`. |
| `R_RAISED` | `m_rate` 600, `p_rate` 600, `rm` 800 | Smallest multiple of 200 that passes every row of the scene. |
| `R_LADDER` | 1,000 | Every rung and the slope row pass. |
| `R_VL` | 1,200 | The spread interval excludes 1 with probability 0.95 (0.961). |

Two rows of the law M `rate` learner are routed to reporting before the run
(`DESIGN_ROUTED`). Their predicted standardized bias is -0.198 and -0.196. No count of at most
1,600 passes them with probability 0.90.

## Run

Run a disposable smoke of the primary phase:

```bash
python tests/canonical/adaptive_iptw/regenerate.py --primary-only --replicates 20 --jobs 1 --output .tmp/adaptive-iptw-smoke
```

The property phase has no smoke option in the driver. Call
`adaptive_iptw_properties.generate_property_rows(n_jobs=1, budget=20)` instead.

After the declaration commit, regenerate the study and refresh its document:

```bash
python tests/canonical/adaptive_iptw/regenerate.py
python -m tests.studies.evidence.document --slug adaptive-iptw
```
