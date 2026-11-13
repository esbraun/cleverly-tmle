# Binding isotonic projection of a survival curve (`isotonic-binding`)

This study measures `LTMLE(isotonic=True)` with `survival_model="iterated_mean"` on a two-node
survival law with a continuous baseline covariate, where the projection moves the never
regimen's horizon-1 risk on about a third of draws (Fix-8 plan sections 2.7, 3.8 and 5.8). It
reads the interval at the projected level, which keeps the fit's standard error, and the share
of draws on which the level moves. It has no comparator: every cell is read against the
quadrature truth.

| file | role |
| --- | --- |
| `tests/studies/canonical_isotonic_binding.py` | The record and the primary phase. |
| `tests/studies/isotonic_binding_properties.py` | The property families and their rules. |
| `tests/studies/_isotonic_binding_law.py` | The law, the pinned learners, the fit, the truths and the score identity. |
| `tests/studies/isotonic_binding_pilot.py` | The pre-read and the tolerance check, with their committed tables. |
| `tests/unit/test_isotonic_binding_design.py` | The design test: every declared number, recomputed. |
| `tests/unit/test_isotonic_binding_witness.py` | The fast-tier witnesses of the identity and the move, with mutation controls. |

## State of this directory

Declared, not run. No artifact is committed, so the record is not in `registered()`. At the
registration `G31.isotonic` gains the evidence `[curve, "isotonic-binding"]` and the cells
`["isotonic_t2", "projected_t1", "binding_witness", "material_move"]`, and the evidence page
`docs/technical-reference/method-evidence/isotonic-binding.md` lands. A red gated cell becomes a
shortfall of `G31.isotonic` instead (`SupportRow.shortfalls`). A red calibration control is a
defect suspect, and a red calibration positive follows the shared finite-sample reading. A red
witness (`binding_witness` or `material_move`) is a shortfall whose sentence states the
measured share, and the design test checks that the row carries it. The row's limit is to read,
with the measured numbers:

> On the isotonic-binding law (two nodes, a continuous baseline covariate, a strictly
> increasing risk) the projection moved the never regimen's horizon-1 level on {share} of
> draws, by at least 0.1 SE on {share_01} and by at most {max} SE, and the interval at the projected level read
> coverage {cov}. The projected and unprojected intervals disagreed on {disagree} of draws, so
> the projection changed coverage by at most that ({coverage difference and its interval}). On
> the larger-move law (treatment slope 2.0) the moves reached {max_gs2} SE and the coverage
> difference read {difference_gs2}, but there the unprojected interval itself under-covers
> (SE ratio {se_ratio_gs2}, about 0.94 in the pre-read). The moves arise because the two horizons' node-1
> regressions use different learner classes; a flat truth and other learner pairings are
> untested.

The `point-survival-curve` page's limitation row then points to the new page.

The pre-read pools the plan's 2,400 draws (seeds 10000-11199 and 20000-21199, reproduced
exactly by `run --plan-seeds`: coverage 0.9417, SE ratio 0.968) with 4,800 fresh draws. Pooled,
the calibration cell reads coverage 0.9468 and SE ratio 0.984 on 7,200 draws, and the
arm-pooled predictive rule returns 3,600 replications. The level binds on 0.358 of draws and
moves by at least 0.1 SE on 0.191, by at most 0.648 SE. The two intervals disagree on 0.47% of
draws. On the larger-move law (1,600 draws) the projected interval reads 0.935 against 0.933
unprojected, with an SE ratio of 0.937, so that law is reported only.

Five of the 7,200 pre-read draws had one logistic fit that stopped at `max_iter` at
`tol=1e-13`. On the three of them among the fresh draws, refits at `tol=1e-10` converged and
moved every level by the same amount to 1e-9 relative. The 300 draws of the tolerance table
had none. The run publishes each stream's count of such fits as `stalled_fits` on the
`reported_levels/stalled_fits` and `reported_large_move/stalled_fits` rows.

The plan's tolerance rule asks the design test to refit 300 draws. It instead asserts the plan's
four conditions on the committed 300-draw table (`tests/studies/isotonic_binding_tolerance.csv`)
and refits three of its draws at both tolerances: draw 1, whose never level does not move;
draw 0, which moves by 0.095 SE, just under the material threshold; and draw 9, which moves by
0.107 SE, so a refit sets the gated 0.1 SE indicator. That checks the table is current and
keeps about 40 s of fits out of the fast tier.

## Run

Run a disposable smoke of the primary phase:

```bash
python tests/canonical/isotonic_binding/regenerate.py --replicates 3 --output /tmp/isotonic-binding-smoke
```

Run the declared study:

```bash
python tests/canonical/isotonic_binding/regenerate.py --output <empty scratch directory>
```
