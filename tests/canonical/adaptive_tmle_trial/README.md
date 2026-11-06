# adaptive-tmle-trial (X90 T3, declared, not yet run)

The gated study of adaptive TMLE for a randomized trial pooled with external data
(`TrialWithExternalData`, `TrialATE`, `AdaptiveTMLEMethod`). The module docstring of
`tests/studies/adaptive_tmle_trial.py` is the declaration: scenarios, estimands, controls and
the reading of each. `tests/unit/test_adaptive_tmle_trial_design.py` recomputes every declared
constant (truths, separations, control powers, sizing) without a run.

## Declared run

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m tests.canonical.adaptive_tmle_trial.regenerate --output /tmp/atmle-trial-run --jobs 4
```

A smoke run passes `--replicates K` (any K other than the declared budget) and writes nothing
inside the repository.

## Gates the driver reads

* The primary performance tests (`performance-tests.csv`): bias leg, coverage floor, SE sanity
  band for each scenario and its gated estimand. `b1y` is a reported stress row
  (`reported_scenarios`): its three legs are published with `reported = True` and pass by
  declaration. Only the failure bound still gates it.
* `controls.csv` through `scientific_failures`: the efficiency contrast (upper bound of
  RMSE(A-TMLE) / RMSE(trial-only TMLE) below 1 on `a0`; reported on `a1a`, `a1b`), K1, K2, K3,
  K5. The K4 ladder is a reporting row.
* `reported.csv`: the coverage of all three targets per scenario, and the projection bias in
  SDs, as reported rows.

The R `atmle` comparator is declared in `COMPARATOR` and runs as a separate pairing record.

## Sizing and cost

`PRIMARY_REPLICATES` is `repeated_sizing.primary_replicates` at the pilot that
`tests/unit/test_adaptive_tmle_trial_design.py` pins (each cell at the one-sided 80% lower bound
of its pilot coverage). The pinned pilot is commit `685171b2`, run on the 16-core laptop on
2026-11-06: 800 replicates of each scenario on `PILOT_SEEDS`, 0 lost replicates, every control
passed (efficiency contrast on `a0`, RMSE ratio 0.703 [0.654, 0.754]; `k3_targeted` 0.941). The
covered counts of the 16 gated scenarios give `PRIMARY_REPLICATES = 2,800`.

`b1y` is a reported stress row and is not in the sizing. Its pilot count was 708 of 800, and the
rule finds no budget up to its cap of 50,000 that meets a gated cell at that coverage, which is
the declared reason it is not gated (see the module docstring and the Evidence section of
`docs/technical-reference/adaptive-tmle.md`). Its Psi coverage, projection coverage, bias and SE
ratio are published and read by no verdict.

The pilot costs 143.24 core-seconds per replicate set (all 17 scenarios). The declared run costs
2,800 x 143.24 s, about 111 core-hours: about 28 wall-hours on 4 workers, so it goes to the local
queue (16 cores, about 7 wall-hours) as the 24-hour rule requires. The per-scenario seconds are
in the pilot's `cells[*].seconds`.

## Design review readings (scratch draws, outside the repository)

* Efficiency. With `A` forced into the bias model (the T3 engine, atmle's `force_A`), the
  external arms get free mean shifts and the exact-law SD ratio on `a0` is 1.00, which the T3
  pilot's RMSE ratios (1.00, 0.99, 1.07) matched. With `A` penalized, the ratio is 0.48 at the
  oracle selection, and 200 replicates read RMSE ratio 0.56 on `a0`, 1.05 on `a1a` and 1.14 on
  `a1b` (forced: 1.03 and 1.11 on the same draws). So `a0` gates and `a1a`, `a1b` report.
* `a5`. The T3 SE ratio of 0.22 came from the nuisance learners, not the refit: in fold fits the
  spline observation model predicted 0.010 to 0.011 for trial controls whose true observation
  probability is above 0.3, and the four-cell spline regression of `Q(S, W, A)` mispredicted
  the trial control at the smallest `W1` by 14 outcome SDs. The inverse observation weight
  carried both into the bias pseudo-outcome, the inner CV then chose the smallest penalty
  (all 44 bias columns), and fold estimates reached 32. The selected sets were well conditioned
  (condition number at most 2e2). With the correct, linear-within-cell learners, 200
  replicates read SE ratio 0.95, coverage 0.965, estimates 1.28 to 1.76.
* `h2_trial`: 240 replicates read standardized bias +0.075 and coverage 0.933.
