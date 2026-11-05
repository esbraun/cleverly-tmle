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
  band for each scenario and its gated estimand.
* `controls.csv` through `scientific_failures`: the efficiency contrast (upper bound of
  RMSE(A-TMLE) / RMSE(trial-only TMLE) below 1 on `a0`, `a1a`, `a1b`), K1, K2, K3, K5. The K4
  ladder is a reporting row.
* `reported.csv`: the coverage of all three targets per scenario, and the projection bias in
  SDs, as reported rows.

The R `atmle` comparator is declared in `COMPARATOR` and runs as a separate pairing record.

## Cost (1 core, pilot means)

3,200 replicates per scenario (the design test records the sizing). 98.8 core-seconds per
replicate set, 87.8 core-hours in all, about 22 wall-hours on 4 workers.

| cell | n | reps | s/rep | core-hours |
| --- | --- | --- | --- | --- |
| a0 | 4000 | 3200 | 6.8 | 6.1 |
| a1a | 4000 | 3200 | 8.7 | 7.8 |
| a1b | 4000 | 3200 | 9.0 | 8.0 |
| a2c | 4000 | 3200 | 6.1 | 5.4 |
| a2d | 4000 | 3200 | 6.8 | 6.0 |
| a3 | 4000 | 3200 | 5.2 | 4.6 |
| a3_pi | 4000 | 3200 | 9.4 | 8.4 |
| a4 | 4000 | 3200 | 5.4 | 4.8 |
| a5 | 4000 | 3200 | 9.7 | 8.6 |
| b1y | 4000 | 3200 | 3.7 | 3.3 |
| e1 | 4000 | 3200 | 6.4 | 5.6 |
| e1_k2_n1000 | 1000 | 3200 | 0.5 | 0.4 |
| e1_k2_n2000 | 2000 | 3200 | 0.6 | 0.6 |
| f1a | 4000 | 3200 | 2.1 | 1.9 |
| f1b | 4000 | 3200 | 2.4 | 2.1 |
| h2_pooled | 4000 | 3200 | 8.1 | 7.2 |
| h2_trial | 4000 | 3200 | 7.9 | 7.1 |

## Pilot reading (400 replicates per gated cell, outside the repository)

Coverage 0.92 to 0.96 and SE ratio 0.91 to 1.03 on every gated cell except `a5`, whose SE ratio
is 0.22 (a few replicates select an unstable working model, estimates from -2.3 to 4.2 with
large SEs). The efficiency contrast reads RMSE ratios 1.00, 0.99 and 1.07 on `a0`, `a1a`, `a1b`,
so its gate is expected to fail.
