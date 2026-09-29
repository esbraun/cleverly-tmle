# RM18 design CD: the density attribution of the shift row

This directory holds the design that RM18 of `docs/roadmap.md` declares in "Design CD", for the
red paired row `ate_shift[+0.25 vs natural course]` of `shift-policies`. It draws no new sample,
runs no R, and changes no study, verdict or committed row.

| file | what it holds |
| --- | --- |
| `run.py` | the refit of the 800 registered draws, the analytic retarget, and the reading |
| `cd-validation.csv` | the R4 checks: the Cb refit rows, and the (Cb, L) excess bound and resolution |
| `cd-rows.csv.gz` | the read estimand's rows of the three arms: Cb (`cleverly`), Ca (`cleverly-analytic-density`) and L (`lmtp`, committed) |
| `cd-reading.csv` | `sigma*`, each arm's ratios, the three `D` differences, the three framework excess bounds and resolutions, and the reading |
| `run.log` | the run record |

`tests/unit/test_rm18_comparator_density_diagnostic.py` runs in the fast tier. It checks `sigma*`
against a Gauss-Hermite rule to 1e-6, and the analytic ratio against the normal tilt with a
wrong-direction mutation. It checks the (Cb, L) excess against `equivalence.csv`, one refit, and
the reading rule with a mutation at each clause. A synthetic three-arm table shows that the
reading reads the (Ca, L) bound and that each mutation moves it. The test also checks that the
committed rows meet the declared budget, rebuilds `cd-reading.csv`, and retargets one committed
Ca row. A missing committed file fails the test.

The design ran once. The roadmap at commit `985849c6` gives the reading in
"[What design CD found](https://github.com/esbraun/cleverly-tmle/blob/985849c668a18cf800094714ddae3fea0675cb91/docs/roadmap.md#what-design-cd-found)".

## Run

```bash
python -m tests.diagnostics.rm18_comparator_density.run --part CD --output <scratch>
```

The interpreter, `PYTHONPATH`, thread limits, options, refusals and failure behaviour are those
of [`../rm18_fixed_weights/`](../rm18_fixed_weights/README.md).

## How the code resolves the declaration

This table was fixed and committed before the run.

| point | resolution | reason |
| --- | --- | --- |
| Cb | `draw_scenario(SCENARIO, 2000, k)` and `fit_cleverly(frame)` of `canonical_shift_policies`, transcribed by `cleverly_rows` | the registered fit and rows |
| Ca | `ShiftSet.evaluate(shifts(), result.data, AnalyticDensity(W))`, with the fit's own contrast reference, placed in `nuisance.shifts`, then `result.estimator.retarget(..., estimands=("ey_shift", "ate_shift"))` | the `reversed_ratio_control` seam that the declaration names |
| `AnalyticDensity` | `density_at(a) = phi((a - mu(W)) / sigma) / sigma`, with `mu` and `sigma` from the law's own `dose_mean` and `dose_scale`, where sigma is 1. `crossing_fraction` returns 1, because the exact density resolves every shift, and `ShiftSet.evaluate` reads it only to warn about bin resolution. The uncapped-support `PositivityWarning` is contained, as `fit_shift_estimator` contains it for Cb | the declared density |
| `sigma*` | the declared closed form, with `Var(A) = dose_scale^2 + sum(loading^2)` read off `dose_mean`, 1.58 | the declared bound |
| `D` intervals | the framework `bootstrap` over the three arms' `std_error` columns, paired by replicate, 10,000 draws, seed `stream_seed(SHIFT, "rm18", "comparator-density", "bootstrap", "D")`. The three differences come from the same index matrix | the declared paired bootstrap |
| framework excess | `comparison._bounds` for each pair, with the pair as subject and reference, on the registered stream `stream_seed(SHIFT, "equivalence", scenario, estimand)`. Its `calibration_excess_upper` is the bound | the declared framework calibration excess, bound and resolution |
| harness validation | Cb reproduces every committed `cleverly` row of the five estimands to R4. The (Cb, L) bound and resolution reproduce the values in the committed `equivalence.csv` to R4. The refit is the Cb arm itself, because CD draws nothing new | R4 |
| reading | read from the top of the declared table, with "`D_Ca - D_L` covering 0 or below 0" as "not above 0" | the declared rule |

## Sources

| claim | source | locator |
| --- | --- | --- |
| the efficient influence function of the shift contrast | Díaz Muñoz and van der Laan (2012), *Biometrics* 68(2):541-549, DOI 10.1111/j.1541-0420.2011.01685.x | Result 1, equation (5) |
| `lmtp` receives the analytic ratio with `.trim = 1`, so its classifier and trimming do not apply | `tests/canonical/lmtp_point_adapter.R` and `tests/canonical/lmtp_shift/run_study.R` | the density-ratio arguments |

## Runtime

One Cb fit takes 1.9 to 2.1 s on one core, and a retarget 0.01 s. The declared run took 774.9 s
of wall time on 16 logical cores. `run.log` records it. A smoke run at `--replicates 4` took
13 s.
