# Stacked arm-indexed missing-outcome CV-TMLE evidence

This directory freezes the registered repeated-sampling study for the one-repeat, pooled,
whole-sample-evaluated stacked CV-TMLE of arm-indexed means and contrasts under MAR.
`docs/roadmap.md` RM9 ("Registered study") is the contract.

`tests/studies/mar_arm_indexed_laws.py` declares four laws. Each law has a three-level
covariate `W`, and every nuisance is a table indexed by `W` and the arm.

| law | arms | outcome | estimands |
| --- | --- | --- | --- |
| L1 | 2 | binary | `ey0`, `ey1`, `ate`, `rr`, `or` |
| L2 | 2 | continuous on `(-2, 8)` | `ey0`, `ey1`, `ate` |
| L3 | 3 | binary | `ey[a]` for each arm; `ate`, `rr`, `or` against `high` |
| L4 | 3 | continuous on `(-2, 8)` | `ey[a]` for each arm; `ate` against `high` |

The continuous fits declare `q_bounds=(-2, 8)`, the known support. The primary study fits
separate depth-five trees, with minimum leaf size 25, for the outcome regression, the treatment
mechanism, and the response mechanism. Each law has nine or fewer covariate cells, so each tree
can fit the saturated model. The primary rows therefore do not test calibration under a
misspecified data-adaptive learner.

The property study has four parts:

| part | what it checks |
| --- | --- |
| calibration | coverage and SE calibration of every estimand with the primary trees, and a shrunken-SE control of each |
| simultaneous band | joint coverage of the default multiplier band over each law's estimands, and the same fits' pointwise intervals read jointly as a control |
| union model | each ATE contrast with only `Q`, only `g`, or only `pi` wrong, and a control with `Q` and `pi` both wrong |
| overfitting pair | a fully grown outcome tree with eight noise covariates and the exact mechanisms, fitted out of fold and in sample on the same draws |

The comparator is R `tmle` 2.1.1 with the same stitched out-of-fold outcome, treatment, and
response predictions. L1 and L2 use its native two-arm path. L3 and L4 run its population-mean
path once for each arm: `A` is constant, `Delta` is `1{A = a} Delta`, and `Y` is `NA` outside
the arm's respondents. `run_study.R` builds the joint covariance and each reference contrast
from the returned per-arm influence curves. R and `cleverly` both use the centered `n - 1`
variance of the influence curve.

R takes a continuous outcome's scale from every non-`NA` `Y` (`tmle.R` line 1120). Each
continuous fit therefore sets `Y` to `-2` and `8` on two rows whose response indicator is zero.
`probe_scale_workaround.R` checks that workaround on replication 0 and writes
`scale-probe.csv`. The regeneration refuses publication when any probe row fails. The probe is
not named `run_*.R`, because `tests/unit/test_canonical_runner_parity.py` holds those files to
the published-row contract.

Run a disposable primary smoke study from the repository root:

```powershell
uv run --extra dev python -m tests.canonical.tmle_mar_arm_indexed_cvtmle.regenerate --replicates 4 --n 200 --skip-properties --allow-failures --output build/tmle-mar-arm-indexed-cvtmle-smoke --jobs 16 --reference-jobs 16
```

That smoke probes the Python and R primary fitting paths and the scale probe. Exercise one fit
from every property cell before the full run:

```powershell
uv run --extra dev python -c "from tests.studies.mar_arm_indexed_cvtmle_properties import generate_smoke_property_rows; rows = generate_smoke_property_rows(); assert len(rows) == 84; print(rows.groupby('property').size())"
```

Regenerate the complete declared study:

```powershell
uv run --extra dev python -m tests.canonical.tmle_mar_arm_indexed_cvtmle.regenerate --jobs 16 --reference-jobs 16
```
