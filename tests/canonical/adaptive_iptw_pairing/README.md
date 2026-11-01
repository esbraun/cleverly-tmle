# AdaptiveIPTW against R `drtmle` (`adaptive-iptw-pairing`)

This study pairs `AdaptiveIPTW` with `drtmle::adaptive_iptw` (drtmle 1.1.2, commit 538a3a2) on
the gate study's laws (`adaptive-iptw`). Both implementations see the same draws, the same folds
and the same initial mechanism at each key. The paired verdict is the per-replicate agreement of
the two estimates, in units of the package's standard error.

| file | role |
| --- | --- |
| `tests/studies/canonical_adaptive_iptw_pairing.py` | The record, draws, scenarios, fits, the samples the runner reads, and the agreement rule. |
| `tests/studies/adaptive_iptw_pairing_properties.py` | No property cells; the module the driver needs. |
| `tests/canonical/adaptive_iptw_pairing/run_study.R` | The R phase: one `adaptive_iptw` call per key, with its checks. |
| `tests/canonical/adaptive_iptw_pairing/regenerate.py` | The declared-run driver and the reading of its results. |
| `tests/unit/test_adaptive_iptw_pairing_design.py` | The design test: runner tables, the fold rule, mutation controls, the agreement rule. |

## How the folds reach drtmle

`adaptive_iptw` draws its folds with its one call to `sample(seq_len(n))`. The Python phase
draws one permutation per cross-fitted draw (one per stratum on law ST), deals row `perm[i]` to
fold `i mod K` and writes each row's 1-based position. The runner calls a copy of
`adaptive_iptw` whose environment is a child of the drtmle namespace holding a `sample` stub
that returns `order(position)`. The body and formals are drtmle's, checked on every call.

## Checks on every call

| check | rule | on failure |
| --- | --- | --- |
| composition | the key's mechanism composed in R from the raw factors equals the package's to 1e-12 | fatal |
| fold | drtmle's initial `Qr` equals a default-control `glm` rebuilt on the supplied folds to 1e-10 | fatal |
| stub | `sample` appears once in the body, the stub runs once per cross-fitted call, and the permutation is one | fatal |
| nuisance gate | the package's initial `Qr` equals the converged `glm` to 1e-8 | recorded failure of the replicate |
| curve | each rebuilt curve's variance equals drtmle's `cov` to relative 1e-10 | fatal |
| range | the initial and final mechanisms lie strictly inside `[tolg, 1 - tolg]` | recorded failure of the replicate |
| convergence | on a targeted key, drtmle's final score is within `tolIC` (the package's own loop must stop at its tolerance, checked in Python) | recorded failure of the replicate |

## Agreement

`agreement()` writes `agreement.csv`. A key agrees when the 99th percentile, over the
replicates both sides returned, of `|cleverly - drtmle| / SE` and of the relative SE difference
both stay under the scenario's bound, and the initial estimates agree to 1e-10. A replicate
either side lost is a recorded failure, which `account_failures` counts against each side's
`failure_bound`. The bound is 1e-4 in sample and for the one-step form, where both run the same arithmetic.
It is 0.05 for a cross-fitted targeted fit: drtmle tilts each row along its own fold's
covariate and the package tilts each fold's slab, so the two differ at order `1/n` after the
first round (declared difference D9). On the C l3, CN and MA composites (`DESCRIPTIVE`) that
gap reached 0.07 to 0.22 SE in the pilot, so their targeted agreement is reported with no
verdict and their one-step forms carry it at 1e-4. That set was chosen after the pilot. A key
whose mean targeting movement is under ten times its bound is labelled "targeting not
witnessed".
