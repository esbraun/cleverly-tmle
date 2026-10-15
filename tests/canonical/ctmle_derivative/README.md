# C-TMLE0 derivative covariate versus R `ctmle`

This directory holds the repeated-sampling evidence for `CTMLE(strategy="derivative")`. The
declared comparator is R [`ctmle`](https://github.com/jucheng1992/ctmle)
`ctmleGlmnet(ctmletype = 3)` at commit `18de559`. The comparator runs are pending, as the
registry declares with `pending_reference`. They wait for beta-plans TRACKER item 18. The
committed results are the Python primary phase and the property phase. `equivalence.csv` has no
rows, and the manifest records the comparator with `status: pending`. The R image is the
selector study's image, built from `../ctmle_selector/Dockerfile`. `manifest.json` records the
package commit, the source hashes, the configuration, the result hashes and the package
versions.

Python draws every binary-outcome sample and its exact ATE truth. Both sides bound both
mechanisms at 0.025 and 0.975.

| scenario | outcome model | mechanism pair |
| --- | --- | --- |
| `binary_declared_pair` | the main-terms logistic GLM | both sides fit the lasso at the pair in `penalties.csv` |
| `binary_pipeline` | the main-terms logistic GLM | each side runs its own penalty cross-validation and partner rule |
| `binary_constant_q` | `mean(Y)` for both arms | the pair in `penalties.csv` |

Under the correct GLM the derivative columns barely move the estimate. Under the constant
outcome model they move it by about 0.23 SD on each replicate. When the R runs exist, the SD of
the Python minus R difference must be at most 0.1 times the SD of the C-TMLE0 minus TMLE
difference on the two declared-pair scenarios. `paired-tmle.csv` publishes both SDs and the
verdict. While the comparator is pending, the witness columns of that file are empty. R bounds
only the initial outcome fit. The package also bounds each targeted arm prediction at 0.005
and 0.995, so the constant-outcome scenario is expected to differ by about 1e-5 in SD.

`penalties.csv` is the lower median of the cross-validated penalty over the draws in
`penalty-design.csv`, with that draw's grid neighbour as the partner. Rewrite both files with
`python -m tests.studies.canonical_ctmle_derivative`.

The property study is independent of R. It has 21 property rows, and all pass. It checks
interval calibration in four cells, cluster-robust inference against an IID control, type-I
error, power, the root-n ladder, double robustness with correct and wrong outcome models, and
targeting necessity. Each calibration, clustered, null, power and double-robustness fit also
records the ordinary TMLE on the same initial fit. A red cell would follow the declared reading
in `tests/studies/ctmle_derivative_properties.py`. No cell is red.

Regenerate from the repository root. While the comparator is pending, the run needs no Docker:

```powershell
uv run --extra dev python -m tests.canonical.ctmle_derivative.regenerate
```

To reuse the committed property files, add `--skip-properties`. The driver then refits the
Python primary phase only. When a later change registers the comparator, it moves the name from
`pending_reference` to `reference` and runs the same command with Docker running. The R phase
then runs and fills `equivalence.csv` and the witness columns of `paired-tmle.csv`.

The comparison covers binary, two-arm, complete-outcome ATE estimation with the lasso pair.
Continuous outcomes, flexible learner libraries and simultaneous intervals are outside scope.
