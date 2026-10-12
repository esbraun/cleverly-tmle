# C-TMLE0 derivative covariate versus R `ctmle`

This directory holds the paired repeated-sampling comparison of
`CTMLE(strategy="derivative")` with R [`ctmle`](https://github.com/jucheng1992/ctmle)
`ctmleGlmnet(ctmletype = 3)` at commit `18de559`. The R image is the selector study's image,
built from `../ctmle_selector/Dockerfile`. `manifest.json` records the package commit, the
source hashes, the configuration, the result hashes and the package versions.

Python draws every binary-outcome sample and its exact ATE truth. Both sides fit the same
main-terms logistic outcome model and bound both mechanisms at 0.025 and 0.975.

| scenario | mechanism pair |
| --- | --- |
| `binary_declared_pair` | both sides fit the lasso at the pair in `penalties.csv` |
| `binary_pipeline` | each side runs its own penalty cross-validation and partner rule |

`penalties.csv` is the lower median of the cross-validated penalty over the draws in
`penalty-design.csv`, with that draw's grid neighbour as the partner. Rewrite both files with
`python -m tests.studies.canonical_ctmle_derivative`.

The property study is independent of R. It checks interval calibration in five cells,
type-I error, power, the root-n ladder, double robustness and targeting necessity. Each
calibration, null and power fit also records the ordinary TMLE on the same initial fit
(`paired-tmle.csv`). A red cell follows the declared reading in
`tests/studies/ctmle_derivative_properties.py`.

Regenerate from the repository root with Docker running:

```powershell
uv run --extra dev python -m tests.canonical.ctmle_derivative.regenerate
```

The comparison covers binary, two-arm, complete-outcome ATE estimation with the lasso pair.
Continuous outcomes, flexible learner libraries and simultaneous intervals are outside scope.
