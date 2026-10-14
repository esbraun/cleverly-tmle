# C-TMLE0 derivative covariate versus R `ctmle`

This directory holds the paired repeated-sampling comparison of
`CTMLE(strategy="derivative")` with R [`ctmle`](https://github.com/jucheng1992/ctmle)
`ctmleGlmnet(ctmletype = 3)` at commit `18de559`. The R image is the selector study's image,
built from `../ctmle_selector/Dockerfile`. `manifest.json` records the package commit, the
source hashes, the configuration, the result hashes and the package versions.

Python draws every binary-outcome sample and its exact ATE truth. Both sides bound both
mechanisms at 0.025 and 0.975.

| scenario | outcome model | mechanism pair |
| --- | --- | --- |
| `binary_declared_pair` | the main-terms logistic GLM | both sides fit the lasso at the pair in `penalties.csv` |
| `binary_pipeline` | the main-terms logistic GLM | each side runs its own penalty cross-validation and partner rule |
| `binary_constant_q` | `mean(Y)` for both arms | the pair in `penalties.csv` |

Under the correct GLM the derivative columns barely move the estimate. Under the constant
outcome model they move it by about 0.23 SD on each replicate. On the two declared-pair
scenarios, the SD of the Python minus R difference must be at most 0.1 times the SD of the
C-TMLE0 minus TMLE difference. `paired-tmle.csv` publishes both SDs and the verdict. R bounds
only the initial outcome fit. The package also bounds each targeted arm prediction at 0.005
and 0.995, so the constant-outcome scenario differs by about 1e-5 in SD.

`penalties.csv` is the lower median of the cross-validated penalty over the draws in
`penalty-design.csv`, with that draw's grid neighbour as the partner. Rewrite both files with
`python -m tests.studies.canonical_ctmle_derivative`.

The property study is independent of R. It checks interval calibration in four cells,
cluster-robust inference against an IID control, type-I error, power, the root-n ladder,
double robustness with correct and wrong outcome models, and targeting necessity. Each
calibration, clustered, null, power and double-robustness fit also records the ordinary TMLE
on the same initial fit. A red cell follows the declared reading in
`tests/studies/ctmle_derivative_properties.py`.

Regenerate from the repository root with Docker running:

```powershell
uv run --extra dev python -m tests.canonical.ctmle_derivative.regenerate
```

The comparison covers binary, two-arm, complete-outcome ATE estimation with the lasso pair.
Continuous outcomes, flexible learner libraries and simultaneous intervals are outside scope.
