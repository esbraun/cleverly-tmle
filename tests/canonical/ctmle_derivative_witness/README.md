# R witness of C-TMLE0 (`ctmle` 0.1.2, `ctmletype = 3`)

This directory holds one construction-matched comparison. It is not a registered study.

`regenerate.py` draws `inputs.csv` from a fixed seed (n = 1,000, six covariates) and writes
the declared settings to `penalties.csv`. The pinned image `cleverly-ctmle-reference:18de559`,
built from `tests/canonical/ctmle_selector/Dockerfile`, runs `witness.R` on them.

| block | mechanism `g1W` | partner `g1WPrev` | outcome | serves |
| --- | --- | --- | --- | --- |
| `A-binary` | `glmnet` at `lam` = 0.012 | `glmnet` at `lam_plus` = 0.018 | binary | parity |
| `A-continuous` | as `A-binary` | as `A-binary` | continuous, scaled by `range(Y)` | parity, R quirk 2 |
| `B` | as `A-binary` | `plogis(3 * qlogis(g1WPrev))`, which crosses `gbound` where `g1W` does not | binary | the bound on the partner |
| `C` | `gM`, clipped at both bounds | `gMPrev`, half of the logit of `gM` | binary | the bound on the mechanism in the ratio |

Both initial outcome regressions are fixed working models that leave the confounder `W3` out.
The derivative columns therefore move the fit.

`witness.R` does these steps:

1. It fits `glmnet` at the two declared penalties with `thresh = 1e-14`, and again with
   `thresh = 1e-18`. `versions.csv` records the largest difference as `self_gap`.
2. `type3()` copies `R/ctmle_glmnet.R` lines 100-112 and 133-174. It has two edits: the `glm`
   control is an argument, and the function returns its intermediate arrays. The script stops
   unless the default pass equals `ctmleGlmnet(ctmletype = 3)` in `est`, `var.psi` and `CI`
   bit for bit.
3. The script also stops unless `ctmleGeneral(ctmletype = 3)` and
   `ctmleGlmnet(..., fluctuation = "linear")` equal `ctmleGlmnet(ctmletype = 3)` bit for bit.
4. A second pass of the copy at `glm.control(epsilon = 1e-14, maxit = 100)` gives the
   reference values in `summary.csv` and `rows-<block>.csv`.

`regenerate.py` then checks four conditions before it writes `manifest.json`:

- No finite difference is zero, so R's `1e-10` substitution (quirk 1) cannot act.
- Every targeted prediction lies strictly inside `(0.005, 0.995)`, so the package's shrink
  after each step (quirk 3) cannot act.
- The normalized four-column design has a smallest singular value above `1e-6`.
- Block B crosses `gbound` on at least 20 rows. Block C clips at least 20 rows at each bound,
  and on each of them the bounded partner differs from the bounded mechanism.

`tests/unit/test_ctmle_derivative_witness.py` runs the package's update on R's own arrays at
`alpha = 0.995` and `g_bounds = (0.025, 0.975)`. It compares the coefficients at `1e-6`
relative, and the targeted fit, the curve `Dstar`, the estimate and `var.psi` at `1e-6`
absolute. R's `var()` divides by `n - 1`, so the test multiplies the package variance by
`n / (n - 1)`.

Regenerate from the repository root with Docker running:

```powershell
python -m tests.canonical.ctmle_derivative_witness.regenerate
```
