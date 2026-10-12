# R `ctmle` candidate-sequence witness

This directory holds one exactness comparison for the `lasso` and `learners` strategies of
`cleverly.estimators.CTMLE`. It is not a registered study.

## What runs

`regenerate.py` draws one fixture sample with `harness.draw_inputs` and writes `inputs.csv` and
`grid.csv`. The pinned image `cleverly-ctmle-reference:18de559` then runs `witness.R` on them.
That image is built from the unchanged `tests/canonical/ctmle_selector/Dockerfile`. `witness.R`
calls `glmnet.control(fdev = 0, devmax = 1)`, the same adapter setting as the registered study.

| input | content |
| --- | --- |
| rows | 1,000 |
| covariates | `W1` strong instrument; `W2`, `W3` confounders; `W4` outcome only; `W5`, `W6` noise |
| `Q0W`, `Q1W` | the package's main-terms logistic fit, in `[0.02, 0.98]`. The outcome law has a `W2 * W3` term |
| `fold` | a five-fold assignment |
| `A3`, `w` | a three-arm treatment and row weights, for the solver rows only |
| `grid.csv` | 20 log-spaced penalties from `0.9` to `0.002` times `lambda_max` |

| R output | R call | package test |
| --- | --- | --- |
| `g_tight.csv`, `coef_tight.csv` | `glmnet(x, A, "binomial", lambda = grid, thresh = 1e-14, maxit = 1e7)` | the package KKT residual is at most `1e-10`; the package residual of `glmnet`'s coefficients is at most `1e-6`; the predictions agree |
| `g_weighted.csv`, `coef_weighted.csv` | the same with `weights = w` | as above |
| `g_multinomial.csv`, `coef_multinomial.csv` | the same with `family = "multinomial"` on `A3` | KKT and predictions |
| `facts.csv` `lambda_max`, `lambda_max_weighted` | the first value of the default path | `default_penalty_grid()[0]` at a relative `1e-10` |
| `g_internal.csv`, `g_glm.csv` | R's own arrays: the default path read with `predict(s = grid)`, and the `glm` propensity | the inputs of the next two rows |
| `general_path.csv` | `ctmleGeneral(ctmletype = 1, gn_candidates = G, gn_candidates_cv = G, PEN = FALSE)` with `G = cbind(g_internal, g_glm)` | `_indexed_path` on `G`: the accepted indices are exact; estimates and losses agree to `1e-6` |
| `glmnet_path.csv`, `g_final_cols.csv` | `ctmleGlmnet(ctmletype = 1, lambdas = grid, folds = fold, PEN = FALSE)`, and the explicit-lambda fit of its final step at R's `best_k` | `_indexed_path` on `g_internal`: the accepted positions are exact; estimates and losses agree to `1e-6`; the final step on `g_final_cols` gives R's estimate to `1e-6` |

Each prediction tolerance is ten times the self-gap that `witness.R` measures against a run at
`thresh = 1e-16` and a higher `maxit`. `manifest.json` records each tolerance under
`prediction_tolerance`.

The package's own cross-validation is not compared with R. R does not refit the outcome or the
candidates inside each fold, and it refits `epsilon` on all rows there (X16 plan, section 1.4,
quirk 2).

## The seed search

`regenerate.py` tries each seed of `SEEDS` in order and keeps the first that passes. A seed must
pass these conditions:

| condition | why |
| --- | --- |
| the initial `Q` lies in `[0.02, 0.98]`, and every targeted `Q` lies inside `(0.005, 0.995)` | both sides' `Q` bounds are no-ops |
| the path accepts at least two candidates and skips one (W1) | the jump rule binds |
| the stop rule ends the construction before the last candidate (W2) | the stop rule binds |
| each deciding score gap exceeds `100 x 1e-8` of the round's best score | R's `glm` tolerance cannot flip a decision |
| each accepted `ctmleGeneral` column lies at squared distance `1e-4` or more from every earlier column of its block | R's `< 1e-5` column match finds it alone (quirk 5) |
| R does not select the last penalty | `ctmleGlmnet` type 1 errors there (quirk 7) |
| the final step moves the estimate by more than `1e-4`, and its index is neither end of the remaining penalties (W3) | the final step's argmin binds |

The screen runs on the package's own solver arrays first. Only a seed that passes it runs R. The
R outputs must then pass the screen again on R's arrays, and the conditions on R's selection.
`manifest.json` records every rejected seed and its reason under `search`.

## Regenerate

Run from the repository root with Docker running:

```powershell
python -m tests.canonical.ctmle_sequence_witness.regenerate
```

`tests/unit/test_ctmle_sequence_witness.py` checks the file hashes, redraws the inputs from the
seed, and runs the comparisons.

## Precision of the R outputs

R's `write.csv` prints 15 significant digits. The tests read every CSV with the
`round_trip` parser. They compare the positions that R marks as accepted, not the printed
penalties. The 15-digit rounding is far inside each stated tolerance.
