# C-TMLE candidate sequences versus R `ctmle`

This directory holds the registered study `canonical-ctmle-sequences`. The study compares the
lasso and the declared-learner candidate sequences of Cleverly's C-TMLE with R
[`ctmle`](https://github.com/jucheng1992/ctmle) 0.1.2 at commit `18de559`. The study is
declared and not yet run. This directory holds no replicate, summary, or manifest yet.

## Scenarios

Python draws every sample and its exact ATE truth. Each scenario has its own law on four
covariates, `W1` to `W4`, and its own penalty grid. Both implementations fit the same main-terms
GLM outcome model and the same propensity bounds, 0.025 and 0.975. The GLM omits the `A W3`
term of both laws, so each candidate's TMLE limit depends on how much of `W3` its mechanism reads.

| scenario | law | Cleverly | R `ctmle` | selection folds |
| --- | --- | --- | --- | --- |
| `binary_lasso` | `x16_lasso_law`: `W3` enters the lasso path late | lasso sequence on `STUDY_LAMBDAS["binary_lasso"]` and the penalty 0 | `ctmleGlmnet`, `ctmletype = 1`, then the type-1 final step from R's `best_k` and from the package's index | the package's five unstratified selection folds, which R reads from the `selection_fold` column |
| `binary_learners` | `x16_learners_law`: entries 2 and 4 leave out most of `W3` | four declared lasso learners at entries 2, 4, 16 and 19 of the `binary_learners` grid | `build_gn_seq`, then `ctmleGeneral`, `ctmletype = 1` | R's folds from `r_folds.csv.gz` |

`tests/unit/test_ctmle_sequences_design.py` checks the population claims of both laws: the spread
of the candidate limits, the plug-in bias, and the share of `g` outside the bounds.

## What R writes

`run_ctmle_sequences.R` writes the columns below beyond the replicate schema. The driver
writes them to `r-selection.csv` and `replay.csv`, and summarizes them in
`r-selection-summary.csv` and `replay-summary.csv`. A vector column holds `%.17g` values joined
by `;`.

| column | scenario | content |
| --- | --- | --- |
| `best_k` | both | the candidate that R's cross-validation selects, 1-based |
| `n_candidates` | both | the number of candidates that R's full-sample construction accepts |
| `adapter_used` | both | 1 when the runner rebuilt a type-1 fit whose best candidate is the last one |
| `path_min_lambda` | lasso | the smallest penalty of the default `glmnet` path on all rows. The accepted refit and the evaluation read this path |
| `path_min_lambda_fold_1` to `path_min_lambda_fold_5` | lasso | the smallest penalty of the default path on the rows outside R's fold `v`. The construction inside fold `v` scores its tries on this path |
| `adapter_error_matched` | lasso | 1 when the caught `ctmleGlmnet` message holds the quirk-7 text |
| `rebuild_best_k`, `rebuild_agrees` | lasso | the `best_k` of `rebuild_type1`, and 1 when it agrees exactly with a completed `ctmleGlmnet` |
| `q_coef` | both | `coef(glm)`: the intercept, `A`, and `W1` to `W4` |
| `g_coef` | both | the propensity coefficients of each candidate, candidate first. Lasso: the default path read at each penalty. Learners: the four `SL.glmnet_new` fits, then `SL.glm` |
| `r_accepted` | both | the accepted candidates, 1-based |
| `cand_est`, `cand_loss` | both | the estimate and the negative log-likelihood of each candidate |
| `final_r_*` | lasso | R's type-1 final step from `best_k`: the start, the coefficients, the option losses, the final grid index and the estimate. Missing when `best_k` is the last index |
| `final_pkg_*` | lasso | the same step from the package's cross-validated index, which the sample frame carries in `package_selected`. Missing when that index is the last one |

`predict(s = 0)` clamps the penalty 0 to the minimum of the path that it reads. The construction
reads a path on its training rows (`functions_glmnet.R`, lines 132 to 135). R's fold `v` is the
package's selection fold `v - 1`. The `binary_learners` rows hold no path minima.

The runner stops when `plogis(cbind(1, W) %*% coef)` misses an array that R used by more than
`1e-12`. It checks the lasso candidates, the final-step arrays, the learners candidates and the
outcome fit. `rebuild_type1.R` holds the type-1 adapter and the final step, which the runner and
the witness's `rebuild_check.R` share.

## The replay gates

R's cross-validation leaks (quirk 2), so the two selected estimates are not one estimator. The
paired rows of `equivalence.csv` are descriptive and carry no `passed` column. The replay
gates compare what both sides compute from the same inputs.
`tests/studies/ctmle_sequence_replay.py` rebuilds R's arrays from the coefficients and runs the
package's construction and final step on them.

| gate | rule, per scenario |
| --- | --- |
| accepted candidates | identical on every unflagged replicate |
| candidate estimates and losses | within `1e-6`, absolute for an estimate and relative for a loss |
| final step (`binary_lasso`) | for each start that R wrote, the final index is identical, and the estimate and every option loss are within `1e-6` |
| counts | at least 400 unflagged replicates, and for `binary_lasso` at least 100 final-step comparisons |

A replicate is flagged and left out when a deciding score gap lies below `1e-6` of the round's
best score (`near_tie`), or when a targeted `Q` of the package leaves `(0.005, 0.995)` (`clip`).
R writes no targeted `Q`, so the `clip` flag reads the package side only.
`tests/unit/test_ctmle_sequence_replay.py` recomputes the verdict from `replay.csv`.

## Declared differences

The two implementations differ in these ways. A red paired row is read against this list first.

| difference | R `ctmle` | `cleverly` | scenarios |
| --- | --- | --- | --- |
| outcome refit | R does not refit the outcome inside a fold | the package refits the outcome inside each selection fold | both |
| fluctuation inside a fold (quirk 2) | R refits `epsilon` on every row, then scores the validation rows | the fluctuation uses the training rows only | both |
| `glmnet` in the evaluation (quirk 3) | R refits the path on every row | each fold fits its own training rows | `binary_lasso` |
| penalty 0 (quirk 4) | `predict` clamps the penalty 0 to the path minimum. The path minima above record each clamp | the penalty 0 is solved exactly | `binary_lasso` |
| construction criterion | the loss plus `var(D*)`, whatever `PEN` is (`functions_glmnet.R`, lines 172 to 177; `functions_general.R`, lines 65 to 72). The stop rule can bind | the loss only, because the study sets `penalty=False`. The stop rule does not bind | both |
| construction penalty inside a fold (quirk 8) | `var(D*)` times `n_train / n` | no penalty | both |
| terminal candidate | R reads the `cv` column of `SL.glm` | the package refits its terminal under the split | `binary_learners` |

The study pairs `penalty=False` with `PEN = FALSE`, the convention of the selector study
`canonical-ctmle-selector`. No package setting reproduces R's penalized construction together
with R's unpenalized selection loss. `tests/unit/test_ctmle_sequence_witness.py` checks the
penalized construction against R exactly.

## Expected adapter rate

The declaration expects R to need the type-1 adapter on close to every `binary_lasso`
replicate. Quirk 2 drives R's cross-validated loss toward the last candidate. The rate is
published in `r-selection-summary.csv` and is not gated.

| estimate | result | source |
| --- | --- | --- |
| R selects the last penalty | 10 of 10 probe replicates, one-sided 95% Clopper-Pearson lower bound 0.74 | the rev 2.1 smoke probe, X16 plan section 3.4 |
| compared replicates without the adapter | about 0 of 800, 95% upper bound about 207 | the same probe |
| R fails with the quirk-7 error | 9 of the 12 witness seeds that reached R | `search` in `tests/canonical/ctmle_sequence_witness/manifest.json`, on the witness law |

The package's cross-validated index is a different quantity. On the probe of the
`x16_lasso_law`, the package's cross-validation selected the last index on 574 of 1,200 fits.
The runner adapts an error only when its message holds the quirk-7 text and the rebuilt
`best_k` is the last penalty. `rebuild_type1` also runs where `ctmleGlmnet` completes, and the
runner stops unless both agree exactly. The paired summary leaves out the adapter replicates, so
its interval is conditional on `ctmleGlmnet` completing without the adapter.

## Declaration inputs

The study reads two committed inputs and one pilot table. Write and commit all three before the
first run. The manifest of the run hashes the two inputs.

| file | content | command |
| --- | --- | --- |
| `lambdas.csv` | one grid per scenario: 20 penalties, `scenario`, `entry` and `lambda` | `python -m tests.canonical.ctmle_sequences.design lambdas` |
| `r_folds.csv.gz` | R's fold of each `binary_learners` row: `replicate`, `row`, `fold` | `python -m tests.canonical.ctmle_sequences.design folds` |
| `power-probe.csv` | the pilot: 1,200 fits per law on the `pilot` seed stream, with each mutation's bias | `python -m tests.canonical.ctmle_sequences.power_probe` |

The grid rule is in `primary_grid()` of `tests/studies/canonical_ctmle_sequences.py`. The
fast test `tests/unit/test_ctmle_sequences_design.py` recomputes the rule and compares it with
`lambdas.csv`. The same file reads `power-probe.csv` and asserts the floors of plan section
3.4.3: each gate passes with probability 0.99 or more when the code is correct, and each
mutation fails a gate with probability 0.8 or more.

The `folds` command needs Docker and the reference image. The command writes the replication
seeds of the runner and runs `make_r_folds.R` on them. For each replicate, the script sets the
replicate's seed and draws five folds the way `cv_general` draws them. The runner draws the folds
again from the same seed and stops if they differ from the file.

To run `make_r_folds.R` by hand, write the seeds first:

```powershell
uv run --extra dev python -m tests.canonical.ctmle_sequences.design seeds --output $env:TEMP\seeds.csv
docker run --rm --entrypoint Rscript -v ${PWD}\tests\canonical\ctmle_sequences:/fixture:ro -v ${env:TEMP}:/work cleverly-ctmle-reference:18de559 /fixture/make_r_folds.R /work/seeds.csv 2000 /work/r_folds.csv.gz
```

## Regenerate

Run the study from the repository root with Docker running:

```powershell
uv run --extra dev python -m tests.canonical.ctmle_sequences.regenerate
```

The reference image is the image of `tests/canonical/ctmle_selector`. The runner, the harness,
`rebuild_type1.R` and the two declaration inputs mount read-only at `/fixture`.

`r_folds.csv.gz` holds folds at `n = 2000` only. A probe at another sample size stops in
`binary_learners`. A probe with fewer replicates reads the first rows of the file:

```powershell
uv run --extra dev python -m tests.canonical.ctmle_sequences.regenerate --replicates 10 --primary-only --output $env:TEMP\ctmle-sequences-probe
```

## Property cells

The property study is `tests/studies/ctmle_sequence_properties.py`. It uses Cleverly's public
nested cross-fit and no R. The cells are independent evidence, not part of the R comparison.

| property | cells | verdict |
| --- | --- | --- |
| `double_robustness` | four correct-nuisance cells, `both_wrong_lasso` and `both_wrong_learners` | the shared double-robustness rule |
| `selector_necessity` | `lasso_collaborative`, `learners_collaborative`, `empty_control` | the paired bootstrap 99% upper bound of each collaborative RMSE over the control RMSE is at most 0.50 |
| `multi_arm` | `lasso_treatment_correct`, `empty_control` | the treatment-correct cell passes and the control is discriminated |

The outcome-correct cells sample `outcome_correct_dgp()`, which adds `f = tanh(W1) tanh(W2)` to
both mechanisms. The correct outcome learner appends `f` to the design. No candidate mechanism
reads `f`. Each both-wrong control fits its strategy with a constant outcome on the same law.
`property-selection.csv` records the cross-validated index and the final index of every lasso and
learners fit, 1-based.

## Scope

The comparison covers binary, two-arm, complete-outcome ATE estimation. Continuous outcomes,
missing outcomes, weights, clusters, and other estimands are outside this comparison.
