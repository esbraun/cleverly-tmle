# C-TMLE candidate sequences versus R `ctmle`

This directory holds the registered study `canonical-ctmle-sequences`. The study compares the
lasso and the declared-learner candidate sequences of Cleverly's C-TMLE with R
[`ctmle`](https://github.com/jucheng1992/ctmle) 0.1.2 at commit `18de559`. The study is
declared and not yet run. This directory holds no replicate, summary, or manifest yet.

## Scenarios

Python draws every sample and its exact ATE truth from the three-covariate binary law. Both
implementations fit the same GLM outcome model and the same propensity bounds, 0.025 and 0.975.

| scenario | Cleverly | R `ctmle` | selection folds |
| --- | --- | --- | --- |
| `binary_lasso` | lasso sequence on `STUDY_LAMBDAS` and the penalty 0 | `ctmleGlmnet`, `ctmletype = 1` | the package's five unstratified selection folds, which R reads from the `selection_fold` column |
| `binary_learners` | four declared lasso learners at `LEARNER_LAMBDAS` | `build_gn_seq`, then `ctmleGeneral`, `ctmletype = 1` | R's folds from `r_folds.csv.gz` |

`run_ctmle_sequences.R` records nine columns beyond the replicate schema. The driver writes them
to `r-selection.csv` and summarizes them in `r-selection-summary.csv`.

| column | content |
| --- | --- |
| `best_k` | the candidate that R selects, 1-based |
| `n_candidates` | the number of candidates that R's full-sample construction accepts |
| `adapter_used` | whether the runner rebuilt a type-1 fit whose best candidate is the last one |
| `path_min_lambda` | the smallest penalty of the default `glmnet` path on all rows. The accepted refit and the evaluation read this path |
| `path_min_lambda_fold_1` to `path_min_lambda_fold_5` | the smallest penalty of the default path on the rows outside R's fold `v`. The construction inside fold `v` scores its tries on this path |

`predict(s = 0)` clamps the penalty 0 to the minimum of the path that it reads. The construction
reads a path on its training rows (`functions_glmnet.R`, lines 132 to 135). R's fold `v` is the
package's selection fold `v - 1`. The `binary_learners` rows hold no path minima.

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

The declaration expects R to need the type-1 adapter on at least one third of the
`binary_lasso` replicates. The table gives the two estimates behind this expectation.

| estimate | result | source |
| --- | --- | --- |
| the package selects the penalty 0 | 67 of 200 fits, 34% | `fit_cleverly` on `binary_lasso` replicates 800 to 999 of the study's seed stream. The run uses replicates 0 to 799 only |
| R fails with the quirk-7 error | 9 of the 12 witness seeds that reached R | `search` in `tests/canonical/ctmle_sequence_witness/manifest.json`, on the witness law |

Quirk 2 moves R's selection toward later indices. R's rate is therefore expected to be at or above
the package's rate. `r-selection-summary.csv` reports the measured rate. The paired summary leaves
out the adapter replicates.

## Declaration inputs

The study reads two committed inputs. Write and commit both before the first run. The manifest
of the run hashes both files.

| file | content | command |
| --- | --- | --- |
| `lambdas.csv` | the primary grid: 20 penalties, `entry` and `lambda` | `python -m tests.canonical.ctmle_sequences.design lambdas` |
| `r_folds.csv.gz` | R's fold of each `binary_learners` row: `replicate`, `row`, `fold` | `python -m tests.canonical.ctmle_sequences.design folds` |

The grid rule is in `primary_grid()` of `tests/studies/canonical_ctmle_sequences.py`. The
fast test `tests/unit/test_ctmle_sequences_design.py` recomputes the rule and compares it with
`lambdas.csv`.

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
and the two declaration inputs mount read-only at `/fixture`.

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
| `double_robustness` | four correct-nuisance cells and `both_wrong` | the shared double-robustness rule |
| `selector_necessity` | `lasso_collaborative`, `learners_collaborative`, `empty_control` | each collaborative RMSE is at most 0.50 of the paired control RMSE |
| `multi_arm` | `lasso_treatment_correct`, `empty_control` | the treatment-correct cell passes and the control is discriminated |

## Scope

The comparison covers binary, two-arm, complete-outcome ATE estimation. Continuous outcomes,
missing outcomes, weights, clusters, and other estimands are outside this comparison.
