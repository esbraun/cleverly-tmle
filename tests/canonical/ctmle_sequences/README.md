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
| `binary_lasso` | lasso sequence on `STUDY_LAMBDAS` and the penalty 0 | `ctmleGlmnet`, `ctmletype = 1` | each side draws its own five folds |
| `binary_learners` | four declared lasso learners at `LEARNER_LAMBDAS` | `build_gn_seq`, then `ctmleGeneral`, `ctmletype = 1` | R's folds from `r_folds.csv.gz` |

`run_ctmle_sequences.R` records four columns beyond the replicate schema. The driver writes them
to `r-selection.csv` and summarizes them in `r-selection-summary.csv`.

| column | content |
| --- | --- |
| `best_k` | the candidate that R selects, 1-based |
| `n_candidates` | the number of candidates that R scores |
| `adapter_used` | whether the runner rebuilt a type-1 fit whose best candidate is the last one |
| `path_min_lambda` | the smallest penalty of the default `glmnet` path, which `predict` clamps to |

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
