# C-TMLE candidate sequences versus R `ctmle`

This directory holds the registered study `canonical-ctmle-sequences`. The study compares the
lasso and the declared-learner candidate sequences of Cleverly's C-TMLE with R
[`ctmle`](https://github.com/jucheng1992/ctmle) 0.1.2 at commit `18de559`. The Python phase
and the property study have run. The R runs are pending (Study queue item 9), so this directory
holds no R row, no replay file and no R selection file yet.

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
| `cand_est`, `cand_loss` | both | the estimate and the negative log-likelihood of each candidate. Lasso: R's own values. Learners: computed from R's `Qstars`, bounded into `[1 - alpha, alpha]` as the package bounds them |
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
best score (`near_tie`). The package bounds every targeted `Q` into `[1 - alpha, alpha]`. R
bounds only the initial `Q` and each accepted update (`functions_general.R`, lines 185 to 218).
The `clip` flag covers that difference, and it reads the package side, because R writes no
targeted `Q`.

| scenario | what R writes | `clip` flags a replicate when |
| --- | --- | --- |
| `binary_lasso` | R's own unbounded values | a targeted `Q` of the package is not strictly inside `(1 - alpha, alpha)` |
| `binary_learners` | values from R's `Qstars`, bounded as the package bounds them | a targeted `Q` of the package on an observed row is not strictly inside `(1 - alpha, alpha)`. There the package solves the score of the bounded fit and R's `glm` the unbounded one |

The flag reads every try that the construction scores on the analysis sample. This includes a
try that loses its round and a round-1 try that an acceptance discards. Each of these tries sets
a score that can decide the path. `replay-summary.csv` also records the largest gaps on the
flagged replicates in the `flagged_max_*` columns. No gate reads those columns.

A `binary_learners` flag on every bound would leave too few replicates. The low bound binds on a
counterfactual arm on 124 of 200 replicates in the design review 2 emulation. A bound on an
observed row binds on 10 of 200 replicates of the pre-flight in
[Smoke before the declaration](#smoke-before-the-declaration).
`tests/unit/test_ctmle_sequence_replay.py` recomputes the verdict from `replay.csv`.

A red replay gate does not stop the run. `test_ctmle_sequence_replay.py` fails on a red gate
in the committed `replay-summary.csv` until the roadmap's "Red-cell owners" table lists
`X16-sequence-reading`.

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

The study reads two committed inputs and two pilot tables. Write and commit all four before the
first run. The manifest of the run hashes the two inputs.

| file | content | command |
| --- | --- | --- |
| `lambdas.csv` | one grid per scenario: 20 penalties, `scenario`, `entry` and `lambda` | `python -m tests.canonical.ctmle_sequences.design lambdas` |
| `r_folds.csv.gz` | R's fold of each `binary_learners` row: `replicate`, `row`, `fold` | `python -m tests.canonical.ctmle_sequences.design folds` |
| `power-probe.csv` | the pilot: 1,200 fits per law on the `pilot` seed stream, with each mutation's bias | `python -m tests.canonical.ctmle_sequences.power_probe` |
| `property-pilot.csv` | the property pilot: 4,800 fits of each property cell that no other study measures, on the `pilot` seed stream | `python -m tests.canonical.ctmle_sequences.power_probe --properties` |

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

## Smoke before the declaration

Run the steps below outside the artifact directory before the declaration commit. Stop and
re-review the plan when a step fails.

| step | command or file | pass rule |
| --- | --- | --- |
| Python pre-flight | `uv run --extra dev python -m tests.canonical.ctmle_sequences.preflight --replicates 200` | the unflagged share of each scenario exceeds `MIN_UNFLAGGED / 800`, every unflagged index matches, and every unflagged gap is below `1e-7` |
| R smoke | the probe command under [Regenerate](#regenerate), 10 replicates | the facts of plan section 3.5 hold. Every unflagged replay gap is below `1e-7`. `replay-summary.csv` records the flag rates |
| property pilot | `uv run --extra dev python -m tests.canonical.ctmle_sequences.power_probe --properties` | `DECLARED_GATES` in `tests/unit/test_ctmle_sequences_design.py` holds each measured bias to three decimals. Each gate passes with probability 0.99 or more, and so does the family |

The pre-flight stands in for R. It fits R's arrays with the package. It computes R's values with
the package's bound on each candidate removed, and it bounds R's written values as the runner
does. It reports the flag counts and the largest unflagged gaps. It cannot see the tolerance of
R's `glm`, so a correct replay reads a gap of 0. The pre-flight measured the result below. The
`clip` flag read every scored try in this run.

| scenario | replicates | `near_tie` | `clip` | unflagged | final-step comparisons | largest unflagged gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `binary_lasso` | 200 | 31 | 3 | 166 | 95 | 0 |
| `binary_learners` | 200 | 4 | 10 | 186 | 0 | 0 |

At 800 replicates these shares give about 664 and 744 unflagged replicates against the 400 that
the gate needs.

The property pilot measures the property cells that no other study measures, and every cell
that fits the declared learners. It fits each cell 4,800 times on the `pilot` seed stream, which
shares no seed with a registered replicate. The SE column is the delta-method Monte Carlo SE of
the bias in its own SD. The four learners cells ran again on the nested declaration, with
`python -m tests.canonical.ctmle_sequences.power_probe --properties --learners`.

| cell | bias in SD | Monte Carlo SE | rule | pass probability |
| --- | ---: | ---: | --- | ---: |
| `double_robustness/lasso_treatment_correct` | -0.042 | 0.014 | inside the margin at 1,200 replicates | 0.999998 |
| `double_robustness/learners_treatment_correct` | -0.080 | 0.014 | inside the margin at 1,200 replicates | 0.99954 |
| `double_robustness/learners_outcome_correct` | -0.007 | 0.014 | inside the margin at 1,200 replicates | 1.0 |
| `double_robustness/both_wrong_learners` | 4.511 | 0.049 | outside the margin at 1,200 replicates | 1.0 |
| `multi_arm/lasso_treatment_correct` | -0.041 | 0.014 | inside the margin at 1,200 replicates | 0.999998 |
| `multi_arm/empty_control` | 2.761 | 0.031 | outside the margin at 1,200 replicates | 1.0 |
| `selector_necessity/learners_collaborative` | 0.315 | 0.014 | the declared expected red, not a gate | 0.000005 |

The family of every gate passes with probability 0.9991. The first declaration's learners cells
read -0.042 (pilot), -0.043 and 4.27 (design review probe). `STAND_INS` in
`tests/studies/canonical_ctmle_sequences.py` names the keys of `DECLARED_GATES` that hold a
bias from another study. It is empty. `regenerate.py` refuses a full run while `STAND_INS` names
a key. A probe run with another `--replicates` or with `--primary-only` does not need this
check.

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

## Expected red rows

The declaration expects the red rows below before the run. `EXPECTED_RED` in
`tests/studies/canonical_ctmle_sequences.py` holds them, and
`tests/unit/test_ctmle_sequences_design.py` checks that the committed red set equals it.

| row | why it is expected red | owner |
| --- | --- | --- |
| `binary_lasso` coverage | the plan's probe read a coverage of 0.899 and an SE ratio of 0.881 | `X16-plugin-variance` |
| `selector_necessity/learners_collaborative` | its bias clause: the binary selector precedent is 0.217 SD, so the clause passes with probability about 0.05 (design review S3) | F18 |
| `selector_necessity/lasso_collaborative`, `selector_necessity/empty_control` | the family's joint clause only, which the red bias clause above fails | F18 |

The RMSE bound of each collaborative cell is the necessity family's discriminating gate. It is
expected to pass.

The first run owned the `binary_lasso` coverage row under F18. A reading after that run moved it
to `X16-plugin-variance`. The final step reports the penalty 0 on 771 of 800 fits. Plain TMLE on
the same 800 samples covers 0.9225 with an SE ratio of 0.913, and both estimators cover below
0.85 in the lowest quartile of the reported standard error
(`tests/diagnostics/x16_lasso_coverage/`). The shortfall is the plug-in variance under
near-positivity, and not the selection that F18 owns. The row does not test Delta(P1) of F18.

## History

| run | change | result |
| --- | --- | --- |
| first, at `d106a637` (commit `caf17b8c`) | the declared learners of the property cells were `W2`, then `W1` | `selector_necessity/learners_collaborative` was red on its RMSE bound as well as its bias. Bias 0.575 SD (99% interval 0.0101 to 0.0139 against a margin of 0.0052), coverage 0.814, SE ratio 0.602, RMSE ratio 0.472 with a 99% upper bound of 0.511 against 0.50 |
| second | the declared learners became nested: `W2`, then `W1` and `W2` | see the committed `properties.csv` |

The cause of the first result is a study-design defect. `W1` alone fits the treatment worse than
`W2` alone, so the declaration broke the template's decreasing treatment loss.
`treatment_risk_decreasing` was false on every fit. Along the path, index 1 fluctuates the
targeted fit of index 0, so it inherits the bias of index 0. At n = 200,000 index 1 lies about
4.0 SD from the truth along the path, and about 0.08 SD as a standalone fit. The first run's
design test read the standalone fit and did not see the defect. The test now reads each index
along the built path, and it keeps the first declaration as a mutation control. The package
now warns when a declared candidate's treatment loss rises by more than 1% over the candidate
before it.

The nested declaration also changes the double-robustness learners cells. The property pilot
measured all four learners cells again on the `pilot` seed stream.

## Scope

The comparison covers binary, two-arm, complete-outcome ATE estimation. Continuous outcomes,
missing outcomes, weights, clusters, and other estimands are outside this comparison.
