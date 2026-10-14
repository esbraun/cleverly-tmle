# C-TMLE candidate sequences

This study validates the lasso and the declared-learner candidate sequences of `cleverly`'s
C-TMLE. The canonical comparison uses R [`ctmle`](https://github.com/jucheng1992/ctmle) 0.1.2 at
pinned commit
[`18de559`](https://github.com/jucheng1992/ctmle/tree/18de559f47dc1286617350a0668391e80e1dbf7c).
The theory is van der Laan and Gruber (2010).

The study is declared and not yet run. The three generated sections below are empty until the
committed results exist. The `reporting` publication policy was declared before the run.

## What was compared

| setting | `cleverly` | R `ctmle` |
| --- | --- | --- |
| scenarios | `binary_lasso` and `binary_learners` | `ctmleGlmnet` for `binary_lasso`, and `build_gn_seq` with `ctmleGeneral` for `binary_learners` |
| laws | `x16_lasso_law` for `binary_lasso` and `x16_learners_law` for `binary_learners`. Each law has four covariates, `W1` to `W4`. `W1` is an instrument, and `W3` is a confounder and an effect modifier | the identical rows |
| candidate sequence | `binary_lasso`: the 20 penalties of its grid in `lambdas.csv` and the penalty 0. `binary_learners`: four lasso learners at entries 2, 4, 16 and 19 of its grid, and a main-terms logistic terminal | the same penalties, or the same four learners and `SL.glm` |
| datasets | binary-outcome samples generated in Python with their exact ATE | the identical rows and all four covariates |
| selection folds | `binary_lasso`: an unstratified five-fold split that the fit asserts. `binary_learners`: R's folds from `r_folds.csv.gz` | `binary_lasso`: the package's own five folds. `binary_learners`: the same folds, drawn again from the replicate's seed |
| outcome regression | a logistic GLM on `A` and `W1` to `W4`. It omits the `A W3` term of both laws | the same GLM |
| propensity bounds | 0.025 to 0.975 | 0.025 to 0.975 |
| construction loss | the loss only (`penalty=False`) | the loss plus $\operatorname{var}(D^*)$, whatever `PEN` is. Inside each fold, R multiplies that term by $n_{\text{train}}/n$ |
| selection loss | the cross-validated loss, unpenalized | the cross-validated loss, unpenalized (`PEN = FALSE`) |
| cross-fitting | disabled for the comparison | not applicable |
| variance | `plugin_std_error`, the plug-in variance of $D^*$ | `var.psi` of the type-1 final step, or `varDstar_origscale[best_k] / n` |

The outcome GLM is misspecified on both laws. Each candidate's TMLE limit therefore depends on
how much of `W3` its mechanism reads. `tests/unit/test_ctmle_sequences_design.py` asserts the
spread of these limits before the run.

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_sequences/README.md#declared-differences)
lists each declared difference between the two fits. R refits the fluctuation on every row
inside each fold (quirk 2). This leak drives R's cross-validation toward the last candidate, so
the two selected estimates are not one estimator. The paired rows of the agreement section are
therefore descriptive. They carry no `passed` column. The red-cell ledger reads none of them.

The file `r-selection.csv` holds R's selected candidate of each replicate. The package's
cross-validated index, final index and accepted candidates stand beside it. The file
`r-selection-summary.csv` counts the type-1 adapter replicates. Its paired interval reads only
the replicates where `ctmleGlmnet` completed without the adapter. The interval is therefore
conditional on that event.

The replay gates carry the comparison with R. R writes its outcome and propensity coefficients.
The module `tests/studies/ctmle_sequence_replay.py` rebuilds R's arrays from them. It then runs
the package's construction and final step on R's arrays and R's initial outcome fit.

| replay gate | rule, per scenario |
| --- | --- |
| accepted candidates | identical on every unflagged replicate |
| candidate estimates and losses | within $10^{-6}$, absolute for an estimate and relative for a loss |
| final step, `binary_lasso` | for each start that R writes, the final index is identical, and the estimate and every option loss are within $10^{-6}$ |
| counts | at least 400 unflagged replicates, and for `binary_lasso` at least 100 final-step comparisons |

The replay leaves out a flagged replicate. A replicate is flagged when a deciding score gap lies
below $10^{-6}$ of the round's best score. It is also flagged when a targeted outcome value of
the package leaves $(0.005, 0.995)$. The file `replay-summary.csv` records the counts, the flags
and the largest gap of each kind.

The property study uses a public nested cross-fit configuration: five outer folds, three
selection folds, and two inner folds. Every two-arm property cell samples a proportion and
declares an outcome support of 0 to 1. The two multi-arm cells sample the binary three-arm law.
`property-selection.csv` records the cross-validated index and the final index of every lasso
and learners property fit.

## Accuracy against known truth

<!-- generated: accuracy -->
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
<!-- /generated -->

## Declared design

The values in this table are declared before the run.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | replications per scenario |
| `n` | 2000 | observations per replication |
| paired role | descriptive | `paired_role` of the record in `tests/studies/canonical_ctmle_sequences.py` |
| replay tolerance | $10^{-6}$ | `GLM_TOLERANCE` in `tests/studies/ctmle_sequence_replay.py` |
| replay counts | 400 unflagged replicates per scenario, 100 final-step comparisons for `binary_lasso` | `MIN_UNFLAGGED` and `MIN_FINAL` in the same module |
| selector RMSE ratio | 0.50, applied to the paired bootstrap 99% upper bound of each ratio | `SELECTOR_RMSE_RATIO` and `RMSE_BOUND_LEVEL` in `tests/studies/ctmle_sequence_properties.py` |
| property grid | 11 penalties and the penalty 0 | `PROPERTY_GRID_SIZE` in the same module |
| red-cell owners | F18 for every coverage or standard-error-ratio red cell, and for every `selector_necessity` red cell. `X16-sequence-reading` for a red package bias row, a red replay gate, or a positive double-robustness cell that is red on its bias | `RED_CELL_OWNERS` in `tests/studies/canonical_ctmle_sequences.py` |
| expected adapter rate | close to every `binary_lasso` replicate. On the rev 2.1 law, R selected the last penalty on 10 of 10 probe replicates. The package's CV-selected index is a different quantity | the fixture README, "Expected adapter rate" |
| learners candidates of the property cells | a main-terms logistic regression on `W2`, then on `W1` | `LEARNER_COLUMNS` in `tests/studies/ctmle_sequence_properties.py`. On the instrument law, index 0 omits the confounder. `tests/unit/test_ctmle_sequences_design.py` shows that its bias is outside the margin |

## Limitations

The [red-cell ledger](red-cells.md) lists each red cell of a study with the roadmap ask that owns
it. This study has no committed cell yet.

| limitation | what it means for use |
| --- | --- |
| The study is not yet run | No row of this page is evidence until the committed results exist |
| R refits on every row inside each fold | R refits the fluctuation on every row, and its evaluation refits the lasso path on every row. `cleverly` refits inside each selection fold. The selected candidate can differ for this reason alone. The paired rows are descriptive for this reason |
| R penalizes its construction | R adds $\operatorname{var}(D^*)$ to each construction score, whatever `PEN` is. `cleverly` with `penalty=False` does not. The replay gates use `penalty=True` on both sides for this reason |
| A green coverage or SE-ratio row says nothing about F18 | Where the outcome model is correct, Ju et al. (2018) Lemma 2 gives $\Delta(P_1) = 0$, so coverage holds conditionally on the outcome fit and its limit, which F18's reopening condition excludes. Where the outcome model is wrong, one law's coverage bounds no remainder term |
| The multi-arm lasso cell tests the unpenalized end of the path | The cell reported that end on every smoke seed. It does not test interior selection at three arms |
| The lasso final step returns to the unpenalized end on most fits | On these laws the lasso estimate and plain TMLE are close in distribution. The replay gates, not the truth gates, show that the sequence is built as R builds it |
| The parity claim is narrow | It covers binary, two-arm, complete-outcome, GLM-outcome, non-cross-fitted ATE estimation only |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_sequences/README.md)
records the declaration inputs and the regeneration commands.
