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
| candidate sequence | the 20 penalties of `lambdas.csv` and the penalty 0, or four lasso learners at grid entries 4, 8, 12 and 16 | the same penalties, or the same four learners and `SL.glm` |
| datasets | binary-outcome samples generated in Python with their exact ATE | the identical rows and all three covariates |
| selection folds | `binary_lasso`: an unstratified five-fold split that the fit asserts. `binary_learners`: R's folds from `r_folds.csv.gz` | `binary_lasso`: the package's own five folds. `binary_learners`: the same folds, drawn again from the replicate's seed |
| outcome regression | a logistic GLM on `A`, `W1`, `W2` and `W3` | the same GLM |
| propensity bounds | 0.025 to 0.975 | 0.025 to 0.975 |
| construction loss | the loss only (`penalty=False`) | the loss plus $\operatorname{var}(D^*)$, whatever `PEN` is. Inside each fold, R multiplies that term by $n_{\text{train}}/n$ |
| selection loss | the cross-validated loss, unpenalized | the cross-validated loss, unpenalized (`PEN = FALSE`) |
| cross-fitting | disabled for the comparison | not applicable |
| variance | `plugin_std_error`, the plug-in variance of $D^*$ | `var.psi` of the type-1 final step, or `varDstar_origscale[best_k] / n` |

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_sequences/README.md#declared-differences)
lists each declared difference between the two fits. The R runner records the selected candidate
of each replicate and the minimum of each `glmnet` path that R reads. `r-selection.csv` holds
these records, and `r-selection-summary.csv` summarizes them.

The property study uses a public nested cross-fit configuration: five outer folds, three
selection folds, and two inner folds. Every two-arm property cell samples a proportion and
declares an outcome support of 0 to 1. The two multi-arm cells sample the binary three-arm law.

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
| selector RMSE ratio | 0.50 | `SELECTOR_RMSE_RATIO` in `tests/studies/ctmle_sequence_properties.py` |
| property grid | 11 penalties and the penalty 0 | `PROPERTY_GRID_SIZE` in the same module |
| red-cell owners | F18 for every coverage or standard-error-ratio red cell. `X16-sequence-reading` for a red paired row, or for a positive double-robustness cell that is red on its bias | `RED_CELL_OWNERS` in `tests/studies/canonical_ctmle_sequences.py` |
| expected adapter rate | at least one third of the `binary_lasso` replicates. The package selects the penalty 0 in 67 of 200 fits on replicates that the run does not use | the fixture README, "Expected adapter rate" |
| learners candidates | a main-terms logistic regression on `W2`, then on `W1` | `LEARNER_COLUMNS` in `tests/studies/ctmle_sequence_properties.py`. On the instrument law, index 0 omits the confounder. `tests/unit/test_ctmle_sequences_design.py` shows that its bias is outside the margin |

## Limitations

The [red-cell ledger](red-cells.md) lists each red cell of a study with the roadmap ask that owns
it. This study has no committed cell yet.

| limitation | what it means for use |
| --- | --- |
| The study is not yet run | No row of this page is evidence until the committed results exist |
| R refits on every row inside each fold | R refits the fluctuation on every row, and its evaluation refits the lasso path on every row. `cleverly` refits inside each selection fold. The selected candidate can differ for this reason alone |
| R penalizes its construction | R adds $\operatorname{var}(D^*)$ to each construction score, whatever `PEN` is. `cleverly` with `penalty=False` does not. The two paths can accept different candidates |
| The parity claim is narrow | It is binary, two-arm, complete-outcome, GLM-outcome, non-cross-fitted, unpenalized ATE only |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_sequences/README.md)
records the declaration inputs and the regeneration commands.
