# Estimator variants

## Ordinary TMLE and CV-TMLE

`TMLEMethod` fits the estimand-specific nuisance functions, targets the relevant loss, evaluates the
plug-in parameter, and reports influence-curve inference. With outer cross-fitting, every nuisance
prediction used for an observation is produced without that observation in the nuisance training
set.

The default CV-TMLE construction stacks out-of-fold predictions and fits one targeting regression
over the validation rows. Zheng & van der Laan (2011) supplies the original cross-validated TMLE
framework; Levy (2018) identifies the stacked construction. The pinned `tmle3` `cvtmle=TRUE` source
and its `sl3` fold/full-prediction dependency corroborate the engineering semantics. They are
recorded in the [general targeted-learning references](../references.md#targeted-learning-in-general).

Implementation: [`estimators/tmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/tmle.py),
[`estimators/recipe.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/recipe.py),
and [`learners/crossfit.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/learners/crossfit.py).
Evidence: out-of-fold leakage controls and externally supplied fold integrity in
[`tests/unit/test_crossfit_leakage.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_crossfit_leakage.py), repeated-split
aggregation in [`tests/unit/test_repeated_crossfit.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_repeated_crossfit.py),
and parallel invariance in
[`tests/unit/test_parallel_invariance.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_parallel_invariance.py).

## Collaborative TMLE

Collaborative TMLE selects the treatment-mechanism nuisance with respect to the target parameter
rather than predictive propensity loss alone. For each candidate along a truncation or covariate
path, the implementation targets the outcome regression and evaluates validation loss; it selects
the candidate, refits the selected specification, and performs the final target update.

`CollaborativeTMLEMethod(strategy="greedy")` uses the package's scalable candidate path.
`strategy="oat"` uses an outcome-adaptive categorical mechanism fit on the matrix of arm-specific
outcome predictions. The method is available only for identified effects whose collaborative
score and selection path are implemented.

Theory: van der Laan & Gruber (2010), Gruber & van der Laan (2010), and Ju et al. (2019), listed
under [Collaborative TMLE](../references.md#collaborative-tmle). The pinned `ctmle3` `LF_oat` and
`tmle3_Spec_TSM_all` source is provenance for the OAT categorical construction; pinned `tmle3` and
`sl3` source informs nested validation/full-fit semantics.

Implementation: [`estimators/ctmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/ctmle.py).
Evidence: the [estimator-variant rows](../evidence.md#estimator-variants-over-registered-targets),
candidate-path identities, selection mutations, multi-arm OAT design tests, and explicit refusal
tests. The external sources do not implement the same complete selector and are not numeric oracles.

## DR-TMLE

DR-TMLE adds reduced regressions and compatible nuisance fluctuations so valid inference can remain
possible when one primary nuisance is inconsistent under the method's rate and regularity
conditions. It does not promise narrower intervals and it is not the ordinary efficient estimator
under misspecification.

The univariate construction estimates reduced outcome and mechanism regressions, solves their
additional score equations, alternates compatible updates, and reports the corrected influence
curve. A bivariate reduction uses the two-column $(\hat Q(a,W),\hat g(a\mid W))$ design and its
distinct score. For multi-valued treatment, each armwise branch is constructed separately; this is
implementation-backed provenance, not an expansion of a binary theorem.

The theory and scope are unusually conditional, so the authoritative account is the
[DR-TMLE production contract](../drtmle.md). It maps van der Laan (2014), Benkeser et al.
(2016/2017), Benkeser & Hejazi (2023), and Díaz & van der Laan (2017) to the precise equations and
supported missing-outcome setting. The pinned R `drtmle` 1.1.2 source supplies formula provenance,
armwise iteration, and `cvFolds` engineering choices.

Implementation: [`estimators/drtmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/drtmle.py),
[`estimators/reduced.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/reduced.py),
and [`validation/drtmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/drtmle.py).

Acceptance evidence is indexed from the [DR-TMLE contract](../drtmle.md) and includes an
independent theorem identity, Gateaux derivative, and second-order remainder under each guard in
[`tests/unit/test_influence_gateaux_drtmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_influence_gateaux_drtmle.py) and
[`tests/unit/test_influence_drtmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_influence_drtmle.py), together with the
cross-fit split contract in
[`tests/unit/test_drtmle_crossfit.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_drtmle_crossfit.py) and the end-to-end
refusals in [`tests/e2e/test_double_robustness.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/e2e/test_double_robustness.py).
Agreement with R `drtmle` alone is not acceptance evidence.
