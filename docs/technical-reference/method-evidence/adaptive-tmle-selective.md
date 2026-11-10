# Adaptive TMLE under selective enrollment

This study measures `AdaptiveTMLEMethod` on a randomized trial pooled with external rows
(`TrialWithExternalData`, `TrialATE`) when enrollment into the trial depends on covariates and the
external bias is large and wiggly. [Adaptive TMLE](../adaptive-tmle.md) gives the estimand, the
estimator and its conditions. The study is the sibling of the
[trial study](adaptive-tmle-trial.md), whose nuisance learners are splines. Here the nuisances are
a Super Learner, whose cross-fitted cost is two orders higher, so the study has its own sample
size and budget.

The study declares no property cell and no reference, so it is not a validation row, as
[adding a method row](../../development/method-benchmarking.md#adding-a-method-row) states, and
the [validation grid](validation-grid.md) does not list it. It publishes under `reporting`. The
declared run of 2026-11-09 read `s6_cv` red on its bias leg. The diagnosis found no defect, so the
cell stays red and the owner row
[`X90-cv-finite-sample`](../../roadmap.md#red-cell-owners) holds it. The numbers of the repeat
run are pending, and this page quotes none.

**No canonical implementation is compared.** The R `atmle` comparator is declared and pending, as
a separate pairing record. It never blocks this study.

## What was tested

| setting | declaration |
| --- | --- |
| law | the paper's law (b) with selective enrollment, $S \sim \mathrm{Bernoulli}(\mathrm{expit}(-0.9 + 0.8 W_1 + 0.5 W_2))$, and an external bias of four residual standard deviations that the dictionary does not span. 1.4% of the covariate mass has an enrollment probability outside [0.05, 0.95] |
| fit | `study.identify(TrialATE(population)).estimate(method=AdaptiveTMLEMethod(...))` with a Super Learner of a spline GLM, shallow regularized gradient boosting and a lasso for each nuisance. Probabilities use `nnloglik` with clip (0.01, 0.99) and the outcome uses NNLS |
| forms | `s6_cv`, the cross-validated form, and `s6_pooled`, the pooled form with no cross-fitting of the working model |
| size | n = 3,200 pooled rows |
| primary budget | 2,400 replications of each scenario, the sizing rule's floor at a 1,200-replicate pilot |
| truth | quadrature of $\Psi$ from the law, and of the working-model projection on each fit's stored fold transformers, so the truth varies by replicate |
| primary verdicts | the framework defaults: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| control | K6, the same fit with main-terms GLM nuisances on the same draws. It is a reporting row because a pilot found no power at this size |
| seeds | `seed=20290321` and `resampling_seed=20290322` |
| policy | `reporting` |

Both scenarios gate $\Psi$. The laws, scenarios, budgets and seeds were fixed before any run, in
`tests/studies/adaptive_tmle_selective.py`, and
`tests/unit/test_adaptive_tmle_trial_design.py` recomputes the truths and the sizing without a run.

## Red cell of this study

The second declared run (2,400 replications, at `c6f9e51c`) read `s6_cv` red on the bias leg and
`s6_pooled` green. The diagnosis found no defect. The cross-validated form's targeting step fits
about 37 selected bias columns on 640 validation rows, and that ratio of columns to validation rows
leaves a finite-sample bias of about 0.2 error SDs. The same draws give the pooled form 0.13 SD
less. The working-model projection sits within 0.0035 of $\Psi$ and the interval covers the
projection at the nominal rate, so the cause is no gap between the two targets.

The one change that removed the term, a targeting direction fitted on the validation fold,
destabilized the trial study's separating law `e1` (its SE ratio fell from 0.97 to 0.86). It is
not adopted. The roadmap row `X90-cv-finite-sample` is a standing record of this reading and states
what would close it. The study publishes the cell red under `reporting` at its declared budget and
margin, and no margin, replication count, law, seed, learner or label moved.

## Result

Pending the repeat run. A repeat at the same seeds, replication count, margins and laws follows the
declaration of 2026-11-10 in the fixture README.

## Limitations

| limitation | what it means for use |
| --- | --- |
| No grid row | The study declares no property cell and no reference, so the validation grid does not list it and the method has no validation row |
| There is no cross-implementation evidence | The R `atmle` pairing is declared, pending, and not run |
| One size and one learner class | Every scenario draws n = 3,200 with one Super Learner library. No other size or library is measured |
| The interval covers the projection | The coverage of $\Psi$ is read only where the dictionary approximates the external bias, and no data check tells the cases apart |
| Binary treatment, difference scale | Other treatments and scales refuse before any learner |
| Compositions outside the study | Weights, clusters, a second outcome and repeated fits are not measured |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/adaptive_tmle_selective/README.md)
gives the run command, the sizing and the declaration history. The committed artifacts will carry
the replications, the controls and the reported readings once the repeat run lands.
