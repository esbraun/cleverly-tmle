# References

Every paper the package's derivations are read off, in one place, with enough of a locator to
find the passage a docstring or a document is pointing at.

**How to read a citation in this repository.** Prose cites an author and a year, and the citation
resolves here. Two examples are "the sequential regression of Bang & Robins (2005)" and "van der
Laan (2014) Theorem 3". Where a document argues *against* a source, or transcribes a display from
it, it carries a section or page number as well; those are the citations worth checking, and they
are the ones that have them.

**Nothing here is stored in the repository.** Two PDFs were, and were deleted once everything they
were cited for had been transcribed into [the DR-TMLE contract](technical-reference/dr-tmle/index.md). That is why the two
`DRTMLE` sources below carry page numbers where the rest carry none. A path to a file only a
previous reader had is not a citation; a page number is.

## Targeted learning, in general

- van der Laan & Rubin (2006), [*Targeted Maximum Likelihood Learning*](https://doi.org/10.2202/1557-4679.1043),
  DOI 10.2202/1557-4679.1043.
- Díaz Muñoz & van der Laan (2012), [*Population Intervention Causal Effects Based on
  Stochastic Interventions*](https://doi.org/10.1111/j.1541-0420.2011.01685.x), DOI
  10.1111/j.1541-0420.2011.01685.x.
- Gruber & van der Laan (2010), *A targeted maximum likelihood estimator of a causal effect on a
  bounded continuous outcome*.
- Gruber & van der Laan (2012),
  [*tmle: An R Package for Targeted Maximum Likelihood Estimation*](https://doi.org/10.18637/jss.v051.i13),
  *Journal of Statistical Software* 51(13), DOI 10.18637/jss.v051.i13. Section 2.1,
  page 5, defines marginal risk and odds ratios from two counterfactual risks. Section 2.7,
  page 10, reports intervals for both ratios on the log scale. Appendix A, page 34, gives
  their log-scale influence curves.
- CRAN `tmle` 2.1.1, source at commit
  [`f8d88a0`](https://github.com/cran/tmle/tree/f8d88a07a3d25c96688221b384043eef7a31fe68).
  [`R/tmle.R`](https://github.com/cran/tmle/blob/f8d88a07a3d25c96688221b384043eef7a31fe68/R/tmle.R)
  normalizes `obsWeights`. It routes them through outcome and treatment fitting, targeting,
  plug-in evaluation, and influence-curve calculations. This source supports the weighted
  ordinary-TMLE refit. It does not implement a simulated common-cause surface.
- Zheng & van der Laan (2011), *Cross-validated targeted minimum-loss-based estimation*.
- Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins (2018),
  [*Double/debiased machine learning for treatment and structural
  parameters*](https://academic.oup.com/ectj/article/21/1/C1/5056401), *The Econometrics
  Journal* 21(1):C1-C68. Definition 3.3 defines the coordinatewise median point;
  Equation (3.14) defines the median of the within-partition variance plus squared split
  displacement.
- zEpid 0.9.1, repeated cross-fit aggregation at commit
  [`16a0f96`, lines 1602-1641](https://github.com/pzivich/zEpid/blob/16a0f96f8b2c65df8715085801f21757d1478e1e/zepid/causal/doublyrobust/crossfit.py#L1602-L1641).
  The `calculate_joint_estimate` median branch implements the same point and variance
  calculation.
  It is secondary aggregation evidence and not a comparator for the complete estimator.
- Levy (2018), *An Easy Implementation of CV-TMLE*, arXiv:1811.04573. The abstract
  distinguishes the original fold-wise plug-in evaluation from the common targeting
  regression pooled over validation folds.
- Coyle et al., R package [`tmle3`](https://github.com/tlverse/tmle3), source at commit
  [`ed72f8a`](https://github.com/tlverse/tmle3/tree/ed72f8a20e64c914ab25ffe015d865f7a9963d27).
  `R/tmle3_Update.R` selects the
  `"validation"` likelihood when `cvtmle=TRUE` and fits one update to the stacked
  validation predictions; `R/Param_TSM.R` evaluates the treatment-specific mean and its
  influence curve from those validation likelihood values. `R/delta_functions.R` defines the
  log-risk and log-odds contrasts. Used as an implementation reference, not as an oracle for the
  estimand derivation or as a moving specification. The simulated-confounding surface reports its
  ratio movement on the same log scale. Levy (2018) is the stable marker for the default stacked
  construction.
- The fold/full prediction mechanism used by `tmle3` lives in its `sl3` dependency, pinned
  here at [`0e8f236`](https://github.com/tlverse/sl3/tree/0e8f2365bcbe54010b8120c04a7a2dcfc8119227).
  `R/Lrnr_cv.R` builds `fold_fits` and, when requested, a `full_fit`; `predict_fold(...,
  "validation")` assembles held-out predictions while `predict_fold(..., "full")` uses the
  all-training fit. This is the design source for C-TMLE's nested selection predictions.
- van der Laan & Gruber (2016), *One-step targeted minimum loss-based estimation*.

## Point treatment and stochastic interventions

- van der Laan (2010), [*Targeted Maximum Likelihood Based Causal Inference: Part I*](https://doi.org/10.2202/1557-4679.1211),
  DOI 10.2202/1557-4679.1211, and [*Part II*](https://doi.org/10.2202/1557-4679.1241),
  DOI 10.2202/1557-4679.1241. These provide the general causal-effect and practical TMLE
  constructions cited by the conditional-population and regimen targets.
- Robins (2004), *Optimal Structural Nested Models for Optimal Sequential Decisions*, in
  *Proceedings of the Second Seattle Symposium on Biostatistics*, pp. 189–326, DOI
  [10.1007/978-1-4419-9076-1_11](https://doi.org/10.1007/978-1-4419-9076-1_11).
- Díaz & van der Laan (2013), [*Assessing the Causal Effect of Policies: An Example Using
  Stochastic Interventions*](https://doi.org/10.1515/ijb-2013-0014), *International Journal of
  Biostatistics* 9(2):161–174, DOI 10.1515/ijb-2013-0014. The companion to the 2012 paper above,
  cited for the stochastic-regime parameter rather than the population-intervention one. The
  byline differs between them: *Díaz Muñoz* in 2012 and *Díaz* in 2013. The prose follows
  each as published.
- Haneuse & Rotnitzky (2013), *Estimation of the effect of interventions that modify the received
  treatment*, *Statistics in Medicine* 32(30):5260–5277, DOI
  [10.1002/sim.5907](https://doi.org/10.1002/sim.5907).
- Díaz, Williams, Hoffman & Schenck (2023), [*Nonparametric Causal Effects Based on Longitudinal
  Modified Treatment Policies*](https://doi.org/10.1080/01621459.2021.1955691), *Journal of the
  American Statistical Association* 118(542):846–857, DOI 10.1080/01621459.2021.1955691. The
  package implements the point-treatment shift case; the citation supplies the general modified-
  policy identification and efficient influence-function theory, not a claim of longitudinal-shift
  support.
- van der Laan & Rose (2011), *Targeted Learning: Causal Inference for Observational and
  Experimental Data*, Springer. Chapter 12 covers marginal structural model targets.

## Collaborative TMLE

- van der Laan & Gruber (2010), [*Collaborative double robust targeted maximum likelihood
  estimation*](https://pmc.ncbi.nlm.nih.gov/articles/PMC2898626/), DOI
  10.2202/1557-4679.1181.
- Gruber & van der Laan (2010), [*An application of collaborative targeted maximum likelihood
  estimation in causal inference and genomics*](https://pmc.ncbi.nlm.nih.gov/articles/PMC3126668/),
  DOI 10.2202/1557-4679.1182.
- Ju, Gruber, Lendle, Chambaz, Franklin, Wyss, Schneeweiss & van der Laan (2019), [*Scalable
  collaborative targeted learning for high-dimensional data*](https://pmc.ncbi.nlm.nih.gov/articles/PMC6086775/),
  DOI 10.1177/0962280217729845.
- The R `ctmle` 0.1.2 implementation at commit
  [`18de559`](https://github.com/jucheng1992/ctmle/tree/18de559f47dc1286617350a0668391e80e1dbf7c).
  `R/ctmle_discrete.R` defines `ctmleDiscrete`, which the pinned selector-parity study calls
  through `tests/canonical/ctmle_selector/run_ctmle.R`. `R/functions_discrete.R` defines its
  `stage2` and `cv` helpers. `R/ctmle_general.R` and `R/functions_general.R` define
  `ctmleGeneral`, `stage2_general`, and `cv_general`. No argument list in those four files takes
  an observation weight, and no fit, sum, mean, or variance in them applies one. This source
  supplies no fixed-weight comparison.
- The `tlverse/ctmle3` implementation at commit
  [`a4ea77b`](https://github.com/tlverse/ctmle3/tree/a4ea77b07747dfee9b2eecb9cbca88262e0559ea).
  `R/LF_oat.R` fits categorical treatment on the complete vector of treatment-specific
  outcome predictions; `R/tmle3_Spec_TSM_all.R` requests all treatment-specific means.
  This is the implementation source for `CTMLE(strategy="oat")`.

## Longitudinal, survival and marginal structural models

- Bang & Robins (2005), *Doubly robust estimation in missing data and causal inference models*.
- van der Laan & Gruber (2012), *Targeted minimum loss based estimation of causal effects of
  multiple time point interventions*, DOI
  [10.1515/1557-4679.1370](https://doi.org/10.1515/1557-4679.1370). The intervention-specific
  mean and its sequential conditional-expectation representation are stated for general
  longitudinal data structures.
- Chaffee & van der Laan (2012), *Targeted Maximum Likelihood Estimation for Dynamic Treatment
  Regimes in Sequentially Randomized Controlled Trials*, DOI
  [10.1515/1557-4679.1406](https://doi.org/10.1515/1557-4679.1406). Treatment rules map histories
  into the treatment node's support; the worked examples do not limit the definition to two arms.
- Stitelman, De Gruttola & van der Laan (2012), *A General Implementation of TMLE for
  Longitudinal Data Applied to Causal Inference in Survival Analysis*, DOI
  [10.1515/1557-4679.1334](https://doi.org/10.1515/1557-4679.1334).
- Petersen, Schwab, Gruber, Blaser, Schomaker & van der Laan (2014), *Targeted Maximum
  Likelihood Estimation for Dynamic and Static Longitudinal Marginal Structural Working
  Models*, DOI [10.1515/jci-2013-0007](https://doi.org/10.1515/jci-2013-0007). Section 3
  defines the pooled longitudinal TMLE; Appendix A derives its efficient influence curve.
- Lendle, Schwab, Petersen & van der Laan (2017), *ltmle: An R Package Implementing
  Targeted Minimum Loss-Based Estimation for Longitudinal Data*, DOI
  [10.18637/jss.v081.i01](https://doi.org/10.18637/jss.v081.i01). The package source's
  `FixedTimeTMLE`, `CalcCumG`, and `UpdateQ` are the implementation locators used by the
  bounded canonical fixture.
- Poulos, Horvitz-Lennon, Zelevinsky et al. (2024), *Targeted learning in observational
  studies with multi-valued treatments: an evaluation of antipsychotic drug treatment safety*,
  DOI [10.1002/sim.10003](https://doi.org/10.1002/sim.10003). The accompanying public
  [`jvpoulos/multi-ltmle`](https://github.com/jvpoulos/multi-ltmle) repository contains
  longitudinal multi-valued simulation code. The paper's identified parameter and estimator are
  for one multi-valued treatment assignment. The repository is supporting implementation
  provenance for the estimator family. It is a simulation repository, so it gives no versioned
  package entry point for a paired study.
- Source audit snapshots (2026-08-16): R `ltmle` at
  [`338c029`](https://github.com/joshuaschwab/ltmle/tree/338c029dae9692ef20714125773da7037688993b)
  (`FixedTimeTMLE`, `CalcCumG`, `UpdateQ`) remains binary implementation provenance; `tmle3` at
  [`ed72f8a`](https://github.com/tlverse/tmle3/tree/ed72f8a20e64c914ab25ffe015d865f7a9963d27)
  (`LF_static`, `Param_TSM`, `Param_MSM`) confirms equality-density/static-intervention indexing
  but is not a longitudinal categorical oracle. The companion Poulos repository was inspected at
  [`0e8dc6e`](https://github.com/jvpoulos/multi-ltmle/tree/0e8dc6eca1012e5a3eab7aa80b772cf432b8f032).
- MSM study audit (2026-08-26): the same pinned `tmle3` `Param_MSM` supplies the Gaussian
  identity-link point projection after its arm-indicator coefficients and joint influence curves
  are mapped to the declared basis. Its documented custom-weight path needs a classed function to
  pass two premature string-sentinel comparisons in this release. The longitudinal comparator
  fits each plan with the pinned R `ltmle`, then projects the estimates and joint influence curves.
  Raw `ltmleMSM` coefficients are excluded because its quasibinomial projection is a different
  parameter.
- Source audit snapshot (2026-08-24): R `lmtp` 1.5.4 at
  [`f04a2b4`](https://github.com/nt-williams/lmtp/tree/f04a2b47f46debc515ce4ae778e05ebfde922c44).
  Its `cf_tmle` function runs one complete backward recursion per outer fold. Its
  `estimate_tmle` function fits and targets on training rows, then predicts validation rows.
  The public API accepts a fold count but not a realized assignment. A paired study therefore
  needs a pinned internal adapter before it can claim exact fold parity.
- Fixed-weight audit of the same `lmtp` snapshot:
  [`R/tmle.R`, lines 18-96](https://github.com/nt-williams/lmtp/blob/f04a2b47f46debc515ce4ae778e05ebfde922c44/R/tmle.R#L18-L96)
  keeps task weights on training rows and includes them in each targeting fluctuation. Its
  `run_ensemble` call on lines 44-48 receives no sampling weights, so the nuisance fits are
  unweighted. [`R/theta.R`, lines 1-15](https://github.com/nt-williams/lmtp/blob/f04a2b47f46debc515ce4ae778e05ebfde922c44/R/theta.R#L1-L15)
  supplies the weighted plug-in and influence-function aggregation. These locators support a
  weight-routing audit for modified treatment policies. The registered comparison is unweighted,
  and this source does not implement a simulated common-cause surface.
- Clustered inference audit (2026-08-29): the same `lmtp` snapshot passes its task identifier to
  `ife::ife`. Pinned [`ife` 0.2.3](https://cran.r-project.org/src/contrib/Archive/ife/ife_0.2.3.tar.gz)
  requires equal identifiers before it subtracts arm objects. That subtraction uses the joint
  rowwise influence curve. Its cluster standard error uses the variance of cluster means, so the
  registered study fixes every cluster at ten rows. The fixture pins the source archive by
  SHA-256.
- Categorical longitudinal audit (2026-08-27): the same `lmtp` snapshot accepts categorical
  treatment at multiple nodes for static and dynamic plans. The ordinary study supplies one
  all-row fold. The cross-fitted study supplies the exact five-fold assignment.
- The audit rejects [`npcausal`](https://rdrr.io/github/ehkennedy/npcausal/man/) for this row.
  Its public functions cover point effects, continuous-treatment curves, counterfactual
  densities, instrumental variables, and incremental interventions. They do not expose a
  deterministic categorical longitudinal regimen estimator.
- [`stremr`](https://github.com/romainkp/stremr) supports categorical longitudinal exposures and
  longitudinal TMLE. It requires long-form input, so it adds a data representation that the
  paired study does not need. The pinned `lmtp` path is the primary comparator.
- The audit also read the Poulos `multi-ltmle` companion. The Poulos entry in the sources above
  records that verdict.
- Competing-risk audit of the same snapshot: its survival path accepts a competing-event column
  through `compete=`. The returned estimate is one minus the cause-specific cumulative incidence.
  That value is not a survival probability. It counts a unit that had the competing event. The
  registered studies transform it to incidence and negate its influence curve.
- The registered competing-risk studies replace one internal `lmtp` function. `run_ensemble`
  becomes a direct single-learner fit. The adapter checks that substitution against `SuperLearner`
  on each run.
- Neugebauer & van der Laan (2007), [*Nonparametric causal effects based on marginal
  structural models*](https://doi.org/10.1016/j.jspi.2005.12.008), DOI
  10.1016/j.jspi.2005.12.008.
- Rosenblum & van der Laan (2010), [*Targeted Maximum Likelihood Estimation of the
  Parameter of a Marginal Structural Model*](https://doi.org/10.2202/1557-4679.1238),
  DOI 10.2202/1557-4679.1238.
- Orellana, Rotnitzky & Robins (2010), *Dynamic regime marginal structural mean models for
  estimation of optimal dynamic treatment regimes*.

## Incremental interventions

- Kennedy (2019), [*Nonparametric causal effects based on incremental propensity score
  interventions*](https://doi.org/10.1080/01621459.2017.1422737), *Journal of the American
  Statistical Association* 114(526):645–656, DOI 10.1080/01621459.2017.1422737.

## Riesz representation and nested targeted learning

- Balkus, Testa & Hejazi (2026),
  [*A Riesz Representer Perspective on Targeted Learning*](https://arxiv.org/abs/2604.21721),
  arXiv:2604.21721v1. Equation (1) gives the single-stage Riesz EIF; Theorem 1 and
  Corollary 1 on pages 5–6 give its general and conditional-mean forms; Theorem 2 on
  pages 6–8 gives the sequential EIF with cumulative representer products; Algorithm 1
  on pages 9–10 gives the nested TMLE order; Sections 5.1–5.2 instantiate point-treatment
  means and longitudinal treatment regimes.
- Chernozhukov, Newey & Singh (2022),
  [*Automatic Debiased Machine Learning of Causal and Structural Effects*](https://arxiv.org/abs/1809.05224),
  *Econometrica* 90(3), 967–1027, DOI 10.3982/ECTA18515. Equation (2.4) gives the
  orthogonal Riesz moment, equation (2.5) its product-bias identity, equations (3.1)–(3.2)
  the cross-fitted debiased and targeted estimators, and equations (3.3), (3.6), and (3.7)
  the dictionary moment, Gram matrix, and penalized minimum-distance Riesz learner.
- Testa, Balkus & Hejazi, R package
  [`RieszCML`](https://github.com/nshlab/RieszCML) at commit
  [`45e8d277`](https://github.com/nshlab/RieszCML/tree/45e8d277930cd0df4eb8a91a7c686ee4c6fdef09).
  `R/ComposedRieszCurve.R` and `R/riesz_tmle.R` are the pinned implementation locators
  for innermost-first storage, suffix cumulative products, sequential targeting, and the
  distinct intervention evaluation `alpha_star`. `tests/testthat/test-double-robustness.R`
  supplies secondary nonzero mutations for reversed products and observed-state plug-in
  updates. The pinned code uses the untargeted curve for a targeted estimate's reported
  variance and falls back to observed `alpha` when `alpha_star` is missing; neither choice
  is adopted without an independent derivation.

## Sensitivity analysis

- Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022), *Long story short: omitted variable bias
  in causal machine learning*.
- VanderWeele & Ding (2017), *Sensitivity analysis in observational research: introducing the
  E-value*.
- Scharfstein, Rotnitzky & Robins (1999), *Adjusting for nonignorable drop-out using semiparametric
  nonresponse models*.
- Sharma & Kiciman (2020), [*DoWhy: An End-to-End Library for Causal
  Inference*](https://arxiv.org/abs/2011.04216), arXiv:2011.04216. Pages 3–4 describe a simulated
  common cause correlated with treatment and outcome. The paper supports a qualitative stress
  surface. It does not derive sensitivity-adjusted inference or a calibration formula.
- Sharma, Syrgkanis, Zhang & Kiciman (2021), [*DoWhy: Addressing Challenges in Expressing and
  Validating Causal Assumptions*](https://arxiv.org/abs/2108.13518). Pages 4–6 state that these
  analyses require plausible domain values and cannot validate identification from observed data.
- The maintained DoWhy simulated common-cause refuter, source at commit `2116d5c`.
  [`_include_confounders_effect`, lines 346-419](https://github.com/py-why/dowhy/blob/2116d5cbace5a057937e03b2efba95c13140cc4c/dowhy/causal_refuters/add_unobserved_common_cause.py#L346-L419)
  supplies the four perturbation branches. [`_simulate_confounders_effect_once`, lines
  807-844](https://github.com/py-why/dowhy/blob/2116d5cbace5a057937e03b2efba95c13140cc4c/dowhy/causal_refuters/add_unobserved_common_cause.py#L807-L844)
  applies a complete fit and effect estimate after the perturbation.
  Its direct simulation branches supply the binary tail flip, $A'=A+k_AU$, $Y'=Y-k_YU$, and the
  binomial outcome tail flip. These are secondary finite-sample conventions only. `cleverly` uses
  original data per cell, one shared latent vector, common refit seeds, an exact zero anchor,
  explicit grids, and retained failures. It does not copy automatic ranges, categorical
  encoded-column deletion, or cumulative mutation of shared data. The refuter takes no weight
  argument in `_include_confounders_effect`, so this source supplies the perturbation and the
  refit only. It does not supply the weighted evaluation.
- The same pinned DoWhy refit preserves `effect_modifier_names` and `target_units`.
  Its [propensity-weighting estimator](https://github.com/py-why/dowhy/blob/2116d5cbace5a057937e03b2efba95c13140cc4c/dowhy/causal_estimators/propensity_score_weighting_estimator.py)
  rebuilds ATT and ATC weights from the treatment in the supplied dataset. The
  `estimate_effect` method selects those weights through `target_units="att"` or `"atc"`.
  Thus observed-treatment membership follows the perturbed data. Fixed baseline strata retain
  their original membership because the perturbation changes neither their names nor values.
  These are qualitative composition conventions; they supply no sensitivity-adjusted interval.
- Van der Laan (2010), [Part I, Section 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC3126670/),
  discusses the effect among treated as a parameter depending on outcome and treatment factors.
  `cleverly` reuses its registered ATT and ATC functionals on each perturbed empirical law.
  Treatment relabeling gives the control-population counterpart. The surface reports the changing
  group's share and does not claim an effect in the original group held fixed.
- DoWhy's pinned [calibration helpers](https://github.com/py-why/dowhy/blob/2116d5cbace5a057937e03b2efba95c13140cc4c/dowhy/causal_refuters/add_unobserved_common_cause.py#L213-L340)
  calibrates encoded coordinates separately. Its binary branch zeros one standardized column.
  Its continuous branch uses one column's correlation times the perturbed variable's standard
  deviation. Neither branch defines a logical categorical benchmark on these strength scales.
- Fixed-weight binary DR-TMLE surfaces compose that DoWhy perturbation with Benkeser, Carone, van
  der Laan & Gilbert (2017), Theorem 1. The theorem supplies the complete-outcome corrected curve
  and remainder conditions. Existing exact-law tests transport every term to the fixed tilt
  $dP_w=w\,dP/E_P[w]$. The joint law factorizes as $P_w\times\Phi$ because each weight depends only
  on its observed row. The pinned R `drtmle` 1.1.2 implementation accepts no observation-weight
  argument, so it supplies no weighted comparison.
- Fixed-weight binary C-TMLE surfaces compose the DoWhy perturbation with van der Laan & Gruber
  (2010), Sections 2, 5.1, and 6. Those sections define C-TMLE for a generic law. The selector
  scores each candidate path against the empirical outcome loss and an optional influence-curve
  penalty. The package substitutes the tilted law $dP_w=w\,dP/E_P[w]$ and its normalized
  empirical measure.
  `tests/unit/test_simulated_confounding.py::test_fixed_weight_ctmle_selector_components_recompute_from_the_refit`
  rebuilds the weighted loss, penalty, fold assignment, and nested cross-validated risk of a
  refit. Component mutations in the same module strip the weights from one production method and
  move that risk. No exact-law transport test covers this composition. The outcome-adaptive route
  uses the same measure for its categorical mechanism. The pinned R `ctmle` and archived `ctmle3`
  sources have no weighted comparison, so this composition claims no parity with either.
- Hartman & Huang (2024), [*Sensitivity Analysis for Survey
  Weights*](https://doi.org/10.1017/pan.2023.12), *Political Analysis* 32(1):1-16.
  Their method bounds the bias from a confounder that the weighting model omits. It supplies a
  bound, a robustness value, and a benchmarking procedure. This surface supplies none of those, so
  it answers a different sensitivity question.
- Hu, Zou, Gu, Ji, Lopez & Kale (2022), [*A flexible sensitivity analysis approach for unmeasured
  confounding with multiple treatments and a binary outcome with application to SEER-Medicare lung
  cancer data*](https://doi.org/10.1214/21-AOAS1530), *The Annals of Applied Statistics*
  16(2):1014–1037, DOI 10.1214/21-AOAS1530. The paper supplies a Monte Carlo sensitivity analysis
  for multiple treatments and a binary outcome. It keeps the treatment fixed and adjusts the
  potential outcomes through confounding functions, inside a Bayesian nested multiple-imputation
  procedure. The abstract on page 1014 states that scope. The method encodes the impact of
  unmeasured confounding on the potential outcomes, and adjusts the estimates of causal effects.
  The paper does not perturb the treatment variable. It therefore supplies no category-valued
  latent treatment perturbation and no refit law for this surface. Cited by roadmap item F8.
- Ou, Tang & Chang (2023), [*Sensitivity Analysis of Causal Treatment Effect Estimation for
  Clustered Observational Data with Unmeasured Confounding*](https://arxiv.org/abs/2301.12396v1),
  arXiv:2301.12396v1. The paper models unmeasured cluster effects through mixed models. It derives
  a bias correction from those models. That construction does not perturb the treatment and the
  outcome before a complete TMLE refit. It does not choose between a row-level, a cluster-level,
  and a mixed latent cause for this surface. Cited by roadmap item F9.

## Negative controls

- Penning de Vries & Groenwold (2023),
  [*Negative controls: Concepts and caveats*](https://doi.org/10.1177/09622802231181230),
  *Statistical Methods in Medical Research* 32(8):1576–1587. Not read here. It is cited for the
  standard caveat rather than for a derivation. A negative control has limited sensitivity and
  specificity for unmeasured confounding, and a null association does not establish the null.

## Refutation

- Sharma & Kiciman (2020), [*DoWhy: An End-to-End Library for Causal
  Inference*](https://arxiv.org/abs/2011.04216), arXiv:2011.04216. The paper defines the four-stage
  framework and describes outcome, bootstrap, and unobserved-confounder refutations. It supports
  the shipped generated-outcome and bootstrap measurement-error refutations. The sensitivity
  section records its separate simulated common-cause role.
- The maintained DoWhy dummy outcome refuter, source at commit
  [`2116d5c`](https://github.com/py-why/dowhy/blob/2116d5cbace5a057937e03b2efba95c13140cc4c/dowhy/causal_refuters/dummy_outcome_refuter.py).
  It supplies secondary control-flow evidence for independent noise and `f(W) + h(A)`. Two of its
  choices are not adopted. The first is a normal rule below 100 draws, in
  `perform_normal_distribution_test` in `dowhy/causal_refuter.py`. The second is the absence of any
  failure policy: the pinned file has no `try` block, and its refits run under `joblib.Parallel`,
  so one failed refit aborts the refutation. `cleverly` retains each failed refit as a
  `ReplicationFailure` record and fails the refutation under the recorded rule.
- The maintained DoWhy bootstrap refuter, source at commit
  [`2116d5c`](https://github.com/py-why/dowhy/blob/2116d5cbace5a057937e03b2efba95c13140cc4c/dowhy/causal_refuters/bootstrap_refuter.py).
  Its measurement-error control flow supplies secondary implementation evidence. The package
  does not copy its dtype check, categorical probability reuse, or shared simulation seed. The
  source locator is an implementation reference and not acceptance evidence.

## Multiple testing

- Benjamini & Hochberg (1995), [*Controlling the False Discovery Rate: A Practical and
  Powerful Approach to Multiple Testing*](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x),
  DOI 10.1111/j.2517-6161.1995.tb02031.x.

## Doubly-robust inference (`DRTMLE`)

The variant rests on three sources. The first two give the estimating equations. The third gives
the implementation for the influence curve. The papers are not kept in the repository, so each
entry below gives section, equation, theorem, or page locators that resolve without a local copy.

Two names are inverted between the papers and the `benkeser/drtmle` source. This package's
`ReducedSet.gr1` is R's `grn2`, and `gr2` is R's `grn1`. That is the single easiest thing here to
transcribe backwards. No R enters this repository or its CI, so the pinned source is provenance
rather than a comparison target.

- van der Laan (2014), *Targeted estimation of nuisance parameters to obtain valid statistical
  inference*, International Journal of Biostatistics 10(1):29–57. **Theorem 3** is the bivariate
  construction's binary targeted recursion and asymptotic-linearity result; its proof supplies the
  corrected influence function and product-remainder conditions. Read first-hand for the bivariate
  implementation; the later univariate result remains the default.
- Benkeser, Carone, van der Laan & Gilbert (2016), *Doubly-robust Nonparametric Inference on the
  Average Treatment Effect*, U.C. Berkeley Division of Biostatistics Working Paper Series, paper
  356. Read first-hand. §3.1 and equation (2) are p. 9; §3.2, Theorem 1 and the recursive algorithm
  are pp. 10–11; appendix A is pp. 19–20, appendix B p. 21, appendix C pp. 21–22. Theorem 1, pp.
  10–11, states nuisance-limit, score, convergence and remainder conditions for asymptotic
  linearity. It is not an unconditional guarantee.
- Benkeser, Carone, van der Laan & Gilbert (2017),
  [*Doubly robust nonparametric inference on the average treatment effect*](https://pmc.ncbi.nlm.nih.gov/articles/PMC5793673/),
  *Biometrika* 104(4):863–880. The **published** version of the above, and authoritative wherever
  the two differ. Read first-hand. Theorem 1 states the score and remainder conditions for
  asymptotic linearity. Section 4 supplies the binary simulation law and its three nuisance
  scenarios.
- Benkeser & Hejazi (2023), *Doubly-Robust Inference in R using `drtmle`*, Observational Studies
  9(2):43–78. Read first-hand. Multi-level treatments are §4.6, pp. 66–67; cross-validated nuisance
  regression is §4.7, p. 69. The package vignette describes both reduced-regression choices for
  user-specified levels of a discrete treatment, and the pinned source implements the bivariate
  branch inside the same per-level loop; that is the provenance for the multi-arm extension, not
  an expansion of van der Laan's binary theorem.
- Díaz & van der Laan (2017), *Doubly robust inference for targeted minimum loss-based estimation
  in randomized trials with missing outcome data*, Statistics in Medicine 36:3807–3819, DOI
  [10.1002/sim.7389](https://doi.org/10.1002/sim.7389). Read first-hand. §2.1 states the observed-
  data model and EIF; equation (6) defines the reductions; Theorems 1–2 and equations (11)–(13)
  give the corrections and targeting algorithm. This establishes the missing-outcome construction
  for randomized treatment; it does not establish the observational-treatment or missing-treatment
  compositions exposed by the canonical package, and it explicitly leaves cross-validation to
  future work.

The `benkeser/drtmle` R package supplies implementation provenance and a bounded numerical
comparison. Agreement with it does not establish the theorem or truth-based validity. The
registered study asks those questions separately. The inspected source is pinned at
[`538a3a2`](https://github.com/benkeser/drtmle/tree/538a3a264c1ca984b6d88978ca7f96165f43152c):
`R/estimate.R` loops the reductions over treatment levels and constructs a compatible initial
mechanism; `R/fluctuate.R` applies independent one-vs-rest mechanism fluctuations.

## Proposed methods on the roadmap

These sources locate methods in the [main roadmap](roadmap.md#main-roadmap). A roadmap citation
does not support a shipped claim. The readiness label on each item states the remaining source
work.

- Tan (2025), [*Sensitivity models and bounds under sequential unmeasured confounding in
  longitudinal studies*](https://doi.org/10.1093/biomet/asae044), *Biometrika* 112(1), DOI
  10.1093/biomet/asae044. The paper defines multi-period sensitivity models and observed-data
  convex representations for sharp and conservative bounds. Cited by roadmap item S4.
- van der Laan, Carone & Luedtke (2024), [*Combining T-learning and DR-learning: a framework for
  oracle-efficient estimation of causal contrasts*](https://arxiv.org/abs/2402.01972),
  arXiv:2402.01972. The paper derives EP learning for heterogeneous causal contrasts. Cited by
  roadmap item P1.
- Rust & Rao (1996), [*Variance estimation for complex surveys using replication
  techniques*](https://doi.org/10.1177/096228029600500305), *Statistical Methods in Medical
  Research* 5(3):283–310, DOI 10.1177/096228029600500305. The paper reviews jackknife, balanced
  repeated replication, and bootstrap variance methods. Cited by roadmap item X2.

- Díaz, Hejazi, Rudolph & van der Laan (2021), [*Non-parametric efficient causal mediation with
  intermediate confounders*](https://doi.org/10.1093/biomet/asaa085), *Biometrika* 108(3):627–641,
  DOI 10.1093/biomet/asaa085. A correction follows at *Biometrika* 111(2):723–726, DOI
  [10.1093/biomet/asae009](https://doi.org/10.1093/biomet/asae009). Read the correction with the
  paper. Cited by roadmap item X5.
- Rytgaard, Gerds & van der Laan (2022), [*Continuous-time targeted minimum loss-based estimation
  of intervention-specific mean outcomes*](https://doi.org/10.1214/21-AOS2114), *Annals of
  Statistics* 50(5):2469–2491, DOI 10.1214/21-AOS2114. Cited by roadmap item X6.
- Rytgaard, Eriksson & van der Laan (2023), [*Estimation of time-specific intervention effects on
  continuously distributed time-to-event outcomes by targeted maximum likelihood
  estimation*](https://doi.org/10.1111/biom.13856), *Biometrics* 79(4):3038–3049, DOI
  10.1111/biom.13856. This is the construction the `concrete` package implements. Cited by roadmap
  item X6.
- Hejazi, van der Laan, Janes, Gilbert & Benkeser (2021), [*Efficient nonparametric inference on
  the effects of stochastic interventions under two-phase sampling, with applications to vaccine
  efficacy trials*](https://doi.org/10.1111/biom.13375), *Biometrics* 77(4):1241–1253, DOI
  10.1111/biom.13375. Cited by roadmap item X7.
- van der Laan (2008), [*Estimation Based on Case-Control Designs with Known Prevalence
  Probability*](https://doi.org/10.2202/1557-4679.1114), *International Journal of Biostatistics*
  4(1), Article 17, DOI 10.2202/1557-4679.1114. Cited by roadmap item X7 for the case-control
  weighting that `TMLE.jl` implements.
