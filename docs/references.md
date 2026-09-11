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
- Zheng & van der Laan (2011), [*Cross-validated targeted minimum-loss-based
  estimation*](https://doi.org/10.1007/978-1-4419-9782-1_27), DOI
  10.1007/978-1-4419-9782-1_27. Its fold-local nuisance construction does not validate a
  stopping index selected by risk aggregated over every evaluation row.
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

- Hubbard & van der Laan (2008), [*Population intervention models in causal
  inference*](https://doi.org/10.1093/biomet/asm097), *Biometrika* 95(1):35–47.
  [Section 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC2464276/) defines intervention-minus-observed
  differences and intervention-to-observed ratios. Section 3 treats one fixed intervention and
  includes the observed-outcome contribution in Equations (5) and (6).
  `cleverly` reports the reversed difference as PAR and the ratio's complement as PAF.
  These are the existing population-intervention parameters, evaluated anew after each
  simulated-confounding perturbation. The paper supplies no simulated-confounding inference.
- Díaz, Carone & van der Laan (2016), [*Second-Order Inference for the Mean of a Variable Missing
  at Random*](https://doi.org/10.1515/ijb-2015-0031), *International Journal of Biostatistics*
  12(1):333–349 ([author manuscript](https://arxiv.org/abs/1511.08369)). Read first-hand.
  Section 2, journal pages 335–337, defines the marginal mean under missingness at random.
  Equations (1)–(3) give its expansion, efficient influence function, and exact remainder.
  Equation (4) and Steps 1–4 give the first-order TMLE. Equation (5) gives the remainder-rate
  condition for asymptotic linearity.

  The paper writes the observed data as `(W, A, AY)`, where `A` is its response indicator.
  Set its `W` to the package's `(A, W)` and its `A` to the package's `Delta`.
  The resulting target is the natural-course mean under the joint observed law of treatment and
  covariates. Its clever covariate is the inverse response score. It has no inverse treatment
  probability and needs no treatment positivity or exchangeability assumption.

  The source treats one bounded outcome mean from unweighted iid observations. It gives an
  equivalent weighted intercept fluctuation, which solves the same score equation through a
  different targeted regression. Its first-order result uses a Donsker condition. It does not
  establish weighted, clustered, stratified, cross-fitted, repeated-split, joint-target, or
  simultaneous inference. Section 5 bootstraps a second-order expansion. It does not establish
  the package's ordinary refit bootstrap.
- Missing-outcome natural-course audit (RM7, 2026-09-11): the record above supports one scalar
  missing-at-random mean from iid observations, with its first-order interval. Its stated limits
  bound the contract. It does not cover population attributable risk or population attributable
  fraction. [RM7](roadmap.md#rm7-missing-outcome-natural-course-mean) records the resulting
  contract.
- R `tmle3` at commit `ed72f8a`,
  [`R/tmle3_Spec_PAR.R`, lines 20–27](https://github.com/tlverse/tmle3/blob/ed72f8a20e64c914ab25ffe015d865f7a9963d27/R/tmle3_Spec_PAR.R#L20-L27),
  combines a treatment-specific mean with a natural-course mean.
  The file [`R/delta_functions.R`, lines 18–25 and 66–78](https://github.com/tlverse/tmle3/blob/ed72f8a20e64c914ab25ffe015d865f7a9963d27/R/delta_functions.R#L18-L78)
  gives PAR as observed-minus-intervention and PAF as one minus intervention/observed.
  Its PAF interval transforms a log contrast. The `cleverly` fraction uses the identity scale,
  including for descriptive simulated-confounding displacement. The existing canonical study
  records that interval distinction; this surface adds no interval claim.
- van der Laan (2010), [*Targeted Maximum Likelihood Based Causal Inference: Part I*](https://doi.org/10.2202/1557-4679.1211),
  DOI 10.2202/1557-4679.1211, and [*Part II*](https://doi.org/10.2202/1557-4679.1241),
  DOI 10.2202/1557-4679.1241. These provide the general causal-effect and practical TMLE
  constructions cited by the conditional-population and regimen targets. Part I,
  [Section 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC3126670/), treats the effect among the
  treated as a parameter of the outcome and treatment factors. The simulated-confounding surface
  reuses that registered ATT functional, and its relabeled ATC counterpart, on each perturbed
  empirical law. The surface reports the changing group's share, and it claims no effect in the
  original treated group held fixed.
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
  10.2202/1557-4679.1181. Section 2.4 selects candidate depth by cross-validated targeted loss.
  Theorem 4 assumes the expansion that defines the adaptive-mechanism influence contribution.
  It does not derive that expansion for the candidate-depth selector. Section 4.1 proposes a
  parametric delta method for a selected parametric mechanism model. Section 4.3 records random
  cross-validation over-selection and leaves its irregularity as an open area.
- van der Laan (2014), [*Targeted estimation of nuisance parameters to obtain valid statistical
  inference*](https://doi.org/10.1515/ijb-2012-0038), DOI 10.1515/ijb-2012-0038. Section 5.4
  targets both nuisance estimators and solves extra score equations for a binary
  treatment-specific mean. The shipped selector does not perform those targeting steps.
- Gruber & van der Laan (2010), [*An application of collaborative targeted maximum likelihood
  estimation in causal inference and genomics*](https://pmc.ncbi.nlm.nih.gov/articles/PMC3126668/),
  DOI 10.2202/1557-4679.1182.
- Ju, Chambaz & van der Laan (2018), [*Collaborative targeted inference from continuously indexed
  nuisance parameter estimators*](https://arxiv.org/abs/1804.00102). Read first-hand. Theorem 1
  permits an additional influence contribution along a twice-differentiable nuisance path for a
  binary scalar target; Lemma 2 makes that contribution zero in an important correct-outcome,
  product-rate regime. Separately, the estimator solves extra derivative-score equations and uses
  an undersmoothing condition. It does not cover the package's discrete selector paths, nested
  folds, or joint target vectors.
- Ju, Gruber, Lendle, Chambaz, Franklin, Wyss, Schneeweiss & van der Laan (2019), [*Scalable
  collaborative targeted learning for high-dimensional data*](https://pmc.ncbi.nlm.nih.gov/articles/PMC6086775/),
  DOI 10.1177/0962280217729845. Sections 4.1 through 5 define the general, greedy, and preordered
  binary-ATE paths. Section 7.4 forms each interval from the ordinary efficient influence curve.
- Ju, Benkeser & van der Laan (2020), [*Robust inference on the average treatment effect using the
  outcome highly adaptive lasso*](https://arxiv.org/abs/1806.06784), *Biometrics* 76(1):109-118,
  DOI 10.1111/biom.13121. This is another adaptive-propensity construction with explicit
  estimating equations and inference. It does not cover a generic selected candidate depth.
- Benkeser, Cai & van der Laan (2020), [*A nonparametric super-efficient estimator of the average
  treatment effect*](https://doi.org/10.1214/19-STS735), DOI 10.1214/19-STS735
  ([preprint](https://arxiv.org/abs/1901.05056)). Theorem 1 derives asymptotic linearity with the
  usual TMLE curve evaluated at the adaptive propensity limit for a treatment-specific mean.
  Its six regularity conditions require the relevant score to be negligible, quarter-rate
  convergence of the outcome and adaptive-propensity estimators, influence-curve convergence,
  smoothness, a higher-order remainder, and an empirical-process condition. Section 3.1 gives a
  cross-validated variance construction in which each fold's outcome and adaptive propensity are
  both learned without that fold's rows. No separate first-order generated-design term appears in
  the theorem's influence function.
  Appendix D explicitly constructs the binary ATE with one propensity fit on
  `(Qbar(1, W), Qbar(0, W))` and sketches a cross-validated C-TMLE. It does not derive treatment
  with more than two levels, the shipped joint all-arm fluctuation and covariance, or simultaneous
  inference.
- van der Vaart, Dudoit & van der Laan (2006), [*Oracle inequalities for multi-fold cross
  validation*](https://doi.org/10.1524/stnd.2006.24.3.351), *Statistics & Decisions*
  24(3):351-371, DOI 10.1524/stnd.2006.24.3.351. The paper bounds the risk of a cross-validation
  selector by the oracle risk. A risk bound is not a limit law. It supplies no sampling
  distribution for a smooth functional of the selected estimator. It also does not show that the
  selected index converges. This entry records the abstract only.
- Shao (1993), [*Linear model selection by cross-validation*](https://doi.org/10.1080/01621459.1993.10476299),
  *Journal of the American Statistical Association* 88(422):486-494, DOI
  10.1080/01621459.1993.10476299. In its linear-model setting, ordinary
  leave-a-fixed-fraction-out cross-validation need not consistently select the true model. It
  supports requiring a separate selector-stability argument rather than inferring stability from
  prediction-risk control.
- Leeb & Pötscher (2006), [*Can one estimate the conditional distribution of post-model-selection
  estimators?*](https://doi.org/10.1214/009053606000000821), *Annals of Statistics*
  34(5):2554-2591, DOI 10.1214/009053606000000821. The paper proves that no estimator of the
  conditional distribution of a post-model-selection estimator is uniformly consistent in its
  finite-dimensional regression and subset-selection setting. The result extends to linear
  functions of that estimator. Read first-hand from arXiv:math/0702703. Nobody here has established
  that its hypotheses transfer to this candidate-path stopping index, so it is a warning against a
  universal repair rather than a theorem about this estimator or its bootstrap.
- Leeb & Pötscher (2008), [*Can one estimate the unconditional distribution of post-model-selection
  estimators?*](https://arxiv.org/abs/0704.1584), *Econometric Theory* 24(2):338-376, DOI
  10.1017/S0266466608080158. Read first-hand. The paper gives local minimax lower bounds and rules
  out locally uniform consistency for the unconditional distribution in finite-dimensional
  regression model selection. This is the closer warning for a refit bootstrap, but its transfer
  to the shipped targeted-loss selector is likewise unproved.
- Loftus (2015), [*Selective inference after
  cross-validation*](https://arxiv.org/abs/1511.08866), arXiv:1511.08866. Read first-hand. Under a
  Gaussian linear model, squared-error cross-validation combined with a training procedure whose
  selection events are linear or quadratic can itself be represented by quadratic constraints,
  allowing conditional selective tests for selected-model regression parameters. The package's
  targeted Bernoulli or squared-error losses, nuisance estimation, and causal target curve have not
  been reduced to this Gaussian quadratic-selection setting.
- Markovic, Xia & Taylor (2017), [*Unifying approach to selective inference with applications to
  cross-validation*](https://arxiv.org/abs/1703.06559), arXiv:1703.06559v3. The method conditions on
  selection and derives selectively valid pivots when the vector of model-quality criteria and the
  inference statistics are jointly asymptotically Gaussian; it also develops Gaussian-randomized
  variants. The shipped selector neither randomizes its criterion nor establishes the required
  joint Gaussian limit for its nested targeted-loss vector and joint target statistic. The paper
  is therefore a candidate repair, not a validation of the reported covariance.
- Hubbard, Kherad-Pajouh & van der Laan (2016), [*Statistical inference for data adaptive target
  parameters*](https://doi.org/10.1515/ijb-2015-0013), *International Journal of Biostatistics*
  12(1):3-19, DOI 10.1515/ijb-2015-0013. The paper defines a sample-split data-adaptive target
  parameter. Read first-hand from the authors' eScholarship manuscript. Honest splitting permits
  arbitrarily adaptive parameter generation. Theorem 3 also permits same-sample generation under
  a uniform asymptotic expansion, Donsker, and influence-curve convergence conditions. Either route
  reports a data-adaptive estimand rather than automatically recovering the fixed target reported
  by the package.
- van der Laan (L.), Carone, Luedtke & van der Laan (M.) (2026), [*Adaptive debiased machine
  learning using data-driven model selection techniques*](https://arxiv.org/abs/2307.12544),
  arXiv:2307.12544v2. Read first-hand. The framework gives regular, locally uniform inference for
  an oracle projection parameter and names collaborative targeted learning as an application.
  Its working model must approximate a fixed oracle model; a cross-validated sieve may use a
  limiting dimension only when that limit exists and is nonrandom. Under the paper's approximation
  and remainder conditions, selection is higher order and the model-based curve can be valid. The
  oracle efficiency bound is *typically*, not necessarily, smaller than the nonparametric bound.
  The shipped selector chooses one global depth from nested targeted-loss folds, and this audit has
  not proved convergence to a fixed nonrandom oracle model or the required remainders. Applicability
  could ratify the present curve or require a different one; it does not predetermine that verdict.
- Cai & van der Laan (2020), [*Nonparametric bootstrap inference for the targeted highly adaptive
  least absolute shrinkage and selection operator (LASSO)
  estimator*](https://doi.org/10.1515/ijb-2017-0070), *International Journal of Biostatistics*
  16(2), DOI 10.1515/ijb-2017-0070. Read first-hand from arXiv:1905.10299. The bootstrap fixes the
  HAL sectional variation-norm bound at or above the original cross-validation choice; it does not
  reselect that tuning rule in every bootstrap sample.
- Tibshirani, Rinaldo, Tibshirani & Wasserman (2018), [*Uniform asymptotic inference and the
  bootstrap after model selection*](https://arxiv.org/abs/1506.06266), *Annals of Statistics*
  46(3):1255-1287, DOI 10.1214/17-AOS1584. The paper proves a conservative selective bootstrap in
  fixed-dimensional regression and records failure as dimension grows. It prevents a blanket
  claim that bootstrap can never work after selection, but does not validate this estimator.
- Beran (1997), [*Diagnosing bootstrap
  success*](https://www.math.utah.edu/~davar/math6070/2013/Beran1997.pdf), *Annals of the Institute
  of Statistical Mathematics* 49:1-24, DOI 10.1023/A:1003114420352. Read first-hand. In its locally
  asymptotically normal setting, superefficiency is sufficient for failure of the intuitive
  bootstrap. This is another reason a refit bootstrap needs its own theorem, not a proof that every
  bootstrap for an outcome-adaptive estimator fails.
- Chernozhukov et al. (2018), [*Double/debiased machine learning for treatment and structural
  parameters*](https://arxiv.org/abs/1608.00060), *Econometrics Journal* 21:C1-C68, DOI
  10.1111/ectj.12097. Orthogonal-score inference permits internally selected nuisance algorithms
  when each is trained wholly outside its score-evaluation fold and meets the required rates. The
  package's global stopping depth depends on every row and is then reused for that row, so the
  simple fold-independence argument does not apply.
- Chen, Syrgkanis & Austern (2022), [*Debiased machine learning without sample-splitting for stable
  estimators*](https://arxiv.org/abs/2206.01825), arXiv:2206.01825. Algorithmic stability is an
  alternative to honest splitting for orthogonal estimators. No leave-one-out stability or
  near-tie margin has been proved for the package's discrete argmin.
- Li, Qiu, Wang & van der Laan (2025), [*Regularized Targeted Maximum Likelihood Estimation in
  Highly Adaptive Lasso Implied Working Models*](https://arxiv.org/abs/2506.17214),
  arXiv:2506.17214, and Xu & van der Laan (2026), [*Adaptive Targeted Maximum Likelihood Estimation
  of the Mean Potential Outcome under a Treatment Rule*](https://arxiv.org/abs/2605.01671),
  arXiv:2605.01671, are recent adjacent adaptive-working-model results. They use explicit working
  projections or regularized targeting rather than the shipped global candidate-depth selector.
- Hahn & Ridder (2013), [*Asymptotic variance of semiparametric estimators with generated
  regressors*](https://doi.org/10.3982/ECTA9609), *Econometrica* 81(1):315-340, DOI
  10.3982/ECTA9609. The abstract derives the first-step contribution to the influence function
  when a later regression uses an estimated regressor. This is the general accounting for a
  generated design. It does not treat a targeted plug-in estimator.
- Escanciano & Pérez-Izquierdo (2023), [*Automatic locally robust GMM with
  machine-learning-generated regressors*](https://arxiv.org/abs/2301.10643), arXiv:2301.10643.
  Read first-hand. Moment functions orthogonal to the second step remove the *indirect* first-step
  effect, not the direct effect of learning the generated regressor. The paper constructs
  corrections for both effects. Its generic GMM result does not imply that the outcome-adaptive
  TMLE's generated-design contribution is zero.
- Rotnitzky, Smucler & Robins (2021), [*Characterization of parameters with a mixed bias
  property*](https://doi.org/10.1093/biomet/asaa054), *Biometrika* 108(1):231-238, DOI
  10.1093/biomet/asaa054. The bias of the one-step estimator is the mean product of the two
  nuisance errors. This entry records the abstract only. Nobody here has checked the
  treatment-specific mean against the paper's characterization.
- Christgau, Lundborg & Hansen (2025), [*Efficient adjustment for complex covariates: Gaining
  efficiency with DOPE*](https://arxiv.org/abs/2402.12980), arXiv:2402.12980v2. Read first-hand.
  The setup allows a finite treatment set and fixed contrasts of treatment-specific means.
  Algorithm 1 learns an outcome-adapted representation, the outcome regression and propensity, and
  the final AIPW estimate on three disjoint samples. Theorem 4.3 gives ordinary-influence-function
  inference centered at the data-adaptive target conditional on the learned representation.
  Proposition 4.4 shows that root-rate representation learning can add a first-order delta-method
  variance for a fixed target. Appendix E gives a cross-fitted algorithm but states that the usual
  proof does not apply directly because the fold oracle terms depend on estimated representations.
  Thus the multi-arm setup is directly relevant, while the shipped shared-multinomial targeting and
  fixed-target joint covariance remain outside the proved result.
- Waagepetersen, Risom, Hansen & Lundborg (2026), [*Outcome-adapted Automatic Debiased Machine
  Learning*](https://arxiv.org/abs/2607.03351), arXiv:2607.03351. Read first-hand. The paper
  decomposes sampling and representation errors and proves sample-split asymptotics for an
  outcome-adapted AutoDML estimator. Its worked treatment example is a binary ATE, and the authors
  explicitly leave cross-fitting outside the theory because it introduces dependence across fold
  representations. It sharpens the generated-representation boundary but does not cover the
  shipped targeted, shared-multinomial construction.
- The R `ctmle` 0.1.2 implementation at commit
  [`18de559`](https://github.com/jucheng1992/ctmle/tree/18de559f47dc1286617350a0668391e80e1dbf7c).
  `R/ctmle_discrete.R` defines `ctmleDiscrete`, which the pinned selector-parity study calls
  through `tests/canonical/ctmle_selector/run_ctmle.R`. `R/functions_discrete.R` defines its
  `stage2` and `cv` helpers. `R/ctmle_general.R` and `R/functions_general.R` define
  `ctmleGeneral`, `stage2_general`, and `cv_general`. No argument list in those four files takes
  an observation weight, and no fit, sum, mean, or variance in them applies one. This source
  supplies no fixed-weight comparison.
  [`calc_varIC`, lines 39-60](https://github.com/jucheng1992/ctmle/blob/18de559f47dc1286617350a0668391e80e1dbf7c/R/functions.R#L39-L60)
  adds a binary logistic parameter-estimation term to each candidate's ATE curve.
  [`ctmleDiscrete`, lines 173-185](https://github.com/jucheng1992/ctmle/blob/18de559f47dc1286617350a0668391e80e1dbf7c/R/ctmle_discrete.R#L173-L185)
  forms the interval from that variance at `best_k`. Neither block differentiates the
  cross-validated stopping rule.
- The `tlverse/ctmle3` implementation at commit
  [`a4ea77b`](https://github.com/tlverse/ctmle3/tree/a4ea77b07747dfee9b2eecb9cbca88262e0559ea).
  [`LF_oat`, lines 110-133](https://github.com/tlverse/ctmle3/blob/a4ea77b07747dfee9b2eecb9cbca88262e0559ea/R/LF_oat.R#L110-L133)
  fits categorical treatment on the complete vector of treatment-specific outcome predictions.
  `R/tmle3_Spec_TSM_all.R` requests all treatment-specific means. This is the implementation
  source for `CTMLE(strategy="oat")`, not a published inference derivation.
- The `benkeser/drtmle` 1.1.2 implementation at commit
  [`538a3a2`](https://github.com/benkeser/drtmle/tree/538a3a264c1ca984b6d88978ca7f96165f43152c)
  supplies multi-arm provenance for the same generated design. Its `adapt_g` option is written by
  an author of Benkeser, Cai and van der Laan (2020).
  [`R/estimate.R`, lines 70-92](https://github.com/benkeser/drtmle/blob/538a3a264c1ca984b6d88978ca7f96165f43152c/R/estimate.R#L70-L92)
  replaces the covariate frame with one column of outcome predictions for each level in `a_0`.
  [`R/drtmle.R`, line 209](https://github.com/benkeser/drtmle/blob/538a3a264c1ca984b6d88978ca7f96165f43152c/R/drtmle.R#L209)
  defaults `a_0` to every observed level, so that design is not restricted to two arms.
  [`R/drtmle.R`, lines 256-262](https://github.com/benkeser/drtmle/blob/538a3a264c1ca984b6d88978ca7f96165f43152c/R/drtmle.R#L256-L262)
  turns the extra targeting off under `adapt_g`. The reported covariance at
  [`R/drtmle.R`, lines 775-820](https://github.com/benkeser/drtmle/blob/538a3a264c1ca984b6d88978ca7f96165f43152c/R/drtmle.R#L775-L820)
  is the sample covariance of the influence-curve matrix divided by `n`. The string `adapt_g`
  appears in no
  variance file. `R/inf_functions.R` and `R/confint.R` each contain zero occurrences. This source
  supplies no generated-design variance term.
- C-TMLE inference source audit (RM5, 2026-09-11): no reviewed source derives the exact
  asymptotic law of the shipped selector or the full multi-arm outcome-adaptive construction. The
  binary scalar outcome-adaptive construction has a positive theorem after the fold-nesting
  correction described below. The shipped binary vector is a finite-dimensional extension whose
  exact scalar expansions are not stated in the paper. The contracts carry those separate
  verdicts.

  The selector path has outer nuisance, selection, and inner selection-training folds. Its
  pointwise and simultaneous intervals use a plug-in curve that treats the selected candidate as
  fixed; no conditional-coverage claim follows. The oracle inequality bounds risk and supplies no
  limit law. Leeb and Pötscher (2006) give a nonuniformity result in a different regression model,
  while Markovic, Xia and Taylor (2017) require a joint Gaussian selection-and-statistic limit that
  has not been shown here. Adaptive debiased machine learning supplies the closest positive
  framework, but the package's selected depth has not been proved to approximate its required
  fixed nonrandom oracle model. [F18](roadmap.md#f18-selector-path-c-tmle-inference) records the
  resulting proof obligations.

  The outcome-adaptive path selects no candidate. Benkeser, Cai and van der Laan (2020) prove a
  binary treatment-specific-mean result, explicitly construct the binary vector-design ATE, and
  outline a cross-validated form.

  DOPE proves ordinary-curve inference for finite treatment sets after a three-way split,
  conditional on the learned representation, and also identifies a possible first-order
  representation term for a fixed target. Its cross-fitted extension is an algorithm without a
  completed proof. Outcome-adapted AutoDML likewise proves sample-split, not cross-fitted,
  inference. These results narrow the remaining multi-arm gap but do not cover the shipped
  shared-multinomial, jointly targeted construction.
  [F19](roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) records what stays open
  past those regimes.

  The 2026-09-11 corrective pass read the theorem text available from arXiv and PubMed Central for
  van der Laan and Gruber (2010), Ju, Chambaz and van der Laan (2018), Benkeser, Cai and van der
  Laan (2020), both Leeb and Pötscher impossibility papers, Loftus (2015), Markovic, Xia and Taylor
  (2017), Cai and van der Laan (2020), adaptive debiased machine learning v2, DOPE v2,
  outcome-adapted AutoDML, and Escanciano and Pérez-Izquierdo (2023). Entries still marked
  abstract-only are context, not premises of the F18 or F19 verdicts. Source code was inspected at
  the pinned commits above.

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
  Section 3.6, journal pages 160-161, regresses each targeted later-node prediction at the next
  node. Section 3.7, journal page 162, gives inference for the completed estimator.
- Lendle, Schwab, Petersen & van der Laan (2017), *ltmle: An R Package Implementing
  Targeted Minimum Loss-Based Estimation for Longitudinal Data*, DOI
  [10.18637/jss.v081.i01](https://doi.org/10.18637/jss.v081.i01). Section 2.4, page 6,
  carries an updated later-node regression into each earlier regression. Section 3.3, page 13,
  defines `gbounds` as bounds on estimated mechanism components.
- Schomaker, Luque-Fernandez, Leroy & Davies (2019), [*Using Longitudinal Targeted Maximum
  Likelihood Estimation in Complex Settings with Dynamic Interventions*](https://doi.org/10.1002/sim.8340),
  *Statistics in Medicine* 38(24):4888-4911. Sections 3.4 and 4.3.2 define the complete
  sequential LTMLE algorithm. Section 5.6 and Table 2 compare complete analyses at bounds 0.01
  and 0.05. Appendix B, pages 24-25, makes the later targeted prediction the earlier response.
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
  remains binary implementation provenance.
  [`CalcCumG`, lines 2009-2011](https://github.com/joshuaschwab/ltmle/blob/338c029dae9692ef20714125773da7037688993b/R/ltmle.R#L2009-L2011)
  bounds raw cumulative mechanism products.
  [`FixedTimeTMLE`, lines 748-782](https://github.com/joshuaschwab/ltmle/blob/338c029dae9692ef20714125773da7037688993b/R/ltmle.R#L748-L782)
  fits and updates each regression, then carries `Qstar` backward. Changing the bound therefore
  requires every earlier bound-dependent regression to run again.
  The `tmle3` snapshot at
  [`ed72f8a`](https://github.com/tlverse/tmle3/tree/ed72f8a20e64c914ab25ffe015d865f7a9963d27)
  (`LF_static`, `Param_TSM`, `Param_MSM`) confirms equality-density/static-intervention indexing
  but is not a longitudinal categorical oracle. The companion Poulos repository was inspected at
  [`0e8dc6e`](https://github.com/jvpoulos/multi-ltmle/tree/0e8dc6eca1012e5a3eab7aa80b772cf432b8f032).
- Longitudinal truncation audit (2026-09-10): Schomaker et al. compare separate fixed-bound LTMLE
  analyses. They do not define a post-fit shortcut that reuses earlier outcome predictions. The
  reviewed sources support a descriptive grid that reuses raw mechanism predictions and reruns the
  full recursion. They do not support a preferred bound, simultaneous curve inference,
  selected-bound inference, or a data-adaptive longitudinal bound.
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
- The same DoWhy function branches on treatment and outcome variable types only. It has no branch
  for a response indicator, intermediate, cluster identifier, longitudinal history, or estimated
  weight model. These omissions locate the fit-wide refusal boundary. They do not prove that no
  other method can supply a law.
- Díaz and van der Laan (2017), under [DR-TMLE](#doubly-robust-inference-drtmle), define the observed-data
  model and estimator for randomized trials with missing outcomes. Their response mechanism is
  part of missing-at-random identification. They do not define a response-indicator perturbation
  for a simulated common cause. Holding the indicator fixed after treatment changes is therefore
  not licensed by that estimator source.
- Tan (2025), under [proposed methods](#proposed-methods-on-the-roadmap), does not define a
  time-indexed latent perturbation followed by a complete LTMLE refit. That omission places
  longitudinal replay in [F13](roadmap.md#f13-longitudinal-simulated-confounding-replay). The
  entry under proposed methods records what the paper does supply.
  [F16](roadmap.md#f16-longitudinal-sensitivity-bound-estimation) states the contracts it leaves
  open.
- The same pinned DoWhy refit preserves `effect_modifier_names` and `target_units`.
  Its [propensity-weighting estimator](https://github.com/py-why/dowhy/blob/2116d5cbace5a057937e03b2efba95c13140cc4c/dowhy/causal_estimators/propensity_score_weighting_estimator.py)
  rebuilds ATT and ATC weights from the treatment in the supplied dataset. The
  `estimate_effect` method selects those weights through `target_units="att"` or `"atc"`.
  Thus observed-treatment membership follows the perturbed data. Fixed baseline strata retain
  their original membership because the perturbation changes neither their names nor values.
  These are qualitative composition conventions; they supply no sensitivity-adjusted interval.
- Sofrygin and van der Laan (2017), [*Semi-Parametric Estimation and Inference for the Mean Outcome
  of the Single Time-Point Intervention in a Causally Connected Population*](https://pmc.ncbi.nlm.nih.gov/articles/PMC5650205/),
  DOI 10.1515/jci-2016-0003. Section 3.2 gives the iid influence curve for a fixed stochastic intervention.
  The subsection titled "EIC for data-adaptive parameter indexed by fixed stochastic intervention"
  fixes $q=g^*$ and displays $q/g\,(Y-Q)+E_q Q-\Psi$.
  The regime stress surface uses this iid formula only. It claims no network inference.
  Static assignments and baseline rules are degenerate fixed densities.
  A law-dependent intervention needs the additional derivative that the paper distinguishes.
- Fixed-regime replay uses the existing [known-regime contract](technical-reference/point-treatment-tmle.md#known-regimes).
  That contract cites Díaz & van der Laan (2013) under [point treatment and stochastic interventions](#point-treatment-and-stochastic-interventions).
  It combines that functional with the pinned DoWhy perturbation and complete refit.

  Pinned `lmtp` [`R/estimators.R`, lines 109–138](https://github.com/nt-williams/lmtp/blob/f04a2b47f46debc515ce4ae778e05ebfde922c44/R/estimators.R#L109-L138)
  rebuilds the task and nuisance fits from supplied data.
  Its [`R/shift.R`, lines 1–45](https://github.com/nt-williams/lmtp/blob/f04a2b47f46debc515ce4ae778e05ebfde922c44/R/shift.R#L1-L45)
  retains the supplied policy function. `cleverly` freezes validated baseline densities across cells.
  This is qualitative replay provenance, not a canonical sensitivity comparison.
- van der Laan & Gruber (2010) is listed under [collaborative TMLE](#collaborative-tmle).
  [Section 6](https://pmc.ncbi.nlm.nih.gov/articles/PMC2898626/) defines the fixed-weight least-squares MSM projection and its normalized influence curve.
  A linear working model gives the existing identity-link projection used by the binary MSM surface.
  The surface holds its design and $h(a,W)$ fixed while refitting each perturbed dataset.
- Pinned `tmle3` [`R/Param_MSM.R`, lines 72–81 and 115–234](https://github.com/tlverse/tmle3/blob/ed72f8a20e64c914ab25ffe015d865f7a9963d27/R/Param_MSM.R#L72-L234)
  supplies arm-counterfactual, custom-weight, design, and projection implementation provenance.
  It selects a logistic working projection for binary outcomes and a Gaussian projection otherwise.
  Its conditional-probability weight default is outside this surface's fixed-weight contract.
  The binary-outcome identity-link surface therefore claims no coefficient parity with that default.
- Incremental replay uses Kennedy (2019), [Section 3.1, equation (1), and Corollaries 1–2](https://arxiv.org/html/1704.00211v3).
  The multiplier stays fixed while the intervention density depends on the treatment mechanism.
  Multiplier one leaves that mechanism unchanged.
  Pinned [`npcausal` `R/ipsi.R`, lines 123–192](https://github.com/ehkennedy/npcausal/blob/56a5ac117a29258b67b94874be662a171b5131f7/R/ipsi.R#L123-L192)
  rebuilds propensity estimates, tilt weights, and the mechanism contribution for fixed `delta.seq`.
  This is estimator provenance. Neither source supplies a simulated-confounding bound or interval.
- The MSM replay audit uses the existing fixed-weight projection and complete estimator.
  Van der Laan and Gruber's Section 6 starts with discrete treatment and a differentiable working model.
  Its worked fluctuation then assumes a coefficient-independent clever covariate.
  It does not directly validate nonlinear fixed-weight alternation or continuous numerical integration.

  Pinned `tmle3` evaluates observed-treatment designs and weights separately from counterfactual projection arrays.
  Its logistic loss and continuous integration differ from this package's fixed-grid least-squares construction.
  No nonlinear or continuous coefficient parity, new estimator, or interval claim follows from this diagnostic audit.
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

These sources locate unshipped methods in the [roadmap](roadmap.md). A roadmap citation does not
support a shipped claim. Each item states the remaining source work.

- Tan (2025), [*Sensitivity models and bounds under sequential unmeasured confounding in
  longitudinal studies*](https://doi.org/10.1093/biomet/asae044), *Biometrika* 112(1), DOI
  10.1093/biomet/asae044. The paper treats a terminal outcome under binary, static longitudinal
  strategies. Section 2 states that it deals with static strategies only. It defines population
  sensitivity models and observed-data convex representations for sharp or conservative bounds.
  It keeps the primary, joint, and product models separate. The primary and joint models share the
  same sharp mean bounds. The product representations are conservative except in the restricted
  cases the paper states. Section 6.2 treats sharp contrasts over multiple strategies in two
  periods only. Section 3.2, after Proposition 1, sketches sample analogues under linear
  parameterizations. That sketch assumes consistent ICE or IPW estimation for a transformed
  outcome. The same passage leaves estimation with sample data to future work. It also supplies
  one large-sample property. A misspecified parameterization of the quantile functions still
  returns a conservative bound. Section 6.3 keeps the paper at the population level. It asks for
  specialized algorithms, and for sample estimation of the ICE and IPW functionals. The paper
  reports no sampling inference for any bound. Cited by future investigation
  [F16](roadmap.md#f16-longitudinal-sensitivity-bound-estimation), which states the missing
  contracts.
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
