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
  DOI 10.2202/1557-4679.1043. Read first-hand. Sections 1 and 3 treat a pathwise differentiable
  Euclidean parameter and a submodel whose scores span every influence-curve component. Section
  4.2, article pages 24–25, gives the expansion, remainder, Donsker, and $L_2$ conditions. Section
  5.1, article pages 27–28, treats CAR-censored parameters. Its Equation (11) is the targeted influence-curve
  equation after the observed-data curve is known. It does not derive RM8's missing-outcome curve
  or response transfer.
- Díaz Muñoz & van der Laan (2012), [*Population Intervention Causal Effects Based on
  Stochastic Interventions*](https://doi.org/10.1111/j.1541-0420.2011.01685.x), DOI
  10.1111/j.1541-0420.2011.01685.x.
- Gruber & van der Laan (2010), *A targeted maximum likelihood estimator of a causal effect on a
  bounded continuous outcome*. Read first-hand in the
  [author working paper](https://biostats.bepress.com/ucbbiostat/paper265), U.C. Berkeley
  Division of Biostatistics Working Paper 265. That file prints no page numbers, so the locators
  below are its PDF pages. This audit did not compare them with the journal version.

  | locator | content |
  | --- | --- |
  | Section 2, PDF page 5 | assumes a known interval $Y \in [a, b]$ and maps $Y$ linearly to $[0, 1]$ |
  | Lemma 1, PDF page 7 | shows that the logistic loss identifies the conditional mean of the transformed outcome |
  | Section 3, PDF page 9 | sets $a$ and $b$ to the sample minimum and maximum, because the simulated outcome is unbounded |
  | Section 4, PDF pages 13–14 | recommends the observed minimum and maximum when no bounds are known a priori |

  The known-interval result supports a fixed `q_bounds` transform. Sections 3 and 4 state the
  data-derived bounds as a practice, and they give no separate derivation for that choice.
- Smith et al. (2025), [*Performance of Cross-Validated Targeted Maximum Likelihood
  Estimation*](https://doi.org/10.1002/sim.70185), *Statistics in Medicine* 44(15-17):e70185,
  DOI 10.1002/sim.70185. Read first-hand. Section 2.1 describes a transform based on the observed
  minimum and maximum. Section 6 states that its simulations used binary outcomes only. The
  authors leave continuous-outcome performance for further research. The paper does not derive
  an interval result for a scale computed from every row before cross-fitting.
- Gruber & van der Laan (2012),
  [*tmle: An R Package for Targeted Maximum Likelihood Estimation*](https://doi.org/10.18637/jss.v051.i13),
  *Journal of Statistical Software* 51(13), DOI 10.18637/jss.v051.i13. Section 2.1,
  page 5, defines marginal risk and odds ratios from two counterfactual risks. Section 2.7,
  page 10, reports intervals for both ratios on the log scale. Appendix A, page 34, gives
  their log-scale influence curves.

  The same article gives the missing-outcome construction for two arms. Page numbers are the
  journal's printed pages.

  | locator | content |
  | --- | --- |
  | Section 1, page 4 | states that the FAQ estimates a categorical treatment through the marginal mean under each level |
  | Section 2.3, pages 6–7 | observed data with a response indicator $\Delta$, the clever covariate $I(A = a, \Delta = 1) / g(A, \Delta \mid W)$ for a generic arm $a$ with $g = g_A g_\Delta$, and an outcome regression fit on complete observations |
  | Section 3.2, page 16 | warns that the default observed-range bounds can fail under missing outcomes, and encourages bounds from domain knowledge |
  | Section 3.3, page 17 | applies the propensity bound to the product $g_A g_\Delta$ when two arms have missing outcomes |
  | Section 6, pages 28–29 | states that the package takes only a binary treatment; for a categorical treatment, it estimates one level at a time as a population mean with the binary indicator $\Delta_a = I(A = a)$, in an example whose data have no missing outcomes; the example gives no variance for a contrast |
  | Appendix A, page 34 | states joint normality with the covariance matrix of a possibly multi-dimensional parameter, and estimates each variance as $\mathrm{VAR}(IC)/n$ |

  The printed Appendix A display for the ATE curve ends in $\bar Q_0(1, W) - \bar Q_0(A, W)$. The
  ATE curve needs $\bar Q_0(0, W)$ in the second term. Line 1606 of the `tmle` 2.1.1 source uses
  `Q[,"Q0W"]`, which is the corrected term.
- CRAN `tmle` 2.1.1, source at commit
  [`f8d88a0`](https://github.com/cran/tmle/tree/f8d88a07a3d25c96688221b384043eef7a31fe68).
  [`R/tmle.R`](https://github.com/cran/tmle/blob/f8d88a07a3d25c96688221b384043eef7a31fe68/R/tmle.R)
  normalizes `obsWeights`. It routes them through outcome and treatment fitting, targeting,
  plug-in evaluation, and influence-curve calculations. This source supports the weighted
  ordinary-TMLE refit. It does not implement a simulated common-cause surface.

  The same file has a population-mean path for missing outcomes. Lines 1737–1739 set a missing or
  all-zero `A` to one. Lines 170–172 then label `EY1` as the population mean. Line 2008 keeps
  the respondent rows, and lines 2020–2021 fit the fluctuation on them with `subset=keep`. Line 1564
  gives the influence curve `Delta / pDelta1 * (Y - QAW) + Q1W - mu1`.

  The table lists the lines that the audit read, from supplied predictions and scaling through the
  curves, variances, and intervals.

  | lines | behavior |
  | --- | --- |
  | 1091–1127 | `.initStage1` bounds the outcome and a supplied `Q`, then maps both to `[0, 1]` |
  | 1095–1096 | a missing `Qbounds` defaults to the respondent range, widened by one percent of each endpoint's magnitude |
  | 1120 | `ab <- range(Ystar, na.rm=TRUE)` sets the scale from every `Y` value that is not `NA`, also when `Qbounds` is supplied; a non-`NA` `Y` on a row with `Delta = 0` enters the range |
  | 1113, 1123 | `.initStage1` clips a supplied `Q` to `Qbounds` and then rescales it by `ab` |
  | 1192–1193 | a supplied `Q` skips the outcome fit and records `"user-supplied values"` |
  | 1328–1330 | line 1328 bounds the initial predictions; `qlogis` applies only when `maptoYstar` is set or the family is binomial |
  | 1841–1848 | a scalar `gbound` becomes `c(gbound, 1)`, also with two arms |
  | 1390–1521 | `estimateG` fits a mechanism; line 1396 skips the fit and lines 1505–1506 record `"user-supplied values"` for a supplied mechanism |
  | 1710–1711 | the defaults are `alpha = 0.9995` and `target.gwt = FALSE` |
  | 1747–1748 | a missing `gbound` defaults to `5/sqrt(sum(Delta))/log(sum(Delta))` |
  | 1901–1902 | two-arm missing-outcome fits bound the product of the treatment and response probabilities |
  | 2014–2016 | without `target.gwt`, the clever covariates are `A/g1W.total` and `(1-A)/g0W.total` |
  | 1592, 1606 | the two-arm `EY1` and ATE curves, which carry the `Delta` factor |
  | 1596, 1610 | each variance is `var(IC)/n` |
  | 1599, 1613 | a binary outcome bounds the `EY1` interval to `[0, 1]` and the ATE interval to `[-1, 1]` |
  | 1623–1624, 1635–1636 | the log-RR and log-OR curves |
  | 1650 | the returned list holds the five curves |
  | 2053–2061 | the `evalATT` path sets the ATT response probabilities; lines 2056–2058 use marginal means when more than 95% of the ATT rows respond, and only the else branch refits intercepts |

  An arm-indexed mean for three or more arms can use the population-mean path once for each arm
  `a`. The table lists the lines of that route.

  | lines | behavior |
  | --- | --- |
  | 1737–1739 | a missing or all-zero `A` becomes one, so a constant `A` selects the population-mean path; pass `Delta = 1{A = a} Delta` |
  | 1101–1104 | a supplied two-column `Q` becomes `QAW`, `Q0W`, and `Q1W`; pass `cbind(Q(a, W), Q(a, W))` |
  | 1505–1511 | a supplied mechanism records `"user-supplied values"`; pass `g1W` of ones with `pDelta1 = g_a pi_a`, or `g1W = g_a` with `pDelta1 = pi_a` |
  | 1814–1818 | a two-column `pDelta1` is duplicated into four named columns |
  | 1901 | the product of `g1W` and `pDelta1` is bounded as `g1W.total` |
  | 2008, 2015, 2020–2021, 2023, 2027 | the respondent rows, the covariate `A/g1W.total`, the fluctuation fit, and the update |
  | 2044–2045 | `calcParameters` evaluates the targeted predictions |
  | 1560–1568 | a constant `A` gives `EY1` and `IC.EY1 = Delta/pDelta1*(Y-QAW) + Q1W - mu1`, with variance `var(IC.EY1)/n` |
  | 2198 | the returned list holds `estimates` |

  The K returned `IC.EY1` curves share rows, so they give the joint covariance and each reference
  contrast. With a continuous outcome mapped to `[0, 1]`, line 1120 reads every non-`NA` `Y`. The
  route therefore sets `Y` to `NA` on rows with `A != a`.

  JSS page 16 states that the function "ignores the Y value for observations having Δ = 0". Line
  1120 still reads that value when it is not `NA`.

  `cleverly` bounds the response probability and the propensity separately
  (`src/cleverly/estimators/targeting.py:333-337` and `349`). The R product bound at lines
  1901–1902 is a different rule. A paired comparison must choose a `gbound` below every supplied
  product. It must also check that the package bounds clip no supplied value.

  The registered stacked MAR natural-course study supplies the stitched out-of-fold `Q(A,W)` and
  `pDelta1(A,W)` values.
  Its adapter duplicates each realized-arm prediction and passes a constant synthetic `A`.
  It carries the original treatment beside `W`, preserving the conditioning set represented by
  each supplied prediction. This selects the population-mean path without fitting a treatment
  mechanism.

  The R path uses the centered, `n - 1` divisor sample variance `var(IC) / n` and bounds its
  binary-outcome interval to `[0, 1]`. `cleverly` uses the uncentered second moment
  `mean(IC^2) / n` and an unbounded Wald interval. The registered comparison keeps both native
  rules. Its interior law does not activate the R interval bound.

  The registered ordinary MAR natural-course study uses the same call with the in-sample `Q(A,W)`
  and `pDelta1(A,W)` predictions of each `cleverly` fit. Its continuous law also plants the two
  `q_bounds` on two rows with `Delta = 0`, because line 1120 reads those rows. A non-cross-fitted
  natural-course fit in `cleverly` uses the centered rule, which equals the R `var(IC)/n`. The
  stacked study's `n/(n-1)` difference therefore does not apply to the ordinary study.
- Zheng & van der Laan (2011), [*Cross-validated targeted minimum-loss-based
  estimation*](https://doi.org/10.1007/978-1-4419-9782-1_27), DOI
  10.1007/978-1-4419-9782-1_27. Read first-hand in the
  [author working paper](https://biostats.bepress.com/ucbbiostat/paper273/), Working Paper 273.
  Page numbers below are the working paper's printed pages, which are the PDF pages minus two. This
  audit did not compare them with the Springer chapter. Sections 2 and 2.1, pages 2–6, train both
  the relevant initial estimator and its nuisance on each training
  complement. They choose one fluctuation by pooled validation loss and average the fold plug-ins.
  Section 2.1 leaves a linear empirical-distribution component untargeted. Theorem 2, pages
  14–18, gives the partial-targeting expansion and conditions without a complexity restriction on
  the initial nuisance classes. Its split vector is an external random draw over a finite support.
  The common fluctuation must converge, and its finite-dimensional fluctuation class must satisfy
  the theorem's entropy condition.

  Set the theorem's updated component to the outcome regression, its nuisance to the response
  mechanism, and its empirical component to the distribution of `(A, W)`. The Díaz, Carone and van
  der Laan mean then has the theorem's required decomposition. Its empirical-law influence term is
  the centered targeted outcome regression. Its remaining term is the response-weighted residual.
  The Díaz exact product remainder supplies the theorem's second-order condition.

  This result supports the original pooled, fold-evaluated construction for one realized V-fold
  partition. It does not validate a stopping index selected by risk over every evaluation row, a
  separate fluctuation in each fold, or repeated-split aggregation. Its fold expectation also does
  not justify treating an arbitrarily imbalanced supplied partition as a row-weighted stacked
  sample.

  The arm-indexed audit uses four further locators.

  | locator | content |
  | --- | --- |
  | Section 2, page 2 | defines the target as $\Psi : \mathcal{M} \to \mathbb{R}^d$, so one fit can target a vector parameter |
  | Section 2, page 3 | assumes "a fixed d-variate function D(P)" as the canonical gradient, and a submodel whose score spans its components |
  | Theorem 1, page 7, and Theorem 2, pages 14–15 | assume that the split vector $B_n$ is uniformly distributed over a finite support |
  | Section 4, pages 19–21 | treats the ATE and maps an outcome in a known interval $(a, b)$ to $(Y - a)/(b - a)$ |
  | Section 4.1, Theorem 3, page 22, and Section 4.2, Theorem 4, page 31 | give the result under the squared-error loss and under the quasi-log-likelihood loss |

  Theorems 3 and 4 treat $O = (W, A, Y)$ with a binary treatment and no missing outcomes. The
  missing-outcome bounded-continuous cell therefore rests on Theorem 2 and the `(W, T_a, U)`
  reduction in the Levy entry. Section 4.2 and Theorem 4 support only the bounded-outcome
  transform.

  Theorem 2 keeps the $d$-variate setup of Section 2, so its expansion holds for the vector of K arm
  means. The joint covariance of that vector follows from the expansion. The paper does not state
  that covariance.

  The theorems place limit and rate conditions on the initial estimators. They do not prescribe how
  a training sample builds them. A pooled outcome regression evaluated at each arm, and separately
  bounded treatment and response probabilities, are therefore admissible initial estimators.
- Polley (2010), [*Super Learner*](https://escholarship.org/content/qt4qn0067v/qt4qn0067v.pdf),
  U.C. Berkeley dissertation. Read first-hand. Section 1.1.2, PDF page 16 (printed page 4),
  defines V-fold assignment independently of the learning observations. Section 2.2, Theorem 1,
  PDF pages 24–25 (printed pages 12–13), states the Super Learner oracle result for that split.
  Theorem 1 assumes bounded data, squared-error loss, and a finite grid of metalearner values.
  The dissertation's software offers `stratifyCV` for a binary outcome (Appendix A, PDF pages
  132–134). Theorem 1 does not cover that option. The package's Super Learner stratifies its
  inner folds on a classification target. In an iid point-treatment or longitudinal cross-fitted
  fit, those folds use only the outer-training rows. If the outer split reads no
  outcome, their fold assignment then reads no outer-held-out outcome. The C-TMLE selection folds
  cross the outer split ([F18](roadmap.md#f18-selector-path-c-tmle-inference)). Polley's oracle
  result does not establish a risk bound for the stratified inner folds.
- Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins (2018),
  [*Double/debiased machine learning for treatment and structural
  parameters*](https://academic.oup.com/ectj/article/21/1/C1/5056401), *The Econometrics
  Journal* 21(1):C1-C68. Read first-hand. Section 3.4 defines the coordinatewise median point.
  Equation (3.14) defines the median of the within-partition variance plus squared split
  displacement. Corollary 3.3 gives the fixed-repeat interval under the paper's DML assumptions.
  It does not establish that a targeted MAR plug-in satisfies those assumptions.

  The arXiv v7 manuscript, [arXiv:1608.00060](https://arxiv.org/abs/1608.00060), gives the split
  in Definitions 3.1 and 3.2, pages 23–24. Each says "Take a K-fold random partition" of the indices,
  with folds of size $N/K$. Neither definition stratifies the partition. This audit did not compare
  that page with the journal version.
  The [fold and outcome-scale rules](technical-reference/cv-tmle.md#fold-and-outcome-scale-rules)
  cite this wording.
- Rafi (2023), [*Efficient Semiparametric Estimation of Average Treatment Effects Under Covariate
  Adaptive Randomization*](https://arxiv.org/abs/2305.08340), arXiv:2305.08340v1. Read first-hand.
  The paper treats a binary treatment under covariate-adaptive randomization, with target
  proportions set by design. Assumption 4.2, page 20, splits each treatment-by-stratum cell into
  folds. The folds "depend only on" an independent uniform draw and the cell size. The estimator is
  a cross-fitted AIPW estimator, not a TMLE. The strata are covariate strata, not outcome strata.
  The paper therefore does not cover an observational cross-fitted fit, which is what the package
  draws its folds for. The
  [fold and outcome-scale rules](technical-reference/cv-tmle.md#fold-and-outcome-scale-rules)
  record this gap.
- Bannick, Shao, Liu, Du, Yi & Ye (2025), [*A general form of covariate adjustment in clinical
  trials under covariate-adaptive randomization*](https://doi.org/10.1093/biomet/asaf029),
  *Biometrika* 112(3), asaf029, DOI 10.1093/biomet/asaf029. This entry records the abstract only.
  The abstract states a rigorous justification of machine learning with cross-fitting for AIPW
  estimators under covariate-adaptive randomization. It supports folds drawn independently of the
  outcomes, and it covers randomized trials only. It does not cover an observational cross-fitted
  fit.
- zEpid 0.9.1, repeated cross-fit aggregation at commit
  [`16a0f96`, lines 1602-1641](https://github.com/pzivich/zEpid/blob/16a0f96f8b2c65df8715085801f21757d1478e1e/zepid/causal/doublyrobust/crossfit.py#L1602-L1641).
  The `calculate_joint_estimate` median branch implements the same point and variance
  calculation.
  It is secondary aggregation evidence and not a comparator for the complete estimator.
  In the same file, `targeting_step`
  ([lines 1644–1672](https://github.com/pzivich/zEpid/blob/16a0f96f8b2c65df8715085801f21757d1478e1e/zepid/causal/doublyrobust/crossfit.py#L1644-L1672))
  fits a separate logistic fluctuation in each split. `SingleCrossfitTMLE` calls it at lines
  1119–1122, and `DoubleCrossfitTMLE` calls it at lines 1545–1548. zEpid therefore targets inside each fold, not with one pooled fluctuation.
- Levy (2018), [*An Easy Implementation of CV-TMLE*](https://arxiv.org/abs/1811.04573),
  arXiv:1811.04573. Read first-hand. The abstract distinguishes the original fold-wise plug-in
  evaluation from a stacked validation update and whole-sample plug-in. It states exact overlap
  for a pooled treatment-specific mean. Section 3.1 supplies the asymptotic expansion after the
  pooled score is solved. The conclusion limits the exact-overlap statement to parameters whose
  clever covariates contain no empirical means.

  Relabel the response indicator as Levy's binary treatment and `(A, W)` as its covariates. Let the
  observed pseudo-outcome be `U = Delta * Y`. Then `E(U | Delta = 1, A, W)` is the Díaz outcome
  regression, and Levy's treatment-specific mean is the missing-at-random natural-course mean.
  The map from `(X, Delta, Delta * Y)` to `(X, T, U)` is a bijective relabeling of the observed
  record. Levy's parameter and influence curve do not read the outcome regression under `T = 0`.
  They therefore reduce to the Díaz observed-data parameter and curve without adding an
  identification or tangent-space result.

  The respondent-row fluctuation covariate is `1 / pi`. Its efficient-influence-function
  multiplier is `Delta / pi`. Neither contains an empirical mean.

  Levy's exact finite-sample overlap applies when the folds have equal sizes. Package-generated
  near-balanced folds instead give the stacked and original fold weights a bounded $O(1/n)$
  difference for fixed $V$. This difference is first-order negligible, but it is not zero when fold
  sizes differ. The mapping supports the implemented stacked, pooled, whole-sample construction for
  one near-balanced V-fold partition. Levy gives no repeated-split result.

  The arXiv v1 PDF prints page numbers that equal the PDF page minus two. Section 1, page 2, defines
  $B_n$ as a random split of the rows. Section 2, page 3, uses standard 10-fold cross-validation for
  an example whose clever covariate contains an empirical mean. Section 3.1, page 7, gives the
  remainder. The conclusion, page 8, states that the general class of valid parameters "remains to
  be more formally generalized".

  For an arm-indexed mean, let `T_a = 1{A = a} * Delta` and keep `U = Delta * Y`. The observed-data
  curve depends on the record only through `(W, T_a, U)`. Its multiplier is `T_a / (g_a * pi_a)`,
  and `g_a * pi_a` is `P(T_a = 1 | W)`. That multiplier contains no empirical mean, so the abstract's
  exact treatment-specific-mean overlap applies to each arm at equal fold sizes. The stacked plug-in weights fold `v`
  by `n_v / n`, while Zheng and van der Laan weight each fold by `1 / V`. The two agree exactly at
  equal fold sizes and differ by $O(1/n)$ for near-balanced folds at fixed $V$.

  Section 1, page 1, maps the model to $\mathbb{R}^d$ with a one-dimensional fluctuation parameter.
  The general definition places no binary restriction on the treatment. A joint K-dimensional
  fluctuation instead rests on the span condition in Zheng and van der Laan, Section 2, page 3. It
  also rests on arm separability. The package clever covariate has one nonzero column in each row
  for any K (`src/cleverly/fluctuation/submodel.py:463-466`). The fluctuation fits only the
  `observed` rows (`src/cleverly/fluctuation/iterative.py:523`).
- Coyle et al., R package [`tmle3`](https://github.com/tlverse/tmle3), source at commit
  [`ed72f8a`](https://github.com/tlverse/tmle3/tree/ed72f8a20e64c914ab25ffe015d865f7a9963d27).
  `R/tmle3_Update.R` selects the
  `"validation"` likelihood when `cvtmle=TRUE` and fits one update to the stacked
  validation predictions; `R/Param_TSM.R` evaluates the treatment-specific mean and its
  influence curve from those validation likelihood values. `R/delta_functions.R` defines the
  log-risk and log-odds contrasts. Used as an implementation reference, not as an oracle for the
  estimand derivation or as a moving specification. The simulated-confounding surface reports its
  ratio movement on the same log scale. Levy (2018) is the stable marker for the default stacked
  construction. It is not an exact comparator for the stacked missing-outcome contracts. A generic treatment-specific outcome learner
  can use the `Delta = 0` pseudo-outcomes when it predicts under `Delta = 1`, while the package's
  missing-outcome regression trains on respondents only.
- The fold/full prediction mechanism used by `tmle3` lives in its `sl3` dependency, pinned
  here at [`0e8f236`](https://github.com/tlverse/sl3/tree/0e8f2365bcbe54010b8120c04a7a2dcfc8119227).
  `R/Lrnr_cv.R` builds `fold_fits` and, when requested, a `full_fit`; `predict_fold(...,
  "validation")` assembles held-out predictions while `predict_fold(..., "full")` uses the
  all-training fit. This is the design source for C-TMLE's nested selection predictions.
- van der Laan & Gruber (2016), *One-step targeted minimum loss-based estimation*.

## Point treatment and stochastic interventions

- Hubbard & van der Laan (2008), [*Population intervention models in causal
  inference*](https://doi.org/10.1093/biomet/asm097), *Biometrika* 95(1):35–47.
  Read first-hand in the [author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC2464276/).
  Section 2 declares complete iid observations $O=(W,A,Y)$. Equation (2) defines
  intervention-minus-observed differences and intervention-to-observed ratios. Section 3 treats
  one fixed intervention. Equations (3)–(6) add the observed-outcome contribution to the
  intervention score. Section 4.4 treats relative risk. Section 4.6 and the Appendix give
  influence-curve covariance and asymptotic inference.

  `cleverly` reverses the paper's difference for PAR. It takes one minus the paper's ratio for PAF.
  The paper contains no response indicator, response propensity, or MAR targeting equation. It
  also supplies no simulated-confounding inference.
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
- Newey & Robins (2018), [*Cross-Fitting and Fast Remainder Rates for Semiparametric
  Estimation*](https://arxiv.org/abs/1801.09138), arXiv:1801.09138. Read first-hand. Section 3
  treats a mean with randomly missing data. Its doubly robust construction estimates the outcome
  and inverse response regressions on distinct subsamples. It is not the package's
  complement-trained pooled CV-TMLE, so it does not govern the stacked missing-outcome contracts.
- Missing-outcome natural-course implementation record (2026-09-11): the source above supports
  one scalar missing-at-random mean from iid observations, with its ordinary first-order interval.
  Its stated limits bound the
  [ordinary-TMLE contract](technical-reference/point-treatment-tmle.md#missing-outcomes-and-controlled-direct-effects).
  Zheng and van der Laan (2011) and Levy (2018) separately support the implemented stacked
  cross-fitted contract. None of these results covers population attributable risk or population
  attributable fraction.
- Source audit of stacked CV-TMLE for arm-indexed means with missing outcomes (2026-09-14): it covers
  cross-fitted TMLE means `ψ_a = E_W E(Y | A = a, Delta = 1, W)` and their contrasts.
  The implemented
  [arm-indexed stacked contract](technical-reference/point-treatment-tmle.md#stacked-cv-tmle-for-arm-indexed-targets)
  follows this audit. The support chain has five links.

  | link | source | result |
  | --- | --- | --- |
  | parameter and curve | Díaz and van der Laan (2017), Section 2.1 and Equation (1); Gruber and van der Laan (2012), Section 2.3 | one arm indicator gives the curve `1{A = a} Delta / (g_a pi_a) (Y - Q(a, W)) + Q(a, W) - ψ_a` in the nonparametric model |
  | cross-fitting | Zheng and van der Laan (2011), Section 2, pages 2–3, and Theorems 1–2 | a vector parameter with a d-variate canonical gradient, an external uniform split, and initial estimators restricted only by limits and rates |
  | remainder | Díaz, Carone and van der Laan (2016), Equation (3) | the exact product remainder supplies the second-order condition for each arm |
  | stacked evaluation | Levy (2018), abstract and conclusion | each arm's multiplier has no empirical mean, so stacked and fold-evaluated points agree at equal fold sizes |
  | contrasts | Gruber and van der Laan (2012), Appendix A; `tmle` 2.1.1 lines 1606, 1623–1624, 1635–1636 | the ATE is linear, and the log RR and log OR curves combine the two same-row arm curves |

  Díaz and van der Laan state their Assumption 2 conditional on `W`. The curve in Equation (1)
  does not use the randomization, so it applies to observational treatment. The curve reads the
  record only through `(W, T_a, U)`, which the Levy entry defines.

  The same chain covers three or more arms. The table gives the locator for each link. Page numbers
  for Díaz and van der Laan are those of the arXiv v1 author manuscript.

  | link | locator | result |
  | --- | --- | --- |
  | per-arm curve | Díaz and van der Laan, Section 2.1, page 6; Equation (1), page 10; Equation (3), pages 11–12 | `A` is "a binary treatment arm indicator", and the application has "four such indicators" |
  | per-arm application | Díaz and van der Laan, Section 2.3, page 8; Tables 1–2, pages 8–9; Figure 1, pages 9–10 | the application fits missingness in each treatment arm and estimates four arm means, one indicator at a time |
  | reduction | Díaz and van der Laan, page 25 | a referee's alternative defines `T = AM` and estimates $E\{E(Y \mid T = 1, W)\}$; the authors reject it as "unsatisfactory because it ignores intrinsic properties of the variables A and M", and the package keeps separate `g_a` and `pi_a` fits |
  | generic arm | Gruber and van der Laan (2012), Section 2.3, page 7; Section 1, page 4; Section 6, pages 28–29 | the clever covariate is written for a generic arm, and a categorical treatment is estimated one level at a time |
  | joint vector | Zheng and van der Laan, Section 2, pages 2–3 | the target lies in $\mathbb{R}^d$ with a d-variate canonical gradient; the joint covariance follows from the expansion, and the paper does not state it |
  | stacked evaluation | Levy, Section 1, page 1 | the definition places no binary restriction on the treatment; its fluctuation parameter is one-dimensional |
  | joint fluctuation | Zheng and van der Laan, Section 2, page 3; `src/cleverly/fluctuation/submodel.py:463-466`; `src/cleverly/fluctuation/iterative.py:523` | the span condition and arm separability support one K-dimensional fluctuation |
  | reference contrasts | `src/cleverly/targets/builtin.py:311-353`, `419-449`; Gruber and van der Laan (2012), Appendix A, page 34 | each `ate`, `rr`, and `or` applies the Appendix A delta-method form to two coordinates of the joint vector |

  Assumption 1 of Díaz and van der Laan, page 7, is written for two arms: "Y = M{AY1 + (1 − A)Y0}".
  The arm mean uses only the `A = 1` branch, so the per-arm identification still holds. The FAQ in
  Gruber and van der Laan (2012) has no missing outcomes and gives no contrast variance.

  No reviewed source states the K-arm contrast curve. Each contrast curve is a standard consequence
  of the joint expansion and the Appendix A delta-method form, and the papers do not state it. The
  default simultaneous band uses Rademacher multiplier draws on the centered curves of the same rows
  (`src/cleverly/methods.py:312`). Its validity follows from the joint expansion and a conditional
  multiplier central limit theorem for a fixed number of estimands. That step is also a standard
  consequence that the papers do not state.

  Two package facts bound the joint fit. Each row of the arm-indexed clever covariate has one
  nonzero column (`src/cleverly/fluctuation/submodel.py:463-466`). In `_newton_logistic`
  (`src/cleverly/fluctuation/iterative.py:696`), the gradient and Hessian separate by arm, but the
  line search and the stopping rule act on the whole vector. A joint fluctuation therefore equals
  the per-arm fits only to solver tolerance.

  The ordinary variance rule is `np.var(ic, ddof=1) / n` (`src/cleverly/inference/cluster.py:138`).
  It equals the R rule `var(IC) / n` at lines 1596 and 1610. The arm-indexed stacked fit uses
  this centered rule (`src/cleverly/estimators/tmle.py:3213-3215`). The stacked natural-course
  mean keeps the second-moment rule (`src/cleverly/inference/cluster.py:240-278`). A stacked curve
  has empirical mean zero to targeting tolerance, so the two rules agree to first order.

  The table gives the source verdict for each composition in the arm-indexed mean group. The
  [arm-indexed contract](technical-reference/scope-and-refusals.md#missing-outcome-arm-indexed-contract)
  admits or refuses each one.

  | composition | source verdict | reason |
  | --- | --- | --- |
  | binary outcome; `ey`, `ey0`, `ey1`, `ate` | source-supported | the five links |
  | binary outcome; `rr` and `or` on the log scale | source-supported | the five links and Appendix A, page 34 |
  | bounded continuous outcome with a fixed `q_bounds`; `ey`, `ey0`, `ey1`, `ate` | source-supported | Zheng and van der Laan, Theorem 2 with the `(W, T_a, U)` reduction; their Theorem 4 and Gruber and van der Laan (2010), Section 2 and Lemma 1, for the transform |
  | continuous outcome with `q_bounds=None` | no reviewed interval result | the scale depends on held-out outcomes, as the next paragraph states; the [fold and outcome-scale rules](technical-reference/cv-tmle.md#fold-and-outcome-scale-rules) require prespecified bounds |
  | three or more arms, with a binary outcome or a bounded continuous outcome with a fixed `q_bounds`; `ey`, and `ate` against the reference arm; `rr` and `or` against the reference arm for a binary outcome | source-supported | the three-arm table above, with the contrast curve as a standard consequence; the planned R comparator runs the population-mean path once for each arm |
  | simultaneous bands over the reported estimands, with the centered rule | source-supported via the vector expansion | Zheng and van der Laan, Theorem 2, give a joint expansion of the vector with curve `D`, so the estimates are jointly asymptotically normal; the default Rademacher multiplier band then follows from a conditional multiplier central limit theorem for a fixed number of estimands, a standard consequence that the paper does not state |
  | ATT or ATC | no source read | the clever covariate carries an empirical arm share, so the Levy overlap statement does not cover it |
  | more than one repeat, or fold-specific targeting | no source read | no direct interval result |
  | fold-evaluated construction | source-supported, separate estimator | Zheng and van der Laan, Sections 2 and 2.1 |
  | supplied split plan | no source read | the balance and weighting requirements are unaudited |
  | folds stratified on treatment or outcome | no reviewed result for this fit | the [fold and outcome-scale rules](technical-reference/cv-tmle.md#fold-and-outcome-scale-rules) require generated unstratified folds for every cross-fitted fit |
  | one fold with cross-fitting | not cross-fitting | one fold trains and evaluates on the same rows |
  | weights, clusters, or baseline strata | no source read | every source treats unweighted iid rows |
  | bootstrap inference | no source read | no source covers a bootstrap of this estimator |
  | linear fluctuation, one-step targeting, or weighted targeting | no source read | Zheng's Theorem 3 treats the squared-error loss without missing outcomes |
  | C-TMLE with cross-fitted arm-indexed missing outcomes | no source read | the collaborative score and selection risk are not derived |

  With `q_bounds=None`, `_scaler` builds the outcome scale from every observed outcome before the
  split (`src/cleverly/estimators/tmle.py:1212-1219`). The outcome learners train on the scaled
  outcome (`src/cleverly/estimators/_nuisance.py:1202`). A held-out outcome therefore sets the
  scale of the training fit. Gruber and van der Laan (2010) derive only the known-interval case.
  Gruber and van der Laan (2012), Section 3.2, warn about observed-range bounds under missingness.

  The audit did not examine five sibling surfaces.

  | sibling surface | fact |
  | --- | --- |
  | shift, incremental, regime, MSM, and controlled-direct-effect targets | they fit today under cross-fitting with missing outcomes, and tests cover them |
  | ordinary arm-indexed missing-outcome study | it registers only binary `ey1`, `ey0`, and `ate` |
  | registered complete-outcome stacked study | it now declares unstratified folds and `q_bounds=(0, 1)` on a bounded law, and it asserts the realized scheme before it reports a row (`tests/studies/canonical_cvtmle.py`). The gap this row recorded is closed |
  | ordinary C-TMLE with missing outcomes | the audit did not read a source for it |
  | the `learner_folds` split inside a Super Learner | it stratifies on the outcome for a binary outcome learner (`src/cleverly/learners/super_learner.py:238-241`). In an iid point-treatment or longitudinal cross-fitted fit, the [fold and outcome-scale rules](technical-reference/cv-tmle.md#fold-and-outcome-scale-rules) find that its fold assignment reads no outer-held-out outcome. That finding requires an outer split that reads no outcome. The C-TMLE selection folds cross the outer split ([F18](roadmap.md#f18-selector-path-c-tmle-inference)). The audit finds no stratified Super Learner oracle result. The implemented preflight requires two rows in each class of the role target in every training complement, for each role whose resolved learner is a package `SuperLearner` with a classification task. At that count every inner training set holds both classes. The preflight cannot see a `SuperLearner` nested inside a user pipeline, which can still fail inside the fit |
- Díaz & van der Laan (2017), [*Doubly robust inference for targeted minimum loss-based estimation
  in randomized trials with missing outcome data*](https://doi.org/10.1002/sim.7389), *Statistics
  in Medicine* 36:3807–3819 ([author manuscript](https://arxiv.org/abs/1704.01538)). Read
  first-hand. Sections 2.1–2.2 state the observed-data model, identification assumptions, and
  positivity conditions. Section 3, Equation (1), gives the treatment-by-response influence
  curve. Equation (3) gives the logistic TMLE. Conditions 1–2 and Equations (4)–(5) give its
  product-rate conditions.

  Section 2.1, page 6, defines `A` as a binary treatment arm indicator. It notes that the
  application has four such indicators. This locator comes from the arXiv v1 author manuscript, and
  this audit did not compare it with the journal version. The
  [DR-TMLE entry](#doubly-robust-inference-drtmle) for the same paper records its other locators.

  The paper fixes a randomized binary treatment. It does not establish the package's observational
  attributable-effect pair. Hubbard and van der Laan supply the complete-data treatment mechanism.
  The reviewed sources do not supply its joint observational response transfer.
- R `tmle3` at commit `ed72f8a`,
  [`R/tmle3_Spec_PAR.R`, lines 20–42](https://github.com/tlverse/tmle3/blob/ed72f8a20e64c914ab25ffe015d865f7a9963d27/R/tmle3_Spec_PAR.R#L20-L42),
  combines a treatment-specific mean with a natural-course mean. It declares $O=(W,A,Y)$.
  The file [`R/delta_functions.R`, lines 18–25 and 66–78](https://github.com/tlverse/tmle3/blob/ed72f8a20e64c914ab25ffe015d865f7a9963d27/R/delta_functions.R#L18-L78)
  gives PAR as observed-minus-intervention and PAF as one minus intervention/observed.
  Its PAF interval transforms a log contrast. The `cleverly` fraction uses the identity scale,
  including for descriptive simulated-confounding displacement. The existing canonical study
  records that interval distinction.

  [`R/Param_mean.R`, lines 49–72](https://github.com/tlverse/tmle3/blob/ed72f8a20e64c914ab25ffe015d865f7a9963d27/R/Param_mean.R#L49-L72)
  uses $Y-\psi$ and requests no update nodes. It has no response score or respondent residual.
  Therefore, `tmle3` is a complete-data reduction comparator rather than RM8 derivation evidence.
- Missing-outcome attributable-effect source audit (2026-09-12): no reviewed paper presents the
  exact MAR PAR and PAF construction. [RM8](roadmap.md#rm8-missing-outcome-attributable-effects)
  records this bounded conclusion. [F20](roadmap.md#f20-missing-outcome-attributable-effects)
  holds the missing published result.

  Díaz, Carone and van der Laan give the MAR natural-course parent, its remainder, and its rates.
  Hubbard and van der Laan give the complete-data reference parent and attributable transforms.
  Díaz and van der Laan give the treatment-by-response parent under randomized treatment. Van der
  Laan and Rubin govern finite-dimensional targeting only after the observed-data curve is known.

  None derives the package's observational natural-course and reference pair under an additional
  response process. The audit therefore cannot fix its joint curve, targeting equations,
  remainder, rate conditions, or covariance by citation. Package and comparator code do not remove
  that stop.
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
  support. Read first-hand in the [published PDF](https://epiresearch.org/wp-content/uploads/2024/04/Nonparametric-Causal-Effects-Based-on-Longitudinal-Modified-Treatment-Policies.pdf).
  Section 5.2, journal page 852, defines random, approximately equal row folds and a continuous
  outcome transform with known bounds. Section 5.2, Steps 1 to 4, journal pages 852 and 853, give
  the cross-fitted TMLE. Step 3 fits each node's fluctuation "using all the data points in the
  sample". That fit uses the out-of-fold predictions as offset and the out-of-fold cumulative
  density ratio as weight. Theorem 3, page 853, gives the TMLE limit under its stated conditions.
  The proof in Section 6 of the arXiv v4 supplement conditions on each fold's initial nuisance
  fit being fixed given its training data. The pooled fluctuation still depends on the full sample
  and is handled as a low-dimensional class. The shipped cross-fitted longitudinal estimator
  implements these steps, and the
  [cross-fitting section](technical-reference/longitudinal-tmle.md#cross-fitting-the-recursion)
  gives the mapping. The registered comparison uses `lmtp` 1.5.4, which targets within each
  training fold instead
  ([method evidence](technical-reference/method-evidence/cross-fitted-end-of-study-longitudinal-tmle.md)).
  Upstream commit [`9996b04`](https://github.com/nt-williams/lmtp/commit/9996b04dcbb3ae0b1ef8862097c36d95e9f2fcf9)
  changed that behavior on 2026-06-09. The commit calls the training-fold fluctuation a bug because
  the EIF was not mean zero. It moves the fluctuation to validation rows and adds a mean-EIF test.
  The forthcoming 1.5.5 [NEWS](https://nt-williams.r-universe.dev/lmtp/NEWS) records the same fix.
  Neither upstream version runs the pooled update, so agreement with 1.5.4 compares two
  constructions under the registered paired margins.

  The same article gives a second estimator in Section 5.3, journal page 854. The section is
  titled "Sequential Regression Estimator Using SDR Unbiased Transformations". The locators below
  come from it.

  | locator | content |
  | --- | --- |
  | Section 5.3, Step 2, page 854 | fits each pseudo-outcome regression "using only data points" that belong to one training fold |
  | Section 5.3, Step 3, page 854 | defines the estimate as an average of influence-function values, and not as a plug-in of a targeted regression |
  | Lemma 4, page 854 | gives "$2\tau$-multiply robust consistency of SDR estimator" |
  | Theorem 4, page 854 | gives weak convergence at the nonparametric efficiency bound |

  This entry paraphrases the conditions of Theorem 4 rather than quoting them. The typeset text
  omits an equals sign in one condition, and Theorem 3 on the facing page prints that sign. The
  authors remark on page 854 that Theorems 3 and 4 need the same rates for root-n consistency.
  They add that the SDR estimator does not seem to confer asymptotic advantages with respect to
  the TMLE.

  Theorem 3 certifies the pooled fluctuation that the shipped cross-fitted longitudinal TMLE runs.
  Theorem 4 certifies a fold-local recursion, and it covers the SDR estimator rather than the
  shipped one. [X4](roadmap.md#x4-sequential-doubly-robust-longitudinal-estimation) plans
  `lmtp_sdr`, and no SDR path ships today.

  [arXiv v4](https://arxiv.org/abs/2006.01366v4) matches these algorithm details. It has Sections
  5.2 and 5.3, the all-row pooling clause, the influence-function average, and Lemma 4. The
  published article supplies the journal page locators above.

  The package no longer stratifies the first-node split, which the
  [fold and outcome-scale rules](technical-reference/cv-tmle.md#fold-and-outcome-scale-rules)
  record.
- van der Laan & Rose (2011), *Targeted Learning: Causal Inference for Observational and
  Experimental Data*, Springer. Chapter 12 covers marginal structural model targets.

## Grouped folds and clustered cross-fitting

A declared `id=` makes the cluster the independent unit, and it makes the outer split a draw of
whole clusters. The sources below were read for that composition. The
[fold and outcome-scale rules](technical-reference/cv-tmle.md#fold-and-outcome-scale-rules) carry
the audit table and the verdicts. This section gives each source in full.

Each entry names the version whose locators it gives. A locator is only meaningful against a named
version. Numbering can move between an author manuscript and the published article.

The Díaz, Williams, Hoffman and Schenck (2023) entry above gives the worked case. arXiv versions
v1 and v2 number its estimator section as Section 4 and its SDR lemma as Lemma 3. Versions v3 and
v4 match the published article, with Section 5 and Lemma 4. The theorem numbers match across all
four preprint versions and the article.

Internal pagination moves with the version as well. Most entries below name an author manuscript,
a preprint or an online-first PDF as the copy this project read. Each of those paginates
differently from the published issue. So no entry in this section quotes an internal page number.
The Díaz entry sits in a different section. It quotes journal pages because it names the published
article as the copy this project read.

- Wang, Park, Small & Li (2024), [*Model-Robust and Efficient Covariate Adjustment for
  Cluster-Randomized Experiments*](https://doi.org/10.1080/01621459.2023.2289693), *Journal of the
  American Statistical Association* 119(548):2959-2971, DOI 10.1080/01621459.2023.2289693. Read
  first-hand in the NIHMS author manuscript
  ([PMC11795269](https://pmc.ncbi.nlm.nih.gov/articles/PMC11795269/)), which carries the published
  volume, issue, pages and DOI. Section 4.2 partitions "m clusters into K parts with roughly equal
  sizes (the size difference is at most 1)". That is the law a grouped `random_partition` draw
  follows. Theorem 4(b) gives asymptotic normality and the efficiency bound when the nuisance
  estimators converge at the fourth root of the cluster count. The estimator is AIPW-type with a
  cluster-level treatment under cluster randomization, and not a TMLE with a row-level treatment.
  Remark 5 says a stratified cluster randomization needs treatment balance within each stratum of
  each fold, and it defers that construction to Rafi (2023). This source supports the split law.
  It does not prove the package's estimator.
- Chiang, Kato, Ma & Sasaki (2022), [*Multiway Cluster Robust Double/Debiased Machine
  Learning*](https://doi.org/10.1080/07350015.2021.1895815), *Journal of Business and Economic
  Statistics* 40(3):1046-1056, DOI 10.1080/07350015.2021.1895815. Read first-hand in the Taylor and
  Francis online-first PDF, whose pagination differs from the issue, so the locators are sections.
  Sections 3.1.2 and 3.2 and Algorithm 1 partition the cluster indices at random into equal parts.
  Assumptions 1 and 3(i) and Theorem 1 give the result. It covers two-way clustering and linear
  Neyman-orthogonal DML scores. It has no stratification, no TMLE, and no one-way theorem.
- Park & Kang, [*A More Efficient, Doubly Robust, Nonparametric Estimator of Treatment Effects in
  Multilevel Studies*](https://arxiv.org/abs/2110.07740), arXiv:2110.07740. Read first-hand in v3,
  which version 4 supersedes. Section 3.3 splits "at the cluster level instead of at the
  individual-level". Supplement A.1, Theorem A.1 under its modified condition (M1), gives the
  result for independent clusters of bounded size. It weights cluster averages, which equals row
  weighting only at equal cluster sizes. It is an unrefereed AIPW or DML result rather than a
  TMLE.
- Karim (2026), [*Cross-Fitted Survey-Weighted TMLE with Design-Based Variance for Causal Machine
  Learning*](https://arxiv.org/abs/2606.30918), arXiv:2606.30918. Read first-hand in the v2 main
  text. The proofs are in a web appendix this project did not read. Section 3.3 and condition (C2)
  ask that folds be "formed of whole PSUs, assigned by a data-independent rule and balanced within
  every stratum". Theorems 1 and 2 give the result. This is the closest estimator read: a row-level
  TMLE with a pooled fluctuation. Its strata are survey sampling strata, and (C2) excludes
  treatment strata. The preprint is unrefereed.
- Benitez, Nugent & Balzer (2023), [*Defining and estimating effects in cluster randomized trials:
  A methods comparison*](https://doi.org/10.1002/sim.9813), *Statistics in Medicine*
  42(19):3443-3466, DOI 10.1002/sim.9813. Read first-hand in the NIHMS author manuscript
  ([PMC10898620](https://pmc.ncbi.nlm.nih.gov/articles/PMC10898620/)). Section 3.1.2 and Section
  3.2.1 give the cluster-sum aggregation for a row-weighted estimand. Section 3.2.1 states that
  sample splitting and variance estimation "must respect the cluster as the independent unit". The
  paper gives no fold law and no theorem for one. It also recommends a $t$ reference when the
  cluster count is small.
- Balzer, Zheng, van der Laan & Petersen (2019), [*A new approach to hierarchical data analysis:
  Targeted maximum likelihood estimation for the causal effect of a cluster-level
  exposure*](https://doi.org/10.1177/0962280218774936), *Statistical Methods in Medical Research*
  28(6):1761-1780, DOI 10.1177/0962280218774936. Read first-hand in the NIHMS author manuscript
  ([PMC6173669](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6173669/)). Section 3.1, Equation (9),
  and Section 4.2, Equations (20) and (21), give a cluster-level exposure and a cluster-level
  estimand. There is no cross-fitting and no split law.
- Balzer, van der Laan & Petersen (2016), [*Adaptive pre-specification in randomized trials with
  and without pair-matching*](https://doi.org/10.1002/sim.7023), *Statistics in Medicine*
  35(25):4528-4545, DOI 10.1002/sim.7023. Read first-hand in the NIHMS author manuscript
  ([PMC5084457](https://pmc.ncbi.nlm.nih.gov/articles/PMC5084457/)). Sections 3.1, 4.1, 5.1 and 6
  select an adjustment variable by cross-validation over independent units, where a row is a
  cluster, in a randomized trial with a fixed GLM library. There is no post-selection theorem and
  no row-level C-TMLE.
- Balzer, van der Laan, Ayieko, Kamya, Chamie, Schwab, Havlir & Petersen (2023), [*Two-Stage TMLE
  to reduce bias and improve efficiency in cluster randomized
  trials*](https://doi.org/10.1093/biostatistics/kxab043), *Biostatistics* 24(2):502-517, DOI
  10.1093/biostatistics/kxab043. Sections 3.2 and 3.3 give a two-stage cluster-level estimator.
  There is no cross-fitting.
- Nugent, Marquez, Charlebois, Abbott & Balzer (2024), [*Blurring cluster randomized trials and
  observational studies: Two-Stage TMLE for subsampling, missingness, and few independent
  units*](https://doi.org/10.1093/biostatistics/kxad015), *Biostatistics* 25(3):599-616, DOI
  10.1093/biostatistics/kxad015. Sections 2.1.3 and 2.2 treat subsampling, missingness and few
  independent units. There is no cross-fitting. The paper recommends a $t$ reference with $J-2$
  degrees of freedom when the cluster count is small.
- Schnitzer, van der Laan, Moodie & Platt (2014), [*Effect of breastfeeding on gastrointestinal
  infection in infants: A targeted maximum likelihood approach for clustered longitudinal
  data*](https://doi.org/10.1214/14-AOAS727), *The Annals of Applied Statistics* 8(2):703-725, DOI
  10.1214/14-AOAS727. Read first-hand in the [arXiv reprint](https://arxiv.org/abs/1407.8371),
  which reproduces the journal text. Section 3.4.1 gives a clustered longitudinal TMLE with a
  sandwich variance and no sample splitting. There is no cross-fitting theorem.

No source above covers treatment-stratified grouped folds for an observational estimator. None
covers cross-fitted longitudinal TMLE with whole-cluster folds. The package refuses both.

The grouped point-treatment split rests on Wang et al. (2024) for the partition, and on the
package's own estimating-equation argument for the rest. The
[fold and outcome-scale rules](technical-reference/cv-tmle.md#fold-and-outcome-scale-rules) state
that argument with its four conditions. The registered
[clustered point-treatment CV-TMLE study](technical-reference/method-evidence/clustered-point-treatment-cv-tmle.md)
is the only empirical witness for it.

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
  *Statistical Methods in Medical Research* 28(2):532-554, DOI 10.1177/0962280217729845,
  PMID 28936917. Sections 4.1 through 5 define the general, greedy, and preordered
  binary-ATE paths. Section 7.4 forms each interval from the ordinary efficient influence curve.
  Read first-hand in the [author preprint](https://arxiv.org/pdf/1703.02237). Algorithm 1,
  PDF page 6, selects a candidate by cross-validated loss. Section 5.5, PDF pages 10–11, selects
  the pre-ordering and the number of covariates with one cross-validation. To save computation,
  the authors "do not rely on a nested cross-validation procedure to select k for each
  pre-ordering strategy m". The source does not describe inner folds that cross-fit nuisances
  within each selection-training fold, as the shipped selector does. It specifies no fold rule for
  the shipped selector at all. The selector now draws both layers unstratified, which the
  [fold and outcome-scale rules](technical-reference/cv-tmle.md#fold-and-outcome-scale-rules)
  record.
- Ju, Schwab & van der Laan (2019), [*On adaptive propensity score truncation in causal
  inference*](https://doi.org/10.1177/0962280218774817), *Statistical Methods in Medical Research*
  28(6):1741-1760, DOI 10.1177/0962280218774817. Read first-hand in the preprint,
  [arXiv:1707.05861v1](https://arxiv.org/abs/1707.05861), and the published numbering may differ.
  Section 3.3.2 selects a propensity truncation level with C-TMLE. The preprint states no theorem.
  Section 4.5 reports that the estimated variance of CV-TMLE, MV-TMLE, and C-TMLE was smaller than
  the true variance in its experiments.
- Ju, Wyss, Franklin, Schneeweiss, Häggström & van der Laan (2019), [*Collaborative-controlled
  LASSO for constructing propensity score-based estimators in high-dimensional
  data*](https://doi.org/10.1177/0962280217744588), *Statistical Methods in Medical Research*
  28(4):1044-1063, DOI 10.1177/0962280217744588. Read first-hand in the preprint,
  [arXiv:1706.10029v1](https://arxiv.org/abs/1706.10029), and the published numbering may differ.
  Section 4.2 states that the C-TMLE estimator is asymptotically linear under regularity
  conditions and cites earlier work for it. The data analysis in Section 7 forms its intervals from
  the analytic influence curve. The paper states no post-selection theorem.

  Three 2019 *SMMR* papers have Cheng Ju as first author. The Ju, Gruber et al. entry above is
  volume 28(2), DOI 10.1177/0962280217729845. The Ju, Schwab and van der Laan entry is also volume
  28(6), and the Ju, Wyss et al. entry is volume 28(4). Cite each one with its second author.
- Ju, Benkeser & van der Laan (2020), [*Robust inference on the average treatment effect using the
  outcome highly adaptive lasso*](https://arxiv.org/abs/1806.06784), *Biometrics* 76(1):109-118,
  DOI 10.1111/biom.13121. This is another adaptive-propensity construction with explicit
  estimating equations and inference. It does not cover a generic selected candidate depth.
  Read first-hand in arXiv:1806.06784v3, and the published numbering may differ. Theorem 1, in
  Section 3.4 on PDF page 17, gives the influence function
  $D(O \mid \bar Q_0, \bar G(\bar Q_0), Q_0) - D_r(O \mid \bar Q_0, \bar G_{r,0})$. The second term
  is first order, and the authors attribute it to "the intentional inconsistent estimation" of the
  propensity. It vanishes when the propensity limit equals the true propensity. The shipped C-TMLE
  intervals carry no such term.
- Benkeser, Cai & van der Laan (2020), [*A nonparametric super-efficient estimator of the average
  treatment effect*](https://doi.org/10.1214/19-STS735), DOI 10.1214/19-STS735
  ([preprint](https://arxiv.org/abs/1901.05056)). Theorem 1 derives asymptotic linearity with the
  usual TMLE curve evaluated at the adaptive propensity limit for a treatment-specific mean.
  Its six regularity conditions require the relevant score to be negligible, quarter-rate
  convergence of the outcome and adaptive-propensity estimators, influence-curve convergence,
  smoothness, a higher-order remainder, and an empirical-process condition. Section 3.1 gives a
  cross-validated variance construction in which each fold's outcome and adaptive propensity are
  both learned without that fold's rows. No separate first-order generated-design term appears in
  the theorem's influence function. Section 3.1 of the
  [author preprint](https://arxiv.org/pdf/1901.05056), PDF page 9, randomly partitions rows into
  approximately equal folds for variance estimation. It gives no treatment- or outcome-stratified
  selection-fold rule.
  Appendix D explicitly constructs the binary ATE with one propensity fit on
  `(Qbar(1, W), Qbar(0, W))` and sketches a cross-validated C-TMLE. It does not derive treatment
  with more than two levels, the shipped joint all-arm fluctuation and covariance, or simultaneous
  inference. The locators above are those of the preprint, arXiv:1901.05056 v1. The published
  article is paywalled, and this audit did not check that its theorem, section, and appendix
  numbers match the preprint.
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
- Cui & Tchetgen Tchetgen (2024), [*Selective machine learning of doubly robust
  functionals*](https://doi.org/10.1093/biomet/asad055), *Biometrika* 111(2):517-535, DOI
  10.1093/biomet/asad055. Read first-hand in
  [arXiv:1911.02029v6](https://arxiv.org/abs/1911.02029), and the published numbering may differ.
  The paper selects nuisance learners by a cross-validated pseudo-risk. Theorem 5.1 is an oracle
  inequality for that selector, and Theorem 5.2 gives its consistency. Neither gives a limit law.
  Section 6 says a Wald interval at the selected learners "is completely blind to the model
  selection step, it may not yield uniformly valid confidence intervals". It describes a
  split-and-swap estimator and a smoothed selector, and leaves their formal comparison outside the
  paper. This supports the fixed-candidate label of F18. It does not address the
  inconsistent-mechanism case of RM12.
- Qiu, Luedtke & Carone (2021), [*Universal sieve-based strategies for efficient estimation using
  machine learning tools*](https://doi.org/10.3150/20-BEJ1309), *Bernoulli* 27(4):2300-2336, DOI
  10.3150/20-BEJ1309. Read first-hand in [arXiv:2003.01856v2](https://arxiv.org/abs/2003.01856),
  and the published numbering may differ. Theorem 4, Section 4.4, keeps the ordinary influence
  function after k-fold cross-validation selects the number of sieve terms. Its conditions must
  hold for some deterministic number of terms, and Condition C5 must hold for every candidate.
  Section 3.3 shows by simulation that a cross-validated HAL variation-norm bound does not give an
  efficient plug-in estimator. The result is a template for a uniform-expansion argument. It does
  not treat a targeting step or a propensity path.
- Bibaut & van der Laan (2017), [*Data-adaptive smoothing for optimal-rate estimation of possibly
  non-regular parameters*](https://arxiv.org/abs/1706.07408), arXiv:1706.07408v2. Read first-hand.
  This search found no journal version. Section 2.1 splits the sample into three subsamples, and
  Theorems 1 and 2 select a scalar smoothing index on subsamples separate from the estimation
  sample. The targets may be non-regular. The package selects its index on the full sample.
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
- Zhang, Shao, Yu & Wang (2018), [*Impact of sufficient dimension reduction in nonparametric
  estimation of causal effect*](https://doi.org/10.1080/24754269.2018.1466100), *Statistical
  Theory and Related Fields* 2(1):89-95, DOI 10.1080/24754269.2018.1466100. This entry records
  the abstract only. The abstract states that estimating the covariate reduction changes the
  asymptotic variance, unless the reduction keeps covariates that are superfluous for estimation.
  It treats nonparametric regression and inverse propensity weighting, and no targeting step.
- Ma, Zhu, Zhang, Tsai & Carroll (2019), [*A robust and efficient approach to causal inference
  based on sparse sufficient dimension reduction*](https://doi.org/10.1214/18-AOS1722), *Annals of
  Statistics* 47(3):1505-1535, DOI 10.1214/18-AOS1722. This entry records the abstract only. The
  abstract states that an average-treatment-effect estimator built on the efficient influence
  function, with both nuisances estimated after a sparse dimension reduction, is asymptotically
  normal and semiparametrically efficient. It treats no targeting step and no shared multinomial
  mechanism.
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
  [`stage1`, lines 84-110](https://github.com/jucheng1992/ctmle/blob/18de559f47dc1286617350a0668391e80e1dbf7c/R/functions.R#L84-L110)
  takes a continuous outcome's bounds from `range(Y)` when the caller supplies none, and
  `maptoYstar` rescales by the same sample range.
  [`cv`, lines 446-447](https://github.com/jucheng1992/ctmle/blob/18de559f47dc1286617350a0668391e80e1dbf7c/R/functions_discrete.R#L446-L447)
  draws its default folds from `sample(1:n, n)` without strata. This source is implementation
  provenance. It states no result for either rule.
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

  A 2026-09-21 search added Cui and Tchetgen Tchetgen (2024), Qiu, Luedtke and Carone (2021),
  Bibaut and van der Laan (2017), Ju, Schwab and van der Laan (2019), and Ju, Wyss et al. (2019).
  It read each one first-hand in the arXiv version its entry names. It also read Theorem 1 of Ju,
  Benkeser and van der Laan (2020) and the `ctmle` outcome-range and fold code.

  It recorded Zhang et al. (2018), Ma et al. (2019), and Bannick et al. (2025) from their
  abstracts. No source it read closes F18, the cross-fitted multi-arm part of F19, or the fold and
  outcome-scale rules.

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
- Díaz, Hoffman, Hejazi & Williams (2024), [*Causal survival analysis under competing risks
  using longitudinal modified treatment policies*](https://doi.org/10.1007/s10985-023-09606-7),
  *Lifetime Data Analysis* 30:213–236. The corrected
  [arXiv v3](https://arxiv.org/abs/2202.03513v3) supplies the version this project follows.
  Proposition 1 identifies the cause-specific cumulative incidence target. Appendix E gives the
  cross-fitted TMLE directly: out-of-fold initial nuisances, one all-row fluctuation per node,
  pooled backward carry, and the score and remainder argument. The
  [2025 author correction](https://doi.org/10.1007/s10985-025-09651-4) corrects the outcome
  definition and dependent results in the original article.
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
- Williams & Díaz (2023), [*lmtp: An R Package for Estimating the Causal Effects of Modified
  Treatment Policies*](https://doi.org/10.1353/obs.2023.0019), *Observational Studies*
  9(2):103–122. This is the software reference for the comparator used by the registered studies;
  exact algorithm claims below remain pinned to the named source release.
- Williams & Díaz (2025), [*Erratum: lmtp: An R Package for Estimating the Causal Effects of
  Modified Treatment Policies*](https://doi.org/10.1353/obs.2025.a973072), *Observational
  Studies* 11(3):365–367. The erratum corrects identification and positivity assumptions. It does
  not change the targeting construction, which is why code-history claims remain tied to the
  versioned source and commit below.
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
- Hernán & Robins, [*Causal Inference: What If*](https://miguelhernan.org/whatifbook), Chapman &
  Hall/CRC. Chapter 20 treats treatment-confounder feedback. Section 20.3, pages 271-273 of the
  edition dated 19 August 2026, shows why adjustment for that confounder opens a collider path. Cited by the
  [longitudinal TMLE tutorial](examples/longitudinal-tmle.ipynb).
- Young, Stensrud, Tchetgen Tchetgen & Hernán (2020), [*A causal framework for classical
  statistical estimands in failure-time settings with competing events*](https://doi.org/10.1002/sim.8471),
  *Statistics in Medicine* 39(8):1199-1236, DOI 10.1002/sim.8471. The paper defines the total
  and controlled direct effects on a cause-specific cumulative incidence, and their identifying
  conditions. Cited by the [time-to-event tutorial](examples/longitudinal-survival.ipynb).

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
  [10.1002/sim.7389](https://doi.org/10.1002/sim.7389). Read first-hand in the arXiv v1 author
  manuscript. The locators below are that manuscript's pages. This audit did not compare them with
  the journal version.

  | locator | content |
  | --- | --- |
  | Section 2.1, page 6 | observed data $(W, A, M, MY)$ in the nonparametric model, with `A` a binary arm indicator |
  | Assumption 2, page 7 | treatment independent of the potential outcome given `W` |
  | Section 3, Equation (1), page 10 | the efficient influence function |
  | Section 3, Equation (3), page 11 | the logistic TMLE among rows with $(A, M) = (1, 1)$ |
  | Equation (6), page 15 | the reductions |
  | Theorem 1, page 16, Equations (11)–(13), page 19, and Theorem 2, page 20 | the corrections and the targeting algorithm |

  The [point-treatment entry](#point-treatment-and-stochastic-interventions) for the same paper
  records its identification and rate locators.

  The paper establishes the missing-outcome construction for randomized treatment. It does not
  establish the observational-treatment or missing-treatment compositions that the canonical
  package exposes. Page 4 calls cross-validated results "straightforward extensions of the work of
  Zheng and van der Laan (2011)". It adds "we do not pursue such results here". Page 26 states that
  such a development "would follow from trivial extensions" of that work.

The `benkeser/drtmle` R package supplies implementation provenance and a bounded numerical
comparison. Agreement with it does not establish the theorem or truth-based validity. The
registered study asks those questions separately. The inspected source is pinned at
[`538a3a2`](https://github.com/benkeser/drtmle/tree/538a3a264c1ca984b6d88978ca7f96165f43152c):
`R/estimate.R` loops the reductions over treatment levels and constructs a compatible initial
mechanism; `R/fluctuate.R` applies independent one-vs-rest mechanism fluctuations.

## The TWINS example

The [TWINS notebook](examples/twins-causal-inference.ipynb) cites these two sources for the data
and for the reading of its association.

- Almond, Chay & Lee (2005), [*The costs of low birth weight*](https://doi.org/10.1162/003355305774268228),
  *Quarterly Journal of Economics* 120(3):1031-1083, DOI 10.1162/003355305774268228. The
  notebook links the NBER working paper w10552.
- Louizos, Shalit, Mooij, Sontag, Zemel & Welling (2017), [*Causal effect inference with deep
  latent-variable models*](https://arxiv.org/abs/1705.08821), NIPS 2017, arXiv:1705.08821. In
  arXiv v2, Section 4.3 describes the TWINS benchmark. It keeps pairs in which both twins weigh
  less than 2 kg, and it defines treatment as being the heavier twin.

The notebook also cites two sources for its sensitivity analysis. Their entries are under
[sensitivity analysis](#sensitivity-analysis).

- Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022), for the omitted-variable bounds.
- VanderWeele & Ding (2017), for the E-value.

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
