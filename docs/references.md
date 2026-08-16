# References

Every paper the package's derivations are read off, in one place, with enough of a locator to
find the passage a docstring or a document is pointing at.

**How to read a citation in this repository.** Prose cites author and year — "the sequential
regression of Bang & Robins (2005)", "van der Laan (2014) Theorem 3" — and resolves here. Where a
document argues *against* a source, or transcribes a display from it, it carries a section or page
number as well; those are the citations worth checking, and they are the ones that have them.

**Nothing here is stored in the repository.** Two PDFs were, and were deleted once everything they
were cited for had been transcribed into
[the DR-TMLE contract](drtmle.md) — which is why the two `DRTMLE` sources
below carry page numbers where the rest carry none. A path to a file only a previous reader had is
not a citation; a page number is.

## Targeted learning, in general

- van der Laan & Rubin (2006), [*Targeted Maximum Likelihood Learning*](https://doi.org/10.2202/1557-4679.1043),
  DOI 10.2202/1557-4679.1043.
- Díaz Muñoz & van der Laan (2012), [*Population Intervention Causal Effects Based on
  Stochastic Interventions*](https://doi.org/10.1111/j.1541-0420.2011.01685.x), DOI
  10.1111/j.1541-0420.2011.01685.x.
- Gruber & van der Laan (2010), *A targeted maximum likelihood estimator of a causal effect on a
  bounded continuous outcome*.
- Gruber & van der Laan (2012), *tmle: An R Package for Targeted Maximum Likelihood Estimation*.
- Zheng & van der Laan (2011), *Cross-validated targeted minimum-loss-based estimation*.
- Levy (2018), *An Easy Implementation of CV-TMLE*, arXiv:1811.04573. The abstract
  distinguishes the original fold-wise plug-in evaluation from the common targeting
  regression pooled over validation folds.
- Coyle et al., R package [`tmle3`](https://github.com/tlverse/tmle3), source at commit
  [`ed72f8a`](https://github.com/tlverse/tmle3/tree/ed72f8a20e64c914ab25ffe015d865f7a9963d27).
  `R/tmle3_Update.R` selects the
  `"validation"` likelihood when `cvtmle=TRUE` and fits one update to the stacked
  validation predictions; `R/Param_TSM.R` evaluates the treatment-specific mean and its
  influence curve from those validation likelihood values. Used as an implementation
  reference, not as an oracle for the estimand derivation or as a moving specification.
  Levy (2018) is the stable marker for the default stacked construction.
- The fold/full prediction mechanism used by `tmle3` lives in its `sl3` dependency, pinned
  here at [`0e8f236`](https://github.com/tlverse/sl3/tree/0e8f2365bcbe54010b8120c04a7a2dcfc8119227).
  `R/Lrnr_cv.R` builds `fold_fits` and, when requested, a `full_fit`; `predict_fold(...,
  "validation")` assembles held-out predictions while `predict_fold(..., "full")` uses the
  all-training fit. This is the design source for C-TMLE's nested selection predictions.
- van der Laan & Gruber (2016), *One-step targeted minimum loss-based estimation*.

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
- The `tlverse/ctmle3` implementation at commit
  [`a4ea77b`](https://github.com/tlverse/ctmle3/tree/a4ea77b07747dfee9b2eecb9cbca88262e0559ea).
  `R/LF_oat.R` fits categorical treatment on the complete vector of treatment-specific
  outcome predictions; `R/tmle3_Spec_TSM_all.R` requests all treatment-specific means.
  This is the implementation source for `CTMLE(strategy="oat")`.

## Longitudinal, survival and marginal structural models

- Bang & Robins (2005), *Doubly robust estimation in missing data and causal inference models*.
- van der Laan & Gruber (2012), *Targeted minimum loss based estimation of causal effects of
  multiple time point interventions*.
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
  [`jvpoulos/multi-ltmle`](https://github.com/jvpoulos/multi-ltmle) repository contains the
  longitudinal multi-valued-treatment simulation and reproduction code.
- Neugebauer & van der Laan (2007), [*Nonparametric causal effects based on marginal
  structural models*](https://doi.org/10.1016/j.jspi.2005.12.008), DOI
  10.1016/j.jspi.2005.12.008.
- Rosenblum & van der Laan (2010), [*Targeted Maximum Likelihood Estimation of the
  Parameter of a Marginal Structural Model*](https://doi.org/10.2202/1557-4679.1238),
  DOI 10.2202/1557-4679.1238.
- Orellana, Rotnitzky & Robins (2010), *Dynamic regime marginal structural mean models for
  estimation of optimal dynamic treatment regimes*.

## Incremental interventions

- Kennedy (2019), *Nonparametric causal effects based on incremental propensity score
  interventions*.

## Sensitivity analysis

- Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022), *Long story short: omitted variable bias
  in causal machine learning*.
- VanderWeele & Ding (2017), *Sensitivity analysis in observational research: introducing the
  E-value*.
- Scharfstein, Rotnitzky & Robins (1999), *Adjusting for nonignorable drop-out using semiparametric
  nonresponse models*.

## Multiple testing

- Benjamini & Hochberg (1995), [*Controlling the False Discovery Rate: A Practical and
  Powerful Approach to Multiple Testing*](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x),
  DOI 10.1111/j.2517-6161.1995.tb02031.x.

## Doubly-robust inference (`DRTMLE`)

The three the variant rests on — the first two for the estimating equations, the third's
implementation for the influence curve. What each supplies, and where in it, is
[the contract's source table](drtmle.md#the-sources).

- van der Laan (2014), *Targeted estimation of nuisance parameters to obtain valid statistical
  inference*, International Journal of Biostatistics 10(1):29–57. **Theorem 3** is the bivariate
  construction's regularity conditions, and is the one thing in this section that has *not* been
  read here.
- Benkeser, Carone, van der Laan & Gilbert (2016), *Doubly-robust Nonparametric Inference on the
  Average Treatment Effect*, U.C. Berkeley Division of Biostatistics Working Paper Series, paper
  356. Read first-hand. §3.1 and equation (2) are p. 9; §3.2, Theorem 1 and the recursive algorithm
  are pp. 10–11; appendix A is pp. 19–20, appendix B p. 21, appendix C pp. 21–22.
- Benkeser, Carone, van der Laan & Gilbert (2017), *Doubly robust nonparametric inference on the
  average treatment effect*, Biometrika 104(4):863–880 (PMC5793673). The **published** version of
  the above, and authoritative wherever the two differ. Not read here — which matters in exactly
  one place, the sign of the mechanism correction, and the working paper's own appendices settle
  that without it.
- Benkeser & Hejazi (2023), *Doubly-Robust Inference in R using `drtmle`*, Observational Studies
  9(2):43–78. Read first-hand. Multi-level treatments are §4.6, pp. 66–67; cross-validated nuisance
  regression is §4.7, p. 69.
- Díaz & van der Laan (2017), *Doubly robust inference for targeted minimum loss-based estimation
  in randomized trials with missing outcome data*, Statistics in Medicine 36:3807–3819, DOI
  [10.1002/sim.7389](https://doi.org/10.1002/sim.7389). Read first-hand. §2.1 states the observed-
  data model and EIF; equation (6) defines the reductions; Theorems 1–2 and equations (11)–(13)
  give the corrections and targeting algorithm. This establishes the missing-outcome construction
  for randomized treatment; it does not establish the observational-treatment or missing-treatment
  compositions exposed by the canonical package, and it explicitly leaves cross-validation to
  future work.

The `benkeser/drtmle` R package's source and reference documentation are cited in a few places as
**provenance** — where a formula was transcribed from, and what it is named there. Running it is
[not acceptance evidence](architecture-invariants.md#validation-and-evidence): two checks that
cannot fail against the same class of error are one check. The inspected source is pinned at
[`538a3a2`](https://github.com/benkeser/drtmle/tree/538a3a264c1ca984b6d88978ca7f96165f43152c):
`R/estimate.R` loops the reductions over treatment levels and constructs a compatible initial
mechanism; `R/fluctuate.R` applies independent one-vs-rest mechanism fluctuations.
