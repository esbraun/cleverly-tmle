# References

Every paper the package's derivations are read off, in one place, with enough of a locator to
find the passage a docstring or a document is pointing at.

**How to read a citation in this repository.** Prose cites author and year — "the sequential
regression of Bang & Robins (2005)", "van der Laan (2014) Theorem 3" — and resolves here. Where a
document argues *against* a source, or transcribes a display from it, it carries a section or page
number as well; those are the citations worth checking, and they are the ones that have them.

**Nothing here is stored in the repository.** Two PDFs were, and were deleted once everything they
were cited for had been transcribed into
[the DRTMLE concordance](drtmle/theorem-concordance.md) — which is why the two `DRTMLE` sources
below carry page numbers where the rest carry none. A path to a file only a previous reader had is
not a citation; a page number is.

## Targeted learning, in general

- van der Laan & Rubin (2006), *Targeted Maximum Likelihood Learning*.
- Gruber & van der Laan (2010), *A targeted maximum likelihood estimator of a causal effect on a
  bounded continuous outcome*.
- Gruber & van der Laan (2012), *tmle: An R Package for Targeted Maximum Likelihood Estimation*.
- Zheng & van der Laan (2011), *Cross-validated targeted minimum-loss-based estimation*.
- van der Laan & Gruber (2016), *One-step targeted minimum loss-based estimation*.

## Collaborative TMLE

- van der Laan & Gruber (2010), *Collaborative double robust targeted maximum likelihood
  estimation*.
- Gruber & van der Laan (2010), *An application of collaborative targeted maximum likelihood
  estimation in causal inference and genomics*.
- Ju, Gruber, Lendle, Chambaz, Franklin, Wyss, Schneeweiss & van der Laan (2019), *Scalable
  collaborative targeted learning for high-dimensional data*.

## Longitudinal, survival and marginal structural models

- Bang & Robins (2005), *Doubly robust estimation in missing data and causal inference models*.
- van der Laan & Gruber (2012), *Targeted minimum loss based estimation of causal effects of
  multiple time point interventions*.
- Neugebauer & van der Laan (2007), *Nonparametric causal effects based on marginal structural
  models*.
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

## Doubly-robust inference (`DRTMLE`)

The three the variant rests on — the first two for the estimating equations, the third's
implementation for the influence curve. What each supplies, and where in it, is
[the concordance's source inventory](drtmle/theorem-concordance.md#0-source-inventory).

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

The `benkeser/drtmle` R package's source and reference documentation are cited in a few places as
**provenance** — where a formula was transcribed from, and what it is named there. Running it is
[retired by decision](roadmap.md#standing-decisions): two checks that cannot fail against the same
class of error are one check.
