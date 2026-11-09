suppressPackageStartupMessages(library(npcausal))
source("/fixture/study_harness.R")
options(digits = 17)

# The npcausal `ctseff` reference of the `dose-response-npcausal` study (X93 S2).
#
# `ctseff` require()s SuperLearner, earth, gam, ranger and KernSmooth, and a missing one is a
# warning there followed by a "could not find function" error deep in the fit.  The image is
# the one the incremental studies build (tests/canonical/npcausal_incremental/Dockerfile);
# KernSmooth ships with rocker/r-ver's recommended packages.  A missing package is a broken
# image, not a fitting failure, so it stops the run here, before any group is fitted.
for (package in c("SuperLearner", "earth", "gam", "ranger", "KernSmooth")) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop(sprintf("ctseff needs the R package %s, which the image does not install", package))
  }
}

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
scenario <- "linear_gaussian"

# One pair per row: the name both implementations report (the triweight bandwidth of
# `DoseResponse`), the dose, and the Gaussian bandwidth `ctseff` receives, h R(K_G) / R(K_T).
# `tests/unit/test_dose_response_npcausal_design.py` checks these literals against
# `tests.studies.dose_response_npcausal.PAIRS`, digit for digit.
PAIRS <- data.frame(
  estimand = c(
    "dose_response[h=1,a=1.5]",
    "dose_response[h=1,a=2.5]",
    "dose_response[h=1,a=3.5]",
    "dose_response[h=1.5,a=1.5]",
    "dose_response[h=1.5,a=2.5]",
    "dose_response[h=1.5,a=3.5]"
  ),
  dose = c(1.5, 2.5, 3.5, 1.5, 2.5, 3.5),
  bandwidth = c(
    0.3457676162028392, 0.3457676162028392, 0.3457676162028392,
    0.5186514243042588, 0.5186514243042588, 0.5186514243042588
  ),
  stringsAsFactors = FALSE
)
# `ctseff`'s evaluation grid is seq(a.rng[1], a.rng[2], length.out = n.pts).  The range is the
# observed dose range widened outward to multiples of 1/8, so every grid point is a dyadic
# rational, every declared dose is a grid point exactly, and the grid spans the data as
# `ctseff`'s default range does (its smoothing splines extrapolate past the grid).
GRID_STEP <- 0.125
# One learner, so SuperLearner's cross-validated weight is 1 and each prediction is that
# learner's full-sample fit: no fit depends on SuperLearner's fold draw.  `.^2` over (W, a) is
# the design of `cleverly`'s outcome learner and holds the law's outcome regression, dose mean
# and dose variance exactly.
SL_LIBRARY <- "SL.glm.interaction"

truth_for <- function(replicate, estimand) {
  selected <- truths$replicate == replicate & truths$scenario == scenario & truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows", sum(selected)))
  truths$truth[selected]
}

fit_bandwidth <- function(frame, bandwidth, doses, low, high, points) {
  study_stage("ctseff", {
    fit <- ctseff(
      y = frame$Y,
      a = frame$A,
      x = as.matrix(frame[, c("W1", "W2", "W3")]),
      bw.seq = bandwidth,
      n.pts = points,
      a.rng = c(low, high),
      sl.lib = SL_LIBRARY
    )
    # A one-value `bw.seq` selects that value; the risk table says which one ran.
    if (!identical(fit$bw.risk$bw, bandwidth)) {
      stop(sprintf("ctseff ran bandwidth %s, not the declared %s", fit$bw.risk$bw, bandwidth))
    }
    at <- match(doses, fit$res$a.vals)
    if (anyNA(at)) {
      stop(sprintf("ctseff's grid misses dose %s", paste(doses[is.na(at)], collapse = ", ")))
    }
    fit$res[at, ]
  })
}

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  n <- nrow(frame)
  low <- floor(min(frame$A) / GRID_STEP) * GRID_STEP
  high <- ceiling(max(frame$A) / GRID_STEP) * GRID_STEP
  points <- as.integer(round((high - low) / GRID_STEP)) + 1L
  rows <- lapply(unique(PAIRS$bandwidth), function(bandwidth) {
    selected <- PAIRS[PAIRS$bandwidth == bandwidth, ]
    res <- fit_bandwidth(frame, bandwidth, selected$dose, low, high, points)
    truth <- vapply(selected$estimand, function(name) truth_for(replicate, name), numeric(1))
    # `res$se` is the standard deviation of the estimated influence function; `ctseff`'s own
    # interval is est -/+ 1.96 se / sqrt(n), which these rows keep.
    std_error <- res$se / sqrt(n)
    data.frame(
      implementation = "npcausal",
      scenario = scenario,
      replicate = replicate,
      n = n,
      estimand = selected$estimand,
      truth = unname(truth),
      estimate = res$est,
      inference_estimate = res$est,
      std_error = std_error,
      ci_lower = res$ci.ll,
      ci_upper = res$ci.ul,
      inference_scale = "identity",
      covered = as.integer(res$ci.ll <= truth & truth <= res$ci.ul),
      # `ctseff` returns no untargeted curve.  Missing, not the estimate: a repeated value
      # would claim a plug-in that exists and did not move.
      initial_estimate = NA_real_,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  })
  do.call(rbind, rows)
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(results, expected, paths$output, versions = study_version("npcausal"))
