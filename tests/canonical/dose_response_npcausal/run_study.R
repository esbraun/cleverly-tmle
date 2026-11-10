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

# Each replicate fits the pairs its truth rows name.  A name is `pair_name(h, a0)` with the
# triweight bandwidth h of `DoseResponse`; `ctseff` receives the Gaussian bandwidth of equal
# roughness per unit bandwidth, h R(K_G) / R(K_T), computed in the operation order of
# `tests.studies.dose_response_npcausal.gaussian_bandwidth` so that the two agree to the bit.
# The declared run names six pairs per replicate; the R pilot
# (tests/canonical/dose_response_npcausal/pilot.py) names the three doses of one candidate
# bandwidth per replicate, so each `ctseff` call is its own group and its own failure record.
PAIR_NAME <- "^dose_response\\[h=([0-9.]+),a=([0-9.]+)\\]$"
gaussian_bandwidth <- function(h) h * (1 / (2 * sqrt(pi))) / (350 / 429)
# `ctseff`'s evaluation grid is seq(a.rng[1], a.rng[2], length.out = n.pts).  The range is the
# observed dose range widened outward to multiples of 1/8, so every grid point is a dyadic
# rational, every declared dose is a grid point exactly, and the grid spans the data as
# `ctseff`'s default range does.  A grid short of the data would leave its smoothing splines to
# extrapolate (linearly) at the outer doses.
GRID_STEP <- 0.125
# One learner, so SuperLearner's cross-validated weight is 1 and each prediction is that
# learner's full-sample fit: no fit depends on SuperLearner's fold draw.  `.^2` over (W, a) is
# the design of `cleverly`'s outcome learner and holds the law's outcome regression, dose mean
# and dose variance exactly.
SL_LIBRARY <- "SL.glm.interaction"

pairs_for <- function(replicate) {
  # The truth file is the driver's; a replicate without truth rows, or a name outside the
  # pattern, is a broken input, not a fitting failure, so both stop the run.
  selected <- truths[truths$replicate == replicate & truths$scenario == scenario, ]
  if (!nrow(selected)) stop(sprintf("no truth rows for replicate %s", replicate))
  if (!all(grepl(PAIR_NAME, selected$estimand))) {
    stop(sprintf("unparsed estimand among %s", paste(selected$estimand, collapse = ", ")))
  }
  h <- as.numeric(sub(PAIR_NAME, "\\1", selected$estimand))
  data.frame(
    estimand = selected$estimand,
    dose = as.numeric(sub(PAIR_NAME, "\\2", selected$estimand)),
    triweight = h,
    bandwidth = gaussian_bandwidth(h),
    truth = selected$truth,
    stringsAsFactors = FALSE
  )
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
    # An NA leave-one-out risk leaves `which.min` empty, so `ctseff` ran its final fit with a
    # zero-length bandwidth.  Whatever that call returned, the row is not the declared fit.
    if (!all(is.finite(fit$bw.risk$risk))) {
      stop("ctseff's leave-one-out bandwidth risk is NA, so its bandwidth selection is empty")
    }
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
  pairs <- pairs_for(replicate)
  rows <- lapply(unique(pairs$bandwidth), function(bandwidth) {
    selected <- pairs[pairs$bandwidth == bandwidth, ]
    res <- fit_bandwidth(frame, bandwidth, selected$dose, low, high, points)
    truth <- selected$truth
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
