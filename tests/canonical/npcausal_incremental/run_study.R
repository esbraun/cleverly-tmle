suppressPackageStartupMessages(library(npcausal))
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
scenario <- "binary_incremental_odds"
z <- qnorm(0.975)

# One `ipsi` call per multiplier, not one call carrying `delta.seq = c(1, 2, 0.5)`.
#
# The vectorised call is the natural way to write this and it is the wrong way, because
# `return_ifvals = TRUE` returns `ifvals - est.eff` where `ifvals` is n x k and `est.eff` is
# length k.  R recycles the subtrahend down the flattened column-major array rather than
# across columns, so with k > 1 every returned influence value has some *other* multiplier's
# estimate removed from it.  `sweep(ifvals, 2, est.eff)` is what that line means.  The bug is
# invisible in `res` and `res.ptwise`, whose standard errors are computed from the raw matrix
# before the centering runs, so a runner that reads only the published table never sees it.
#
# This study needs the influence values themselves, because two of its five estimands are
# contrasts and a contrast's standard error is not recoverable from two marginal ones.  With
# k = 1 the recycling is a scalar subtraction and is correct, so the defect is designed around
# rather than corrected: patching a comparator forfeits what a comparator is for.
DELTAS <- list(
  list(delta = 1, estimand = "ey_ipsi[natural course]"),
  list(delta = 2, estimand = "ey_ipsi[odds x2]"),
  list(delta = 0.5, estimand = "ey_ipsi[odds x0.5]")
)

truth_for <- function(replicate, estimand) {
  selected <- truths$replicate == replicate & truths$scenario == scenario & truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows", sum(selected)))
  truths$truth[selected]
}

fit_delta <- function(frame, delta, seed) {
  # Reseeded to the same value for every multiplier of one replication, which is what makes
  # the three calls share their nuisances rather than merely their data.  `ipsi` draws its
  # cross-fitting assignment first and fits the treatment model next, and neither step reads
  # `delta`; for a single time point the outcome regression is a fit of Y on (W, A) and does
  # not read `delta` either.  Identical seeds therefore give identical folds, identical
  # propensity scores and identical outcome fits, and the contrast below cancels the shared
  # nuisance error instead of adding three independent copies of it.
  set.seed(seed)
  n <- nrow(frame)
  design <- cbind(W1 = as.numeric(frame$W == 1), W2 = as.numeric(frame$W == 2))
  # `nsplits = 2`, not the documented `nsplits = 1`.  The single-split path is advertised in
  # `?ipsi` and is not implemented: the training mask is `slong != split`, which selects no
  # rows at all when there is only one split, so the nuisance fits raise on empty data.  The
  # comparison therefore runs against a cross-fitted npcausal and a non-cross-fitted
  # `cleverly`, and the study's Limits section says so.
  #
  # `SL.glm.interaction` rather than `SL.glm`, because `ipsi` appends the treatment column to
  # the outcome design itself and fits `rtp1 ~ .`.  Main effects alone are additive in A and
  # cannot represent this law.  `.^2` over two indicators and A spans all six (W, A) cells, so
  # the outcome fit is the saturated nonparametric one; the identically-zero W1:W2 column is
  # dropped as collinear.  The same library saturates the three-cell treatment model.
  fit <- ipsi(
    y = frame$Y,
    a = frame$A,
    x.trt = design,
    x.out = design,
    time = rep(1L, n),
    id = seq_len(n),
    delta.seq = delta,
    nsplits = 2,
    progress_bar = FALSE,
    return_ifvals = TRUE,
    fit = "sl",
    sl.lib = c("SL.glm.interaction")
  )
  influence <- as.numeric(fit$ifvals[, 1])
  estimate <- fit$res.ptwise$est[[1]]
  # `res.ptwise$se` is `sd(raw ifvals)`, computed before the centering above.  Reading the
  # published standard error back off the influence values this runner uses is what confirms
  # the two are the same object, so a future upstream change to either one stops the run
  # instead of quietly moving the comparison onto a different quantity.
  if (abs(sd(influence) - fit$res.ptwise$se[[1]]) > 1e-9 * max(1, fit$res.ptwise$se[[1]])) {
    stop("npcausal influence values disagree with its own published standard error")
  }
  list(estimate = estimate, ic = influence)
}

row_for <- function(replicate, name, fit, n) {
  truth <- truth_for(replicate, name)
  standard_error <- sd(fit$ic) / sqrt(n)
  low <- fit$estimate - z * standard_error
  high <- fit$estimate + z * standard_error
  data.frame(
    implementation = "npcausal",
    scenario = scenario,
    replicate = replicate,
    n = n,
    estimand = name,
    truth = truth,
    estimate = fit$estimate,
    inference_estimate = fit$estimate,
    std_error = standard_error,
    ci_lower = low,
    ci_upper = high,
    inference_scale = "identity",
    covered = as.integer(low <= truth && truth <= high),
    # `ipsi` adds its correction to an average of influence values and never forms an
    # untargeted estimate, so there is no plug-in to publish.  Left missing rather than set to
    # the estimate: an absent value says the quantity does not exist, and a repeated one would
    # say it exists and did not move, which is the claim the plug-in witness tests.
    initial_estimate = NA_real_,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  seed <- 10000 * replicate + 7
  fits <- lapply(DELTAS, function(entry) fit_delta(frame, entry$delta, seed))
  names(fits) <- vapply(DELTAS, function(entry) entry$estimand, character(1))
  natural <- fits[["ey_ipsi[natural course]"]]
  contrast <- function(shifted) {
    list(
      estimate = shifted$estimate - natural$estimate,
      ic = shifted$ic - natural$ic
    )
  }
  rows <- lapply(names(fits), function(name) row_for(replicate, name, fits[[name]], nrow(frame)))
  rbind(
    do.call(rbind, rows),
    row_for(
      replicate,
      "ate_ipsi[odds x2 vs natural course]",
      contrast(fits[["ey_ipsi[odds x2]"]]),
      nrow(frame)
    ),
    row_for(
      replicate,
      "ate_ipsi[odds x0.5 vs natural course]",
      contrast(fits[["ey_ipsi[odds x0.5]"]]),
      nrow(frame)
    )
  )
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(results, expected, paths$output, versions = study_version("npcausal"))
