suppressPackageStartupMessages(library(tmle))
source("/fixture/study_harness.R")
options(digits = 17)

args <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(args$samples), stringsAsFactors = FALSE)
truths <- read.csv(args$truths, stringsAsFactors = FALSE)
truth_key <- paste(truths$scenario, truths$replicate, truths$estimand, sep = "|")

fit_one <- function(frame) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  n <- nrow(frame)
  expected_folds <- 0:9
  if (!setequal(unique(frame$fold), expected_folds)) {
    stop("the supplied payload does not contain the fitted ten-fold assignment")
  }
  if (!all(is.finite(frame$qn)) || !all(frame$qn >= 0 & frame$qn <= 1)) {
    stop("the supplied outcome predictions are not probabilities")
  }
  if (!all(is.finite(frame$pin)) || !all(frame$pin >= 0 & frame$pin <= 1)) {
    stop("the supplied response predictions are not probabilities")
  }
  if (!setequal(unique(frame$A), c(0, 1))) {
    stop("the realized natural-course payload must retain both original treatment arms")
  }

  # A constant synthetic treatment selects tmle's missing-outcome population-mean branch.
  # Duplicating the realized-arm predictions makes QAW, Q1W, and pDelta1 equal to the
  # stitched out-of-fold nuisance values for each original (A, W) row.
  fit <- tmle::tmle(
    Y = frame$Y,
    A = rep(1, n),
    W = data.frame(A_original = frame$A, W = frame$W),
    Delta = frame$Delta,
    Q = cbind(frame$qn, frame$qn),
    g1W = rep(1, n),
    pDelta1 = cbind(frame$pin, frame$pin),
    family = "binomial",
    fluctuation = "logistic",
    Qbounds = c(0, 1),
    gbound = c(0.01, 1),
    alpha = 0.9995,
    cvQinit = FALSE,
    prescreenW.g = FALSE,
    target.gwt = FALSE,
    B = 1,
    evalATT = FALSE,
    verbose = FALSE
  )
  if (is.null(fit$estimates$EY1) || !is.null(fit$estimates$EY0) ||
      !is.null(fit$estimates$ATE)) {
    stop("tmle did not select its scalar population-mean result shape")
  }
  if (fit$Qinit$type != "user-supplied values" ||
      fit$g$type != "user-supplied values" ||
      fit$g.Delta$type != "user-supplied values") {
    stop("tmle refitted a nuisance instead of using the supplied predictions")
  }

  truth_row <- match(paste(scenario, replicate, "ey_obs", sep = "|"), truth_key)
  if (is.na(truth_row)) stop(sprintf("missing truth for %s/%s/ey_obs", scenario, replicate))
  truth <- truths$truth[[truth_row]]
  estimate <- as.numeric(fit$estimates$EY1$psi)
  standard_error <- sqrt(as.numeric(fit$estimates$EY1$var.psi))
  interval <- as.numeric(fit$estimates$EY1$CI)
  low <- interval[[1]]
  high <- interval[[2]]
  data.frame(
    implementation = "tmle-r-population-mean",
    scenario = scenario,
    replicate = replicate,
    n = n,
    estimand = "ey_obs",
    truth = truth,
    estimate = estimate,
    inference_estimate = estimate,
    std_error = standard_error,
    ci_lower = low,
    ci_upper = high,
    inference_scale = "identity",
    covered = as.integer(low <= truth && truth <= high),
    initial_estimate = mean(frame$qn),
    stringsAsFactors = FALSE
  )
}

groups <- split(samples, interaction(samples$scenario, samples$replicate, drop = TRUE))
cores <- study_cores(groups)
results <- parallel::mclapply(
  seq_along(groups),
  study_fitter(groups, fit_one),
  mc.cores = cores,
  mc.preschedule = TRUE
)
study_collect(
  results,
  expected = length(groups),
  output = args$output,
  versions = c(
    study_version("tmle"),
    paste("source sha256", Sys.getenv("TMLE_SHA256"))
  )
)
