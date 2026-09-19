suppressPackageStartupMessages(library(tmle))
source("/fixture/study_harness.R")
options(digits = 17)

args <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- as.data.frame(data.table::fread(cmd = paste("gzip -dc", shQuote(args$samples))))
truths <- read.csv(args$truths, stringsAsFactors = FALSE, check.names = FALSE)
truth_key <- paste(truths$scenario, truths$replicate, truths$estimand, sep = "|")

implementation <- "tmle-r-population-mean"
# The declared outcome support.  The binary law is on [0, 1] by definition, and the Python
# fit of the continuous law declares q_bounds = (0, 1).
q_bounds <- c(0, 1)
# R turns a scalar gbound into c(gbound, 1).  Every supplied response prediction must stay
# above it, so R clips nothing that cleverly does not.
gbound <- 0.01
alpha <- 0.9995
laws <- list(
  binary_mar_natural_course = "binomial",
  continuous_mar_natural_course = "gaussian"
)

check_payload <- function(frame) {
  if (!all(is.finite(frame$qn)) || !all(frame$qn >= 0 & frame$qn <= 1)) {
    stop("the supplied outcome predictions are not finite values in [0, 1]")
  }
  if (!all(is.finite(frame$pin)) || !all(frame$pin <= 1)) {
    stop("the supplied response predictions are not finite probabilities")
  }
  if (any(frame$pin <= gbound)) {
    stop("a supplied response prediction is at or below gbound, so R would clip it")
  }
  if (!setequal(unique(frame$A), c(0, 1))) {
    stop("the realized natural-course payload must retain both original treatment arms")
  }
}

# R takes the outcome scale from every non-NA Y (tmle.R line 1120).  Setting Y to the two
# q_bounds on two rows whose response indicator is zero fixes the scale at q_bounds, and
# the zero indicator removes both rows from the fluctuation and the curve.
plant <- function(y, eligible) {
  rows <- which(eligible)
  if (length(rows) < 2L) stop("fewer than two rows are available to plant the outcome scale")
  y[rows[[1]]] <- q_bounds[[1]]
  y[rows[[2]]] <- q_bounds[[2]]
  y
}

assert_scale <- function(y, a, q, delta, family, expected) {
  stage <- tmle:::.initStage1(y, a, q, NULL, delta, expected, alpha, TRUE, family)
  if (!identical(as.numeric(stage$ab), as.numeric(expected))) {
    stop("tmle did not take the declared outcome scale")
  }
}

fit_one <- function(frame) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  family <- laws[[scenario]]
  if (is.null(family)) stop(sprintf("unknown scenario %s", scenario))
  check_payload(frame)
  n <- nrow(frame)
  delta <- frame$Delta
  y <- frame$Y
  qn <- cbind(frame$qn, frame$qn)
  if (family == "gaussian") {
    y <- plant(y, delta == 0)
  }
  assert_scale(y, rep(1, n), qn, delta, family, q_bounds)

  # A constant synthetic treatment selects tmle's missing-outcome population-mean branch.
  # Duplicating the realized-arm predictions makes QAW, Q1W, and pDelta1 equal to the
  # supplied nuisance values for each original (A, W) row.
  fit <- tmle::tmle(
    Y = y,
    A = rep(1, n),
    W = data.frame(A_original = frame$A, W = frame$W),
    Delta = delta,
    Q = qn,
    g1W = rep(1, n),
    pDelta1 = cbind(frame$pin, frame$pin),
    family = family,
    fluctuation = "logistic",
    Qbounds = q_bounds,
    gbound = gbound,
    alpha = alpha,
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
    implementation = implementation,
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
rm(samples)
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
  key = c("scenario", "replicate"),
  versions = c(
    study_version("tmle"),
    paste("source sha256", Sys.getenv("TMLE_SHA256"))
  )
)
