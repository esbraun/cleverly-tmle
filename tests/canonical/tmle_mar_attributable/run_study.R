suppressPackageStartupMessages(library(tmle))
source("/fixture/study_harness.R")
options(digits = 17)

# R tmle 2.1.1 reports no PAR or PAF.  This runner composes them from two shipped paths on
# the same rows and the same supplied predictions:
#
#   * the natural course: the population-mean path, with A = rep(1, n), the realized-arm
#     outcome prediction twice and the realized-arm response prediction twice;
#   * each arm a: the population-mean path with the response indicator 1{A = a} Delta, so
#     the clever covariate is 1{A = a} Delta / (g_a pi_a), the arm-mean covariate.
#
# Every estimate, and each curve PAR and PAF are formed from, is R's own:
# fit$estimates$EY1$psi and fit$estimates$IC$IC.EY1.  R averages a curve within each id and
# reports var(IC) / n.id, so the composed variance reads the same cluster means.

args <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- as.data.frame(data.table::fread(cmd = paste("gzip -dc", shQuote(args$samples))))
truths <- read.csv(args$truths, stringsAsFactors = FALSE, check.names = FALSE)
truth_key <- paste(truths$scenario, truths$replicate, truths$estimand, sep = "|")

implementation <- "tmle-r-composed-attributable"
q_bounds <- c(0, 1)
gbound <- 0.01
alpha <- 0.9995
critical <- stats::qnorm(0.975)
# CausalData sorts the three-arm labels, so code 0 is "high", 1 "low" and 2 "mid".
three_arm_labels <- c("high", "low", "mid")
continuous_scenario <- "continuous_mar_attributable"

check_payload <- function(frame, arms) {
  for (column in c("qn", paste0("q", arms))) {
    values <- frame[[column]]
    if (!all(is.finite(values)) || !all(values >= 0 & values <= 1)) {
      stop(sprintf("the supplied %s predictions are not finite values in [0, 1]", column))
    }
  }
  if (!all(is.finite(frame$pin)) || any(frame$pin <= gbound) || any(frame$pin > 1)) {
    stop("a supplied realized-arm response prediction is outside (gbound, 1]")
  }
  for (arm in arms) {
    product <- frame[[paste0("g", arm)]] * frame[[paste0("pi", arm)]]
    if (!all(is.finite(product)) || any(product <= gbound)) {
      stop(sprintf("the product g * pi at arm %s is at or below gbound, so R would clip it", arm))
    }
  }
  if (!all(is.finite(frame$weight)) || any(frame$weight <= 0)) {
    stop("the supplied observation weights are not finite and positive")
  }
}

plant <- function(y, eligible) {
  rows <- which(eligible)
  if (length(rows) < 2L) stop("fewer than two rows are available to plant the outcome scale")
  y[rows[[1]]] <- q_bounds[[1]]
  y[rows[[2]]] <- q_bounds[[2]]
  y
}

assert_scale <- function(y, a, q, delta, family) {
  expected <- q_bounds
  stage <- tmle:::.initStage1(y, a, q, NULL, delta, expected, alpha, TRUE, family)
  if (!identical(as.numeric(stage$ab), as.numeric(expected))) {
    stop("tmle did not take the declared outcome scale")
  }
}

# One population-mean fit.  `delta` selects the rows whose outcome the fit reads.
population_mean <- function(frame, y, delta, q, g1, p, family) {
  n <- nrow(frame)
  a <- rep(1, n)
  if (family == "gaussian") {
    y <- plant(y, delta == 0)
  }
  assert_scale(y, a, cbind(q, q), delta, family)
  fit <- tmle::tmle(
    Y = y,
    A = a,
    W = data.frame(A_original = frame$A, W = frame$W),
    Delta = delta,
    Q = cbind(q, q),
    g1W = g1,
    pDelta1 = cbind(p, p),
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
    id = frame$cluster,
    obsWeights = frame$weight,
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
  if (is.null(fit$estimates$IC)) stop("tmle returned no influence curves")
  curve <- as.numeric(fit$estimates$IC$IC.EY1)
  if (!all(is.finite(curve))) stop("tmle returned a non-finite population-mean curve")
  list(psi = as.numeric(fit$estimates$EY1$psi), curve = curve)
}

weighted_mean <- function(values, weights) sum(values * weights) / sum(weights)

fit_one <- function(frame) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  n <- nrow(frame)
  # The archive holds every scenario, so a two-arm replication carries empty arm-2 columns.
  columns <- grep("^q[0-9]+$", names(frame), value = TRUE)
  columns <- columns[vapply(columns, function(name) !all(is.na(frame[[name]])), logical(1))]
  arms <- sort(as.integer(sub("^q", "", columns)))
  check_payload(frame, arms)
  family <- if (scenario == continuous_scenario) "gaussian" else "binomial"
  reference <- as.integer(frame$ref[[1]])
  three_arm <- length(arms) == 3L

  natural <- population_mean(
    frame, frame$Y, frame$Delta, frame$qn, rep(1, n), frame$pin, family
  )
  means <- numeric(length(arms))
  curves <- matrix(NA_real_, nrow = length(natural$curve), ncol = length(arms) + 1L)
  curves[, 1L] <- natural$curve
  initial <- numeric(length(arms))
  for (index in seq_along(arms)) {
    arm <- arms[[index]]
    delta <- as.numeric(frame$A == arm & frame$Delta == 1)
    y <- ifelse(delta == 1, frame$Y, NA_real_)
    fit <- population_mean(
      frame, y, delta, frame[[paste0("q", arm)]], frame[[paste0("g", arm)]],
      frame[[paste0("pi", arm)]], family
    )
    if (length(fit$curve) != nrow(curves)) stop("the arm and natural-course curves differ in length")
    means[[index]] <- fit$psi
    curves[, index + 1L] <- fit$curve
    initial[[index]] <- weighted_mean(frame[[paste0("q", arm)]], frame$weight)
  }
  units <- nrow(curves)
  psi_obs <- natural$psi
  initial_obs <- weighted_mean(frame$qn, frame$weight)
  ref_index <- match(reference, arms)
  psi_ref <- means[[ref_index]]
  initial_ref <- initial[[ref_index]]

  arm_name <- function(index) {
    if (three_arm) paste0("ey[", three_arm_labels[[arms[[index]] + 1L]], "]") else paste0("ey", arms[[index]])
  }
  suffix <- if (three_arm) paste0("[", three_arm_labels[[reference + 1L]], "]") else ""
  rows <- list()
  add <- function(name, estimate, gradient, initial_estimate) {
    combined <- as.numeric(curves %*% gradient)
    standard_error <- sqrt(stats::var(combined) / units)
    if (!is.finite(standard_error) || standard_error <= 0) {
      stop(sprintf("the %s standard error is not positive for %s/%s", name, scenario, replicate))
    }
    truth_row <- match(paste(scenario, replicate, name, sep = "|"), truth_key)
    if (is.na(truth_row)) stop(sprintf("missing truth for %s/%s/%s", scenario, replicate, name))
    truth <- truths$truth[[truth_row]]
    low <- estimate - critical * standard_error
    high <- estimate + critical * standard_error
    published <- function() {
      data.frame(
        implementation = implementation,
        scenario = scenario,
        replicate = replicate,
        n = n,
        estimand = name,
        truth = truth,
        estimate = estimate,
        inference_estimate = estimate,
        std_error = standard_error,
        ci_lower = low,
        ci_upper = high,
        inference_scale = "identity",
        covered = as.integer(low <= truth && truth <= high),
        initial_estimate = initial_estimate,
        check.names = FALSE,
        stringsAsFactors = FALSE
      )
    }
    rows[[length(rows) + 1L]] <<- published()
  }
  width <- length(arms) + 1L
  unit <- function(position) {
    gradient <- rep(0, width)
    gradient[[position]] <- 1
    gradient
  }
  add("ey_obs", psi_obs, unit(1L), initial_obs)
  # The binary laws report the reference arm only; the three-arm law reports every arm.
  reported <- if (three_arm) seq_along(arms) else ref_index
  for (index in reported) add(arm_name(index), means[[index]], unit(index + 1L), initial[[index]])
  gradient <- unit(1L) - unit(ref_index + 1L)
  add(paste0("par", suffix), psi_obs - psi_ref, gradient, initial_obs - initial_ref)
  if (family == "binomial") {
    gradient <- rep(0, width)
    gradient[[1L]] <- psi_ref / psi_obs^2
    gradient[[ref_index + 1L]] <- -1 / psi_obs
    add(paste0("paf", suffix), 1 - psi_ref / psi_obs, gradient, 1 - initial_ref / initial_obs)
  }
  do.call(rbind, rows)
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
