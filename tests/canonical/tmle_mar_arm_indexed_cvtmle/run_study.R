suppressPackageStartupMessages(library(tmle))
source("/fixture/study_harness.R")
source("/fixture/tmle_point_adapter.R")
source("/fixture/tmle_continuous_point_adapter.R")
source("/fixture/multi_arm_helpers.R")
options(digits = 17)

args <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- as.data.frame(data.table::fread(cmd = paste("gzip -dc", shQuote(args$samples))))
truths <- read.csv(args$truths, stringsAsFactors = FALSE, check.names = FALSE)

implementation <- "tmle-r-stitched-arm-indexed"
# The known continuous support, which the Python fit declares as q_bounds.
q_bounds <- c(-2, 8)
# R turns a scalar gbound into c(gbound, 1).  Every supplied product g_a pi_a must stay above
# it, so R clips nothing that cleverly does not.
gbound <- 0.001
alpha <- 0.9995
# CausalData sorts the three-arm labels, so arm codes 0, 1, and 2 are these labels.  The
# first is the reference arm of every contrast, which multi_arm_rows_from_moments reads.
three_arm_labels <- c("high", "low", "mid")
laws <- list(
  binary_mar_two_arm_stacked = list(arms = 2L, continuous = FALSE),
  continuous_mar_two_arm_stacked = list(arms = 2L, continuous = TRUE),
  binary_mar_three_arm_stacked = list(arms = 3L, continuous = FALSE),
  continuous_mar_three_arm_stacked = list(arms = 3L, continuous = TRUE)
)

check_payload <- function(frame, arms) {
  if (!setequal(unique(frame$fold), 0:9)) {
    stop("the supplied payload does not contain the fitted ten-fold assignment")
  }
  if (!setequal(unique(frame$A), 0:(arms - 1L))) {
    stop("the payload does not retain every treatment arm")
  }
  for (code in 0:(arms - 1L)) {
    g <- frame[[paste0("g", code)]]
    p <- frame[[paste0("pi", code)]]
    q <- frame[[paste0("q", code)]]
    if (!all(is.finite(g)) || !all(is.finite(p)) || !all(is.finite(q))) {
      stop("the supplied nuisance predictions are not finite")
    }
    if (any(g * p < gbound)) {
      stop("a supplied product g_a pi_a falls below gbound, so R would clip it")
    }
  }
}

# R takes the outcome scale from every non-NA Y (tmle.R line 1120).  Setting Y to the two
# q_bounds on two rows whose response indicator is zero fixes the scale at q_bounds, and
# the zero indicator removes both rows from the fluctuation and the curves.
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

assert_supplied <- function(fit) {
  if (fit$Qinit$type != "user-supplied values" ||
      fit$g$type != "user-supplied values" ||
      fit$g.Delta$type != "user-supplied values") {
    stop("tmle refitted a nuisance instead of using the supplied predictions")
  }
}

fit_tmle <- function(y, a, w, delta, q, g1, p, continuous) {
  tmle::tmle(
    Y = y,
    A = a,
    W = w,
    Delta = delta,
    Q = q,
    g1W = g1,
    pDelta1 = p,
    family = if (continuous) "gaussian" else "binomial",
    fluctuation = "logistic",
    Qbounds = if (continuous) q_bounds else c(0, 1),
    gbound = gbound,
    alpha = alpha,
    cvQinit = FALSE,
    prescreenW.g = FALSE,
    target.gwt = FALSE,
    B = 1,
    evalATT = FALSE,
    verbose = FALSE
  )
}

fit_two_arm <- function(frame, scenario, replicate, continuous) {
  n <- nrow(frame)
  qn <- cbind(frame$q0, frame$q1)
  delta <- frame$Delta
  y <- frame$Y
  if (continuous) {
    y <- plant(y, delta == 0)
    assert_scale(y, frame$A, qn, delta, "gaussian", q_bounds)
  } else {
    assert_scale(y, frame$A, qn, delta, "binomial", c(0, 1))
  }
  fit <- fit_tmle(
    y, frame$A, data.frame(W = frame$W), delta, qn, frame$g1, cbind(frame$pi0, frame$pi1),
    continuous
  )
  assert_supplied(fit)
  if (continuous) {
    tmle_continuous_point_rows(fit, qn, rep(1, n), truths, scenario, replicate, implementation)
  } else {
    tmle_point_rows(fit, qn, rep(1, n), truths, scenario, replicate, implementation)
  }
}

# The continuous three-arm law reports arm means and reference differences only, so the
# shared builder's ratio rows, which would take logs of means outside (0, 1), are not formed.
continuous_rows_from_moments <- function(
    means, covariance, initial, labels, truth, scenario, replicate, n) {
  names(means) <- names(initial) <- labels
  critical <- stats::qnorm(0.975)
  reference_index <- 1L
  make_row <- function(name, estimate, gradient, initial_estimate) {
    variance <- as.numeric(t(gradient) %*% covariance %*% gradient)
    if (!is.finite(variance) || variance <= 0) {
      stop(sprintf("the %s variance is not positive for %s/%s", name, scenario, replicate))
    }
    standard_error <- sqrt(variance)
    interval <- estimate + c(-1, 1) * critical * standard_error
    target <- unname(truth[[name]])
    data.frame(
      implementation = implementation,
      scenario = scenario,
      replicate = replicate,
      n = n,
      estimand = name,
      truth = target,
      estimate = estimate,
      inference_estimate = estimate,
      std_error = standard_error,
      ci_lower = interval[[1]],
      ci_upper = interval[[2]],
      inference_scale = "identity",
      covered = as.integer(interval[[1]] <= target && target <= interval[[2]]),
      initial_estimate = initial_estimate,
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
  }
  rows <- list()
  for (index in seq_along(labels)) {
    gradient <- rep(0, length(labels))
    gradient[[index]] <- 1
    rows[[length(rows) + 1L]] <- make_row(
      paste0("ey[", labels[[index]], "]"), means[[index]], gradient, initial[[index]]
    )
  }
  for (index in seq_along(labels)[-reference_index]) {
    gradient <- rep(0, length(labels))
    gradient[[index]] <- 1
    gradient[[reference_index]] <- -1
    rows[[length(rows) + 1L]] <- make_row(
      paste0("ate[", labels[[index]], " vs ", labels[[reference_index]], "]"),
      means[[index]] - means[[reference_index]],
      gradient,
      initial[[index]] - initial[[reference_index]]
    )
  }
  do.call(rbind, rows)
}

fit_three_arm <- function(frame, scenario, replicate, continuous) {
  n <- nrow(frame)
  means <- numeric(3)
  initial <- numeric(3)
  influence <- matrix(NA_real_, nrow = n, ncol = 3)
  family <- if (continuous) "gaussian" else "binomial"
  for (code in 0:2) {
    q <- frame[[paste0("q", code)]]
    g <- frame[[paste0("g", code)]]
    p <- frame[[paste0("pi", code)]]
    # One population-mean fit per arm: a constant A selects the EY1-only path, the response
    # indicator keeps only the respondents in this arm, and every other outcome is NA.
    delta <- as.numeric(frame$A == code & frame$Delta == 1)
    y <- ifelse(delta == 1, frame$Y, NA_real_)
    a <- rep(1, n)
    if (continuous) {
      y <- plant(y, delta == 0)
      assert_scale(y, a, cbind(q, q), delta, family, q_bounds)
    } else {
      assert_scale(y, a, cbind(q, q), delta, family, c(0, 1))
    }
    # The original arm stays beside W, so the conditioning set matches the predictions.
    w <- data.frame(A_original = frame$A, W = frame$W)
    fit <- fit_tmle(y, a, w, delta, cbind(q, q), g, cbind(p, p), continuous)
    assert_supplied(fit)
    if (is.null(fit$estimates$EY1) || !is.null(fit$estimates$EY0) ||
        !is.null(fit$estimates$ATE)) {
      stop("tmle did not select its scalar population-mean result shape")
    }
    curve <- as.numeric(fit$estimates$IC$IC.EY1)
    if (length(curve) != n || !all(is.finite(curve))) {
      stop("tmle returned no complete population-mean influence curve")
    }
    means[[code + 1L]] <- as.numeric(fit$estimates$EY1$psi)
    influence[, code + 1L] <- curve
    initial[[code + 1L]] <- mean(q)
  }
  covariance <- stats::cov(influence) / n
  selected <- truths$scenario == scenario & truths$replicate == replicate
  truth <- stats::setNames(as.list(truths$truth[selected]), truths$estimand[selected])
  if (continuous) {
    continuous_rows_from_moments(
      means, covariance, initial, three_arm_labels, truth, scenario, replicate, n
    )
  } else {
    multi_arm_rows_from_moments(
      means, covariance, initial, three_arm_labels, truth, implementation, scenario,
      replicate, n
    )
  }
}

fit_one <- function(frame) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  law <- laws[[scenario]]
  if (is.null(law)) stop(sprintf("unknown scenario %s", scenario))
  check_payload(frame, law$arms)
  if (law$arms == 2L) {
    fit_two_arm(frame, scenario, replicate, law$continuous)
  } else {
    fit_three_arm(frame, scenario, replicate, law$continuous)
  }
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
