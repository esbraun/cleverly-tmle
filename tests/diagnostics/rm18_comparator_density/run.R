# The R half of the declared RM18 density-ratio 2x2. No registered runner changes.
suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/tests/canonical/lmtp_point_adapter.R")
source("/fixture/tests/canonical/study_harness.R")
options(digits = 17)

paths <- study_arguments("usage: run.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
scenario <- "continuous_modified_policy"
target <- "ate_shift[+0.25 vs natural course]"
covariates <- c("W1", "W2", "W3")
z <- qnorm(0.975)

# Keep the registered outcome learner byte-for-byte in behavior. The canonical runner's
# wrapper adds A^2 to both observed and shifted designs before fitting the ordinary GLM.
SL.glm.quadratic <- function(Y, X, newX, family, obsWeights, id, ...) {
  X <- as.data.frame(X)
  newX <- as.data.frame(newX)
  X$A_squared <- X$A^2
  newX$A_squared <- newX$A^2
  out <- SuperLearner::SL.glm(
    Y = Y, X = X, newX = newX, family = family,
    obsWeights = obsWeights, id = id, ...
  )
  class(out$fit) <- c("SL.glm.quadratic", class(out$fit))
  out
}

predict.SL.glm.quadratic <- function(object, newdata, ...) {
  newdata <- as.data.frame(newdata)
  newdata$A_squared <- newdata$A^2
  stats::predict(object$object, newdata = newdata, type = "response")
}

truth_for <- function(replicate) {
  selected <- truths$replicate == replicate & truths$scenario == scenario &
    truths$estimand == target
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows", sum(selected)))
  truths$truth[selected]
}

fit_policy <- function(frame, delta, ratio) {
  shifted <- frame[c(covariates, "A", "Y")]
  shifted$A <- frame$A + delta
  fit <- lmtp_point_tmle(
    frame[c(covariates, "A", "Y")], shifted, ratio,
    mtp = TRUE, outcome_type = "continuous", bounds = range(frame$Y),
    W = covariates, learners_outcome = "SL.glm.quadratic"
  )
  list(estimate = fit$estimate@x, initial = fit$initial, ic = fit$estimate@eif)
}

contrast <- function(shifted, natural) {
  list(
    estimate = shifted$estimate - natural$estimate,
    initial = shifted$initial - natural$initial,
    ic = shifted$ic - natural$ic
  )
}

row_for <- function(replicate, cell, fit, n) {
  truth <- truth_for(replicate)
  standard_error <- sd(fit$ic) / sqrt(n)
  low <- fit$estimate - z * standard_error
  high <- fit$estimate + z * standard_error
  data.frame(
    cell = cell, scenario = scenario, replicate = replicate, n = n,
    estimand = target, truth = truth, estimate = fit$estimate,
    inference_estimate = fit$estimate, std_error = standard_error,
    ci_lower = low, ci_upper = high, inference_scale = "identity",
    covered = as.integer(low <= truth && truth <= high),
    initial_estimate = fit$initial, stringsAsFactors = FALSE, check.names = FALSE
  )
}

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  n <- nrow(frame)
  if (n != 2000 || any(frame$replicate != replicate)) {
    stop("the redrawn sample does not have 2,000 aligned rows")
  }
  mean_a <- 2 + 0.7 * frame$W1 - 0.3 * frame$W2
  analytic <- dnorm(frame$A - 0.25, mean = mean_a, sd = 1) /
    dnorm(frame$A, mean = mean_a, sd = 1)
  hazard <- frame$hazard_ratio_quarter
  if (length(hazard) != n || any(!is.finite(hazard)) || any(hazard < 0)) {
    stop("the supplied pooled-hazard ratio is invalid")
  }
  natural <- fit_policy(frame, 0, rep(1, n))
  quarter_analytic <- fit_policy(frame, 0.25, analytic)
  quarter_hazard <- fit_policy(frame, 0.25, hazard)
  rbind(
    row_for(replicate, "R-A", contrast(quarter_analytic, natural), n),
    row_for(replicate, "R-H", contrast(quarter_hazard, natural), n)
  )
}

expected <- length(unique(truths$replicate))
if (!expected %in% c(1L, 800L)) stop("the diagnostic runner accepts one smoke or 800 primary pairs")
results <- study_stream(paths$samples, fit_one, expected)
study_collect(results, expected, paths$output, versions = study_version("lmtp"))
