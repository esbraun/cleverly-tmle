suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_point_adapter.R")
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
scenario <- "continuous_modified_policy"
z <- qnorm(0.975)
covariates <- c("W1", "W2", "W3")

# The primary law contains A^2 so an uncapped shift does not collapse to a constant
# effect.  Register that one declared feature in both the observed and shifted designs;
# everything else remains SuperLearner's ordinary Gaussian GLM.
SL.glm.quadratic <- function(Y, X, newX, family, obsWeights, id, ...) {
  X <- as.data.frame(X)
  newX <- as.data.frame(newX)
  X$A_squared <- X$A^2
  newX$A_squared <- newX$A^2
  out <- SuperLearner::SL.glm(
    Y = Y,
    X = X,
    newX = newX,
    family = family,
    obsWeights = obsWeights,
    id = id,
    ...
  )
  class(out$fit) <- c("SL.glm.quadratic", class(out$fit))
  out
}

predict.SL.glm.quadratic <- function(object, newdata, ...) {
  newdata <- as.data.frame(newdata)
  newdata$A_squared <- newdata$A^2
  stats::predict(object$object, newdata = newdata, type = "response")
}

truth_for <- function(replicate, estimand) {
  selected <- truths$replicate == replicate & truths$scenario == scenario & truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows", sum(selected)))
  truths$truth[selected]
}

fit_policy <- function(frame, delta, cap = NULL) {
  shifted <- frame[c(covariates, "A", "Y")]
  if (is.null(cap)) {
    shifted$A <- frame$A + delta
  } else {
    shifted$A <- ifelse(frame$A + delta > cap, frame$A, frame$A + delta)
  }
  mean_a <- 2 + 0.7 * frame$W1 - 0.3 * frame$W2
  if (delta == 0) {
    ratio <- rep(1, nrow(frame))
  } else {
    shifted_density <- dnorm(frame$A - delta, mean = mean_a, sd = 1)
    natural_density <- dnorm(frame$A, mean = mean_a, sd = 1)
    ratio <- shifted_density / natural_density
    if (!is.null(cap)) {
      ratio <- ratio * (frame$A <= cap) + (frame$A > cap - delta)
    }
  }
  fit <- lmtp_point_tmle(
    frame[c(covariates, "A", "Y")],
    shifted,
    ratio,
    mtp = TRUE,
    outcome_type = "continuous",
    bounds = range(frame$Y),
    W = covariates,
    learners_outcome = "SL.glm.quadratic"
  )
  list(estimate = fit$estimate@x, initial = fit$initial, ic = fit$estimate@eif)
}

row_for <- function(replicate, name, fit, n) {
  truth <- truth_for(replicate, name)
  standard_error <- sd(fit$ic) / sqrt(n)
  low <- fit$estimate - z * standard_error
  high <- fit$estimate + z * standard_error
  data.frame(
    implementation = "lmtp",
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
    initial_estimate = fit$initial,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  natural <- fit_policy(frame, 0)
  quarter <- fit_policy(frame, 0.25)
  capped <- fit_policy(frame, 0.5, 3)
  quarter_contrast <- list(
    estimate = quarter$estimate - natural$estimate,
    initial = quarter$initial - natural$initial,
    ic = quarter$ic - natural$ic
  )
  capped_contrast <- list(
    estimate = capped$estimate - natural$estimate,
    initial = capped$initial - natural$initial,
    ic = capped$ic - natural$ic
  )
  rbind(
    row_for(replicate, "ey_shift[natural course]", natural, nrow(frame)),
    row_for(replicate, "ey_shift[+0.25]", quarter, nrow(frame)),
    row_for(replicate, "ey_shift[+0.5 capped]", capped, nrow(frame)),
    row_for(
      replicate,
      "ate_shift[+0.25 vs natural course]",
      quarter_contrast,
      nrow(frame)
    ),
    row_for(
      replicate,
      "ate_shift[+0.5 capped vs natural course]",
      capped_contrast,
      nrow(frame)
    )
  )
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(results, expected, paths$output, versions = study_version("lmtp"))
