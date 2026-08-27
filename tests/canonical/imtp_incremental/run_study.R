suppressPackageStartupMessages(library(imtp))
future::plan(future::sequential)
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
scenario <- "binary_incremental_odds"
z <- qnorm(0.975)

truth_for <- function(replicate, estimand) {
  selected <- truths$replicate == replicate & truths$scenario == scenario & truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows", sum(selected)))
  truths$truth[selected]
}

fit_delta <- function(frame, delta, seed) {
  set.seed(seed)
  data <- frame[c("W", "A", "Y")]
  data$W1 <- as.numeric(data$W == 1)
  data$W2 <- as.numeric(data$W == 2)
  data$W <- NULL
  imtp_tmle(
    data,
    trt = "A",
    outcome = "Y",
    baseline = c("W1", "W2"),
    delta = delta,
    outcome_type = "binomial",
    learners_outcome = "SL.glm",
    learners_trt = "SL.glm",
    folds = 1,
    .learners_outcome_folds = 2,
    .learners_trt_folds = 2,
    .return_full_fits = FALSE
  )
}

as_parameter <- function(fit) {
  list(estimate = fit$theta, initial = fit$theta, ic = fit$eif)
}

row_for <- function(replicate, name, fit, n) {
  truth <- truth_for(replicate, name)
  standard_error <- sd(fit$ic) / sqrt(n)
  low <- fit$estimate - z * standard_error
  high <- fit$estimate + z * standard_error
  data.frame(
    implementation = "imtp",
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
  natural <- as_parameter(fit_delta(frame, 1, 10000 * replicate + 1))
  doubled <- as_parameter(fit_delta(frame, 2, 10000 * replicate + 2))
  halved <- as_parameter(fit_delta(frame, 0.5, 10000 * replicate + 3))
  doubled_contrast <- list(
    estimate = doubled$estimate - natural$estimate,
    initial = doubled$initial - natural$initial,
    ic = doubled$ic - natural$ic
  )
  halved_contrast <- list(
    estimate = halved$estimate - natural$estimate,
    initial = halved$initial - natural$initial,
    ic = halved$ic - natural$ic
  )
  rbind(
    row_for(replicate, "ey_ipsi[natural course]", natural, nrow(frame)),
    row_for(replicate, "ey_ipsi[odds x2]", doubled, nrow(frame)),
    row_for(replicate, "ey_ipsi[odds x0.5]", halved, nrow(frame)),
    row_for(
      replicate,
      "ate_ipsi[odds x2 vs natural course]",
      doubled_contrast,
      nrow(frame)
    ),
    row_for(
      replicate,
      "ate_ipsi[odds x0.5 vs natural course]",
      halved_contrast,
      nrow(frame)
    )
  )
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(results, expected, paths$output, versions = study_version("imtp"))
