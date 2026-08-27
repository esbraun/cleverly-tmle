suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_point_adapter.R")
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
scenario <- "binary_dynamic_rule"
z <- qnorm(0.975)

truth_for <- function(replicate, estimand) {
  selected <- truths$replicate == replicate & truths$scenario == scenario & truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows", sum(selected)))
  truths$truth[selected]
}

fit_policy <- function(frame, label) {
  assigned <- if (label == "never") {
    rep(0, nrow(frame))
  } else {
    ifelse(frame$W == 1, 0, 1)
  }
  shifted <- frame[c("W", "A", "Y")]
  shifted$A <- assigned
  g1 <- c(0.40, 0.60, 0.25)[as.integer(frame$W) + 1L]
  assigned_probability <- ifelse(assigned == 1, g1, 1 - g1)
  ratio <- ifelse(frame$A == assigned, 1 / assigned_probability, 0)
  fit <- lmtp_point_tmle(
    frame[c("W", "A", "Y")],
    shifted,
    ratio,
    mtp = FALSE,
    outcome_type = "binomial"
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
  never <- fit_policy(frame, "never")
  rule <- fit_policy(frame, "rule")
  contrast <- list(
    estimate = rule$estimate - never$estimate,
    initial = rule$initial - never$initial,
    ic = rule$ic - never$ic
  )
  rbind(
    row_for(replicate, "ey_regime[never]", never, nrow(frame)),
    row_for(replicate, "ey_regime[rule]", rule, nrow(frame)),
    row_for(replicate, "ate_regime[rule vs never]", contrast, nrow(frame))
  )
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(results, expected, paths$output, versions = study_version("lmtp"))
