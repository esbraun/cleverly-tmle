suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_crossfit_adapter.R")
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
scenario <- "clustered_continuous"

truth_for <- function(replicate, estimand) {
  selected <- truths$replicate == replicate & truths$scenario == scenario & truths$estimand == estimand
  if (sum(selected) != 1L) stop(sprintf("truth join found %d rows", sum(selected)))
  truths$truth[selected]
}

fit_arm <- function(frame, arm) {
  natural <- frame[c("W1", "W2", "A", "Y", "cluster")]
  shifted <- natural
  shifted$A <- arm
  p1 <- plogis(0.3 * frame$W1 + 0.6 * frame$W2)
  assigned_probability <- if (arm == 1) p1 else 1 - p1
  ratio <- matrix(ifelse(frame$A == arm, 1 / assigned_probability, 0), ncol = 1L)
  fit <- lmtp_tmle_with_folds(
    natural,
    shifted,
    trt = "A",
    outcome = "Y",
    baseline = c("W1", "W2"),
    id = "cluster",
    outcome_type = "continuous",
    fold_assignment = frame$fold,
    learners_outcome = "SL.glm",
    learners_trt = "SL.glm",
    density_ratios = ratio,
    control = lmtp_control(
      .trim = 1,
      .learners_outcome_folds = 2,
      .learners_trt_folds = 2,
      .return_full_fits = TRUE
    )
  )
  if (!identical(fit$fold_assignment, as.integer(frame$fold))) {
    stop("lmtp did not retain the supplied fold assignment")
  }
  if (!identical(fit$estimate@id, as.character(frame$cluster))) {
    stop("the ife estimate did not retain the cluster identifier")
  }
  list(estimate = fit$estimate, initial = mean(fit$initial))
}

row_for <- function(replicate, name, fit, n) {
  truth <- truth_for(replicate, name)
  low <- fit$estimate@conf_int[[1]]
  high <- fit$estimate@conf_int[[2]]
  data.frame(
    implementation = "lmtp",
    scenario = scenario,
    replicate = replicate,
    n = n,
    estimand = name,
    truth = truth,
    estimate = fit$estimate@x,
    inference_estimate = fit$estimate@x,
    std_error = fit$estimate@std_error,
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
  zero <- fit_arm(frame, 0)
  one <- fit_arm(frame, 1)
  # Arithmetic on the two ife objects uses their joint rowwise EIF. Because both objects
  # retain the same id vector, ife computes the contrast's standard error after aggregation
  # at the cluster level.
  ate <- list(estimate = one$estimate - zero$estimate, initial = one$initial - zero$initial)
  rbind(
    row_for(replicate, "ey0", zero, nrow(frame)),
    row_for(replicate, "ey1", one, nrow(frame)),
    row_for(replicate, "ate", ate, nrow(frame))
  )
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(results, expected, paths$output, versions = c(study_version("lmtp"), study_version("ife")))
