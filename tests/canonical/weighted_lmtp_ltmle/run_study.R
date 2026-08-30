suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_crossfit_adapter.R")
source("/fixture/lmtp_weighted_glm_adapter.R")
source("/fixture/study_harness.R")
source("/fixture/ltmle_regimen_adapter.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
scenario <- "selected_censored_end_of_study"
dynamic_label <- "treat then continue if l2 positive"
plans <- c("never", "always", dynamic_label)
z <- qnorm(0.975)

exact_ratios <- function(frame, arms) {
  a1 <- arms[, 1]
  a2 <- arms[, 2]
  l2 <- ifelse(is.na(frame$L2), 0, frame$L2)
  p_a1 <- plogis(0.3 * frame$W1 - 0.4 * frame$W2)
  g1 <- ifelse(a1 == 1, p_a1, 1 - p_a1)
  c1 <- plogis(2.2 + 0.3 * frame$W1 - 0.3 * a1)
  p_a2 <- plogis(0.5 * l2 + 0.6 * a1 - 0.2 * frame$W2)
  g2 <- ifelse(a2 == 1, p_a2, 1 - p_a2)
  c2 <- plogis(2.4 + 0.2 * l2)
  followed1 <- frame$A1 == a1 & frame$C1 == 1
  followed2 <- !is.na(frame$A2) & frame$A2 == a2 & !is.na(frame$C2) & frame$C2 == 1
  cbind(
    ifelse(followed1, 1 / (g1 * c1), 0),
    ifelse(followed2, 1 / (g2 * c2), 0)
  )
}

fit_plan <- function(frame, label) {
  natural <- frame[c("W1", "W2", "A1", "C1", "L2", "A2", "C2", "Y")]
  natural$obs_weight_aux <- frame$obs_weight
  shifted <- natural
  arms <- regimen_arms(frame, label)
  shifted$A1 <- arms[, 1]
  shifted$A2 <- arms[, 2]
  fit <- lmtp_tmle_with_folds(
    natural, shifted, trt = c("A1", "A2"), outcome = "Y",
    baseline = c("W1", "W2", "obs_weight_aux"), time_vary = list(NULL, "L2"),
    cens = c("C1", "C2"), weights = frame$obs_weight, outcome_type = "binomial",
    fold_assignment = frame$fold, learners_outcome = "SL.weighted.glm",
    learners_trt = "SL.glm", density_ratios = exact_ratios(frame, arms),
    control = lmtp_control(
      .trim = 1, .learners_outcome_folds = 2, .learners_trt_folds = 2,
      .return_full_fits = TRUE
    )
  )
  if (!identical(fit$fold_assignment, as.integer(frame$fold))) {
    stop("lmtp did not retain the supplied fold assignment")
  }
  list(estimate = fit$estimate@x, initial = weighted.mean(fit$initial, frame$obs_weight), ic = fit$estimate@eif)
}

truth_for <- function(replicate, estimand) {
  selected <- truths$replicate == replicate & truths$scenario == scenario & truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows", sum(selected)))
  truths$truth[selected]
}

row_for <- function(replicate, name, estimate, initial, ic, n) {
  truth <- truth_for(replicate, name)
  standard_error <- sd(ic) / sqrt(n)
  low <- estimate - z * standard_error
  high <- estimate + z * standard_error
  data.frame(
    implementation = "lmtp-weighted", scenario = scenario, replicate = replicate, n = n,
    estimand = name, truth = truth, estimate = estimate, inference_estimate = estimate,
    std_error = standard_error, ci_lower = low, ci_upper = high, inference_scale = "identity",
    covered = as.integer(low <= truth && truth <= high), initial_estimate = initial,
    stringsAsFactors = FALSE, check.names = FALSE
  )
}

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  fits <- setNames(lapply(plans, function(label) fit_plan(frame, label)), plans)
  rows <- lapply(plans, function(label) {
    fit <- fits[[label]]
    row_for(replicate, sprintf("ey_regimen[%s]", label), fit$estimate, fit$initial, fit$ic, nrow(frame))
  })
  for (label in c("always", dynamic_label)) {
    left <- fits[[label]]
    right <- fits[["never"]]
    rows[[length(rows) + 1]] <- row_for(
      replicate, sprintf("ate_regimen[%s vs never]", label), left$estimate - right$estimate,
      left$initial - right$initial, left$ic - right$ic, nrow(frame)
    )
  }
  do.call(rbind, rows)
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(results, expected, paths$output, versions = study_version("lmtp"))
