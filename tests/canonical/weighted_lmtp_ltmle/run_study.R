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
  expected_weights <- frame$obs_weight / mean(frame$obs_weight)
  if (!isTRUE(all.equal(as.numeric(fit$estimate@weights), expected_weights, tolerance = 1e-12))) {
    stop("lmtp did not retain the normalized observation weights in its influence estimate")
  }
  expected_se <- sd(fit$estimate@eif * fit$estimate@weights) / sqrt(nrow(frame))
  if (!isTRUE(all.equal(fit$estimate@std_error, expected_se, tolerance = 1e-12))) {
    stop("lmtp weighted standard error does not equal the weighted-EIF formula")
  }
  weighted_eif <- fit$estimate@weights * fit$estimate@eif
  if (abs(mean(weighted_eif) - fit$estimate@x) > 0.02) {
    stop(sprintf(
      "lmtp weighted EIF averages %.17g but reports estimate %.17g",
      mean(weighted_eif), fit$estimate@x
    ))
  }
  list(
    estimate = fit$estimate@x,
    initial = weighted.mean(fit$initial, frame$obs_weight),
    ic = fit$estimate@eif,
    weights = fit$estimate@weights,
    id = fit$estimate@id,
    standard_error = fit$estimate@std_error,
    ht_standard_error = sd(weighted_eif) / sqrt(nrow(frame)),
    hajek_standard_error = sd(
      fit$estimate@weights * (fit$estimate@eif - fit$estimate@x)
    ) / sqrt(nrow(frame))
  )
}

truth_for <- function(replicate, estimand) {
  selected <- truths$replicate == replicate & truths$scenario == scenario & truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows", sum(selected)))
  truths$truth[selected]
}

row_for <- function(
  replicate, name, estimate, initial, standard_error, ht_standard_error,
  hajek_standard_error, n
) {
  truth <- truth_for(replicate, name)
  low <- estimate - z * standard_error
  high <- estimate + z * standard_error
  data.frame(
    implementation = "lmtp-weighted", scenario = scenario, replicate = replicate, n = n,
    estimand = name, truth = truth, estimate = estimate, inference_estimate = estimate,
    std_error = standard_error, ci_lower = low, ci_upper = high, inference_scale = "identity",
    covered = as.integer(low <= truth && truth <= high), initial_estimate = initial,
    native_std_error = standard_error, ht_std_error = ht_standard_error,
    hajek_std_error = hajek_standard_error,
    stringsAsFactors = FALSE, check.names = FALSE
  )
}

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  fits <- setNames(lapply(plans, function(label) fit_plan(frame, label)), plans)
  rows <- lapply(plans, function(label) {
    fit <- fits[[label]]
    row_for(
      replicate, sprintf("ey_regimen[%s]", label), fit$estimate, fit$initial,
      fit$standard_error, fit$ht_standard_error, fit$hajek_standard_error, nrow(frame)
    )
  })
  for (label in c("always", dynamic_label)) {
    left <- fits[[label]]
    right <- fits[["never"]]
    if (!isTRUE(all.equal(left$weights, right$weights, tolerance = 0))) {
      stop("regimen influence estimates carry different observation weights")
    }
    if (!identical(left$id, right$id)) {
      stop("regimen influence estimates carry different row identifiers")
    }
    contrast <- ife::ife(
      left$estimate - right$estimate,
      left$ic - right$ic,
      left$weights,
      left$id
    )
    weighted_eif <- left$weights * (left$ic - right$ic)
    ht_standard_error <- sd(weighted_eif) / sqrt(nrow(frame))
    hajek_standard_error <- sd(
      left$weights * ((left$ic - right$ic) - contrast@x)
    ) / sqrt(nrow(frame))
    if (!isTRUE(all.equal(contrast@std_error, ht_standard_error, tolerance = 1e-12))) {
      stop("lmtp contrast standard error does not equal the Horvitz-Thompson formula")
    }
    rows[[length(rows) + 1]] <- row_for(
      replicate, sprintf("ate_regimen[%s vs never]", label), contrast@x,
      left$initial - right$initial, contrast@std_error, ht_standard_error,
      hajek_standard_error, nrow(frame)
    )
  }
  do.call(rbind, rows)
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(results, expected, paths$output, versions = study_version("lmtp"))
