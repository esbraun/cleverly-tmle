suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_crossfit_adapter.R")
source("/fixture/study_harness.R")
source("/fixture/ltmle_regimen_adapter.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)

scenario <- "censored_survival_curve"
dynamic_label <- "treat then continue if l2 positive"
z <- qnorm(0.975)


exact_ratios <- function(frame, arms, horizon) {
  # The law's own mechanism, node by node, given to lmtp because it has no `gform` argument
  # to be handed one through.  Cleverly receives the same numbers via
  # KnownLongitudinalMechanism, so the paired comparison measures the survival recursion, the
  # targeting and the influence curve rather than two mechanism-fitting pipelines.
  #
  # Per node, not cumulative: column t is zero exactly where the unit left the path at that
  # node, and lmtp multiplies the columns downstream.  The adapter re-checks both halves.
  a1 <- arms[, 1]
  p_a1 <- plogis(0.3 * frame$W1 - 0.4 * frame$W2)
  g1 <- ifelse(a1 == 1, p_a1, 1 - p_a1)
  c1 <- plogis(2.2 + 0.3 * frame$W1 - 0.3 * a1)
  followed1 <- frame$A1 == a1 & frame$C1 == 1
  first <- ifelse(followed1, 1 / (g1 * c1), 0)
  if (horizon == 1) {
    return(matrix(first, ncol = 1))
  }

  a2 <- arms[, 2]
  l2 <- ifelse(is.na(frame$L2), 0, frame$L2)
  p_a2 <- plogis(0.5 * l2 + 0.6 * a1 - 0.2 * frame$W2)
  g2 <- ifelse(a2 == 1, p_a2, 1 - p_a2)
  c2 <- plogis(2.4 + 0.2 * l2)
  # A unit that had the event at the first node has no second-node arm.  ``A2`` is ``NA``
  # there, which is what removes it from the second column rather than a separate mask.
  followed2 <- !is.na(frame$A2) & frame$A2 == a2 & !is.na(frame$C2) & frame$C2 == 1
  cbind(first, ifelse(followed2, 1 / (g2 * c2), 0))
}

fit_plan <- function(frame, label, horizon) {
  if (horizon == 1) {
    natural <- frame[c("W1", "W2", "A1", "C1", "Y1")]
    trt <- "A1"
    outcome <- "Y1"
    time_vary <- list(NULL)
    cens <- "C1"
  } else {
    natural <- frame[c("W1", "W2", "A1", "C1", "Y1", "L2", "A2", "C2", "Y2")]
    trt <- c("A1", "A2")
    outcome <- c("Y1", "Y2")
    time_vary <- list(NULL, "L2")
    cens <- c("C1", "C2")
  }
  shifted <- natural
  arms <- regimen_arms(frame, label, horizon)
  shifted$A1 <- arms[, 1]
  if (horizon == 2) shifted$A2 <- arms[, 2]
  fit <- lmtp_tmle_with_folds(
    natural,
    shifted,
    trt = trt,
    outcome = outcome,
    baseline = c("W1", "W2"),
    time_vary = time_vary,
    cens = cens,
    # lmtp requires at least two event nodes for outcome_type="survival".  At the first
    # horizon cumulative risk is the one-node binary mean, so its binomial path is the same
    # parameter and the horizon-two call below exercises the absorbing-event recursion.
    outcome_type = if (horizon == 1) "binomial" else "survival",
    fold_assignment = frame$fold,
    learners_outcome = "SL.glm",
    learners_trt = "SL.glm",
    density_ratios = exact_ratios(frame, arms, horizon),
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
  if (horizon == 1) {
    list(estimate = fit$estimate@x, initial = mean(fit$initial), ic = fit$estimate@eif)
  } else {
    # lmtp's survival path reports event-free survival. Cleverly reports cumulative risk,
    # so apply the exact one-minus transformation and its influence-curve sign change.
    list(
      estimate = 1 - fit$estimate@x,
      initial = 1 - mean(fit$initial),
      ic = -fit$estimate@eif
    )
  }
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
    implementation = "lmtp",
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
    initial_estimate = initial,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  fits <- list(
    "never @ t=1" = fit_plan(frame, "never", 1),
    "always @ t=1" = fit_plan(frame, "always", 1),
    "never @ t=2" = fit_plan(frame, "never", 2),
    "always @ t=2" = fit_plan(frame, "always", 2),
    "dynamic @ t=2" = fit_plan(frame, dynamic_label, 2)
  )
  levels <- list(
    c("never", "1", "never @ t=1"),
    c("never", "2", "never @ t=2"),
    c("always", "1", "always @ t=1"),
    c("always", "2", "always @ t=2"),
    c(dynamic_label, "2", "dynamic @ t=2")
  )
  rows <- lapply(levels, function(spec) {
    fit <- fits[[spec[[3]]]]
    row_for(
      replicate,
      sprintf("risk_regimen[%s @ t=%s]", spec[[1]], spec[[2]]),
      fit$estimate,
      fit$initial,
      fit$ic,
      nrow(frame)
    )
  })
  comparisons <- list(
    c("always", "1", "always @ t=1", "never @ t=1"),
    c("always", "2", "always @ t=2", "never @ t=2"),
    c(dynamic_label, "2", "dynamic @ t=2", "never @ t=2")
  )
  for (spec in comparisons) {
    left <- fits[[spec[[3]]]]
    right <- fits[[spec[[4]]]]
    rows[[length(rows) + 1]] <- row_for(
      replicate,
      sprintf("ate_regimen[%s vs never @ t=%s]", spec[[1]], spec[[2]]),
      left$estimate - right$estimate,
      left$initial - right$initial,
      left$ic - right$ic,
      nrow(frame)
    )
  }
  do.call(rbind, rows)
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(results, expected, paths$output, versions = study_version("lmtp"))
