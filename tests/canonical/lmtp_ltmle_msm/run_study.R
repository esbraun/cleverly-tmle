suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_crossfit_adapter.R")
source("/fixture/study_harness.R")
source("/fixture/ltmle_regimen_adapter.R")
options(digits = 17)

# The cross-fitted counterpart of ltmle_msm/run_study.R. lmtp has no MSM, so each of the four
# regimens is one lmtp_tmle_with_folds fit on the realized fold column, with the exact per-node
# density ratios and SL.glm sequential regressions of lmtp_ltmle/run_study.R. The coefficients
# are the fixed weighted least-squares projection of the four regimen estimates, and their
# influence curves are the same projection of the four joint per-unit EIFs.

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)

scenario <- "cross_fitted_regimen_projection"
dynamic_label <- "treat then continue if l2 positive"
regimens <- c("never", "always", "early", dynamic_label)
duration <- c(never = 0, always = 2, early = 1)
duration[[dynamic_label]] <- 1
projection_weight <- c(never = 0.1, always = 10, early = 0.1)
projection_weight[[dynamic_label]] <- 10
labels <- c("msm_regimen[(intercept)]", "msm_regimen[duration]")
z <- qnorm(0.975)

exact_ratios <- function(frame, arms) {
  # The law's own per-node density ratios, as lmtp_ltmle/run_study.R writes them.
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
  shifted <- natural
  arms <- regimen_arms(frame, label)
  shifted$A1 <- arms[, 1]
  shifted$A2 <- arms[, 2]
  fit <- lmtp_tmle_with_folds(
    natural,
    shifted,
    trt = c("A1", "A2"),
    outcome = "Y",
    baseline = c("W1", "W2"),
    time_vary = list(NULL, "L2"),
    cens = c("C1", "C2"),
    outcome_type = "binomial",
    fold_assignment = frame$fold,
    learners_outcome = "SL.glm",
    learners_trt = "SL.glm",
    density_ratios = exact_ratios(frame, arms),
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
  list(estimate = fit$estimate@x, initial = mean(fit$initial), ic = fit$estimate@eif)
}

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  fits <- setNames(lapply(regimens, function(label) fit_plan(frame, label)), regimens)
  design <- cbind(1, unname(duration[regimens]))
  weights <- unname(projection_weight[regimens])
  operator <- solve(t(design) %*% (design * weights), t(design * weights))
  estimates <- vapply(fits, `[[`, numeric(1), "estimate")
  initials <- vapply(fits, `[[`, numeric(1), "initial")
  beta <- as.numeric(operator %*% estimates)
  # The identity the paired coefficients rest on: the operator is the weighted least-squares
  # solve, so an independent lm.wfit of the four estimates returns the same coefficients.
  check <- unname(lm.wfit(design, estimates, weights)$coefficients)
  if (max(abs(check - beta)) > 1e-10) {
    stop(sprintf("the projection operator disagrees with lm.wfit by %.3g", max(abs(check - beta))))
  }
  # The joint regimen curves, not four marginal standard errors: every plan is fitted on the
  # same sample, so the coefficient's variance is made of their covariance.
  influence <- do.call(cbind, lapply(fits, `[[`, "ic")) %*% t(operator)
  initial_beta <- as.numeric(operator %*% initials)
  standard_error <- apply(influence, 2, sd) / sqrt(nrow(frame))
  low <- beta - z * standard_error
  high <- beta + z * standard_error

  selected <- truths$replicate == replicate & truths$scenario == scenario
  truth <- truths$truth[selected][match(labels, truths$estimand[selected])]
  if (any(is.na(truth))) stop(sprintf("truth join failed for replicate %s", replicate))
  data.frame(
    implementation = "lmtp projected regimen fits",
    scenario = scenario,
    replicate = replicate,
    n = nrow(frame),
    estimand = labels,
    truth = truth,
    estimate = beta,
    inference_estimate = beta,
    std_error = standard_error,
    ci_lower = low,
    ci_upper = high,
    inference_scale = "identity",
    covered = as.integer(low <= truth & truth <= high),
    initial_estimate = initial_beta,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(results, expected, paths$output, versions = study_version("lmtp"))
