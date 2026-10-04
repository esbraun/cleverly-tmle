suppressPackageStartupMessages(library(ltmle))
source("/fixture/study_harness.R")
options(digits = 17)

# R `ltmle` 1.3-0 with `id=` on one treatment node, for `clustered-few-cluster-tmle`.
#
# Both arms are fitted in sample with the exact propensity as a numeric gform and the
# main-terms outcome formula `Q.kplus1 ~ W1 + W2`, stratified on the arm, which is the
# regression `cleverly`'s LTMLE fits on the followers of each regimen. `variance.method = "ic"`
# and `id` give ltmle's household influence curve: the cluster sum of the curve times J / n,
# with variance over J (`HouseholdIC`, R/ltmle.R lines 1025 to 1032). `fit$IC$tmle` is that
# household curve, and the ATE row reads the difference of the two arms' household curves. ltmle's interval uses a Student t
# reference with J - 1 degrees of freedom below 100 clusters (`GetCI`, lines 1608 to 1616);
# `cleverly` uses J - 2, a declared difference.

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(paths$samples), stringsAsFactors = FALSE, check.names = FALSE)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)

scenario <- "few_unequal_informative_j20"

household_standard_error <- function(household_ic) {
  # With `id=`, `fit$IC$tmle` is already the household curve: one entry per cluster, the
  # cluster sum of the row curve times J / n. Its standard error is sd / sqrt(J).
  sd(household_ic) / sqrt(length(household_ic))
}

t_interval <- function(estimate, standard_error, clusters) {
  critical <- if (clusters < 100) qt(0.975, clusters - 1) else qnorm(0.975)
  c(estimate - critical * standard_error, estimate + critical * standard_error)
}

fit_arm <- function(frame, arm) {
  propensity <- plogis(0.3 * frame$W1 + 0.6 * frame$W2)
  fit <- withCallingHandlers(
    ltmle(
      data = frame[c("W1", "W2", "A", "Y")],
      Anodes = "A",
      Ynodes = "Y",
      Qform = c(Y = "Q.kplus1 ~ W1 + W2"),
      gform = matrix(propensity, ncol = 1),
      abar = arm,
      gbounds = c(1e-8, 1),
      SL.library = "glm",
      stratify = TRUE,
      variance.method = "ic",
      id = frame$cluster,
      estimate.time = FALSE
    ),
    warning = function(condition) {
      stop(sprintf("unexpected ltmle warning: %s", conditionMessage(condition)))
    }
  )
  native <- summary(fit)$treatment
  ic <- as.numeric(fit$IC$tmle)
  if (length(ic) != length(unique(frame$cluster))) {
    stop(sprintf("ltmle returned %d IC entries for %d clusters", length(ic), length(unique(frame$cluster))))
  }
  standard_error <- household_standard_error(ic)
  if (abs(standard_error - native$std.dev) > 1e-10 * max(1, native$std.dev)) {
    stop(sprintf("household SE %.17g differs from ltmle's %.17g", standard_error, native$std.dev))
  }
  coefficients <- fit$fit$Q[[1]][, "Estimate"]
  design <- model.matrix(~ W1 + W2, data = frame)
  initial <- mean(plogis(design[, names(coefficients), drop = FALSE] %*% coefficients))
  list(
    estimate = unname(fit$estimates[["tmle"]]),
    ic = ic,
    std_error = standard_error,
    ci = as.numeric(native$CI),
    initial = initial
  )
}

truth_for <- function(replicate, estimand) {
  selected <- truths$replicate == replicate & truths$scenario == scenario & truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows", sum(selected)))
  truths$truth[selected]
}

row_for <- function(replicate, name, estimate, standard_error, ci, initial, n) {
  truth <- truth_for(replicate, name)
  data.frame(
    implementation = "ltmle",
    scenario = scenario,
    replicate = replicate,
    n = n,
    estimand = name,
    truth = truth,
    estimate = estimate,
    inference_estimate = estimate,
    std_error = standard_error,
    ci_lower = ci[[1]],
    ci_upper = ci[[2]],
    inference_scale = "identity",
    covered = as.integer(ci[[1]] <= truth && truth <= ci[[2]]),
    initial_estimate = initial,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  # The published n is the record's nominal size, as the Python rows publish it.
  n <- frame$nominal_n[[1]]
  clusters <- length(unique(frame$cluster))
  always <- fit_arm(frame, 1)
  never <- fit_arm(frame, 0)
  difference <- always$estimate - never$estimate
  difference_se <- household_standard_error(always$ic - never$ic)
  rbind(
    row_for(replicate, "ey_regimen[never]", never$estimate, never$std_error, never$ci, never$initial, n),
    row_for(replicate, "ey_regimen[always]", always$estimate, always$std_error, always$ci, always$initial, n),
    row_for(
      replicate,
      "ate_regimen[always vs never]",
      difference,
      difference_se,
      t_interval(difference, difference_se, clusters),
      always$initial - never$initial,
      n
    )
  )
}

groups <- split(samples, samples$replicate)
expected <- length(groups)
rm(samples)
invisible(gc())

fit_group <- study_fitter(groups, fit_one)
cores <- study_cores(groups)
results <- parallel::mclapply(seq_along(groups), fit_group, mc.cores = cores, mc.preschedule = FALSE)
study_collect(results, expected, paths$output, versions = study_version("ltmle"))
