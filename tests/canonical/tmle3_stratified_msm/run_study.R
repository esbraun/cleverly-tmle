suppressPackageStartupMessages({
  library(sl3)
  library(tmle3)
})
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(paths$samples), stringsAsFactors = FALSE)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
scenario <- "stratified_msm_identity"
terms <- c("(intercept)", "a", "W")
z <- qnorm(0.975)

# tmle3 has no stratified MSM, so each stratum's coefficients are Param_MSM on the stratum
# subset, built as tests/canonical/tmle3_msm/run_study.R builds the marginal one.  The weight
# is uniform; the classed function lets Param_MSM's sentinel comparisons fall through to its
# documented custom-function branch without modifying the pinned package.
weight <- function(A, V) rep(1, length(A))
class(weight) <- c("cleverly_msm_weight", class(weight))
Ops.cleverly_msm_weight <- function(e1, e2) {
  if (.Generic == "==") return(FALSE)
  NextMethod()
}
learners <- list(A = sl3::Lrnr_glm$new(), Y = sl3::Lrnr_glm$new())

truth_for <- function(replicate, estimand) {
  selected <- truths$replicate == replicate & truths$scenario == scenario & truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows for %s", sum(selected), estimand))
  truths$truth[selected]
}

fit_stratum <- function(subset) {
  stage <- function(label, expression) {
    tryCatch(
      force(expression),
      error = function(condition) {
        stop(sprintf("%s: %s", label, conditionMessage(condition)), call. = FALSE)
      }
    )
  }
  # The one-column W node is simplified to a vector at this release, so a deterministic second
  # baseline column keeps it a named table, as the marginal runner does.
  data <- subset[c("W", "A", "Y")]
  data$W_squared <- data$W^2
  data <- data[c("W", "W_squared", "A", "Y")]
  nodes <- list(W = "W_squared", V = "W", A = "A", Y = "Y")
  spec <- tmle3_Spec_MSM$new(msm = "A + V", weight = weight, weight_ub = NULL)
  task <- stage("task", spec$make_tmle_task(data, nodes))
  initial <- stage("initial likelihood", spec$make_initial_likelihood(task, learners))
  updater <- tmle3_Update$new(cvtmle = FALSE, convergence_type = "sample_size")
  targeted <- Targeted_Likelihood$new(initial, updater)
  parameter <- stage("targeted parameter", spec$make_params(task, targeted))
  updater$tmle_params <- parameter
  stage("targeting", fit_tmle3(task, targeted, parameter, updater))
  values <- stage("targeted estimates", parameter$estimates(task))
  initial_parameter <- stage("initial parameter", suppressWarnings(Param_MSM$new(
    initial, "W", strata_name = "V", msm = "A + V", weight = weight, weight_ub = NULL,
    treatment_values = c(0, 1)
  )))
  initial_values <- stage("initial estimates", initial_parameter$estimates(task))
  # The arm-indicator coefficients to the (1, a, W) basis, as the marginal runner does.
  transform_beta <- function(beta) c(unname(beta[[1]]), unname(beta[[2]] - beta[[1]]), unname(beta[[3]]))
  transform_ic <- function(ic) {
    values <- as.matrix(ic)
    cbind(values[, 1], values[, 2] - values[, 1], values[, 3])
  }
  list(
    estimate = transform_beta(values$psi),
    initial = transform_beta(initial_values$psi),
    ic = transform_ic(values$IC)
  )
}

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  n_total <- nrow(frame)
  out <- list()
  for (stratum in c(0, 1, 2)) {
    subset <- frame[frame$V == stratum, , drop = FALSE]
    fit <- fit_stratum(subset)
    # A subset curve is on the subset's scale, so the standard error is sd / sqrt(n_s): the
    # number the stratified fit's embedded curve gives on the full sample.
    standard_error <- apply(fit$ic, 2, sd) / sqrt(nrow(subset))
    names <- sprintf("msm[%s][V=%d]", terms, stratum)
    truth <- vapply(names, function(name) truth_for(replicate, name), numeric(1))
    out[[length(out) + 1]] <- stratum_rows(fit, standard_error, names, truth, replicate, n_total)
  }
  result <- do.call(rbind, out)
  stopifnot(nrow(result) == 9L, !anyDuplicated(result$estimand))
  result
}

stratum_rows <- function(fit, standard_error, names, truth, replicate, n_total) {
  low <- fit$estimate - z * standard_error
  high <- fit$estimate + z * standard_error
  data.frame(
    implementation = "tmle3-msm-stratified",
    scenario = scenario,
    replicate = replicate,
    n = n_total,
    estimand = names,
    truth = unname(truth),
    estimate = fit$estimate,
    inference_estimate = fit$estimate,
    std_error = standard_error,
    ci_lower = low,
    ci_upper = high,
    inference_scale = "identity",
    covered = as.integer(low <= truth & truth <= high),
    initial_estimate = fit$initial,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

groups <- split(samples, samples$replicate)
expected <- length(groups)
rm(samples)
invisible(gc())
results <- parallel::mclapply(
  seq_along(groups), study_fitter(groups, fit_one),
  mc.cores = study_cores(groups), mc.preschedule = FALSE
)
study_collect(
  results, expected, paths$output,
  versions = c(study_version("tmle3"), study_version("sl3")), na = ""
)
