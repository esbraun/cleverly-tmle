suppressPackageStartupMessages({
  library(data.table)
  library(sl3)
  library(tmle3)
})
source("/fixture/study_harness.R")
options(digits = 17)
stopifnot(requireNamespace("glmnet", quietly = TRUE))

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
z <- qnorm(0.975)

# Transcribed from tests/studies/canonical_point_survival_curve.py; a gate test reads these
# lines back.
grid <- 1:5
arms <- c(arm0 = 0, arm1 = 1)
reference <- "arm0"
maxit <- 100
delta_epsilon <- 0.01

# tmle3's tmle3_Update_survival at ed72f8a, with two R-level defects repaired and nothing else
# changed.  (1) The original tests `if (fit_lambda == 1)` on the whole lambda vector, which is
# an error in R >= 4.2; its tryCatch then prints the error and returns a zero step.  The repair
# tests `length(fit_lambda) == 1`.  (2) The retry assigned `lambda_min_ratio`, a name glmnet
# never reads, so the retry refitted the same path.  The repair assigns `lambda.min.ratio`.
tmle3_Update_survival_repaired <- R6::R6Class(
  classname = "tmle3_Update_survival_repaired",
  inherit = tmle3::tmle3_Update_survival,
  public = list(
    fit_submodel = function(submodel_data) {
      if (self$fit_method != "l2") {
        return(super$fit_submodel(submodel_data))
      }
      epsilon_n <- tryCatch(
        {
          alpha <- 0
          norm_func <- self$norm_l2
          lambda.min.ratio <- 1e-2
          ind <- 1
          while (ind == 1) {
            submodel_fit <- glmnet::glmnet(
              x = submodel_data$H,
              y = submodel_data$observed,
              offset = qlogis(submodel_data$initial),
              family = "binomial",
              alpha = alpha,
              standardize = FALSE,
              intercept = FALSE,
              lambda.min.ratio = lambda.min.ratio,
              nlambda = 1e2
            )
            norms <- apply(submodel_fit$beta, 2, norm_func)
            ind <- max(which(norms <= self$delta_epsilon))
            if (ind > 1) break
            fit_lambda <- submodel_fit$lambda
            if (length(fit_lambda) == 1) {
              stop("only one lambda could be fit")
            }
            norm_ratio <- self$delta_epsilon / norms[2]
            lambda_guess <- fit_lambda[1] - norm_ratio * (fit_lambda[1] - fit_lambda[2])
            lambda.min.ratio <- 0.8 * lambda_guess / fit_lambda[1]
          }
          epsilon_n <- submodel_fit$beta[, ind]
        },
        error = function(e) {
          print(e)
          return(rep(0, ncol(submodel_data$H)))
        }
      )
      epsilon <- epsilon_n
      epsilon[is.na(epsilon)] <- 0
      epsilon
    }
  )
)

learners <- function() {
  # One unpenalized logistic regression per visit on the task's covariates, for the event and
  # for the censoring indicator; a pooled censoring fit would read the administrative
  # censoring at the last visit as a censoring hazard of every visit.
  per_visit <- sl3::Lrnr_stratified$new(learner = sl3::Lrnr_glm$new(), variable_stratify = "t")
  list(A = sl3::Lrnr_glm$new(), N = per_visit, A_c = per_visit)
}

truth_for <- function(replicate, estimand) {
  selected <- truths$replicate == replicate & truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows for %s", sum(selected), estimand))
  truths$truth[selected]
}

row_for <- function(replicate, name, estimate, initial, ic, n) {
  truth <- truth_for(replicate, name)
  standard_error <- sqrt(sum(ic^2)) / n
  low <- estimate - z * standard_error
  high <- estimate + z * standard_error
  data.frame(
    implementation = "tmle3",
    scenario = "survival",
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

fit_arm <- function(data, arm, replicate) {
  spec <- tmle3::tmle_survival(treatment_level = arm, control_level = 1 - arm, target_times = grid)
  nodes <- list(W = c("W1", "W2"), A = "A", T_tilde = "T_tilde", Delta = "Delta", id = "id")
  long <- spec$transform_data(data, nodes)
  task <- spec$make_tmle_task(long$long_data, long$long_node_list)
  observed_times <- sort(unique(task$time))
  if (!identical(as.numeric(observed_times), as.numeric(grid))) {
    stop(sprintf("the long task's visits are %s, not the declared grid",
                 paste(observed_times, collapse = ", ")))
  }
  initial <- spec$make_initial_likelihood(task, learners())
  updater <- tmle3_Update_survival_repaired$new(
    maxit = maxit, cvtmle = FALSE, convergence_type = "sample_size",
    delta_epsilon = delta_epsilon, fit_method = "l2", verbose = FALSE
  )
  targeted <- tmle3::Targeted_Likelihood$new(initial, updater)
  parameters <- spec$make_params(task, targeted)
  updater$tmle_params <- parameters
  printed <- utils::capture.output(fitted <- tmle3::fit_tmle3(task, targeted, parameters, updater))
  errors <- sum(grepl("Error", printed, fixed = TRUE))

  final <- parameters[[1]]$estimates(task)
  ids <- task$id
  first <- !duplicated(ids)
  order_ids <- order(ids[first])
  ic <- as.matrix(final$IC)[first, , drop = FALSE][order_ids, , drop = FALSE]
  n <- nrow(ic)
  scores <- colMeans(ic)
  epsilons <- updater$epsilons
  nonzero <- sum(vapply(epsilons, function(e) any(unlist(e) != 0), logical(1)))
  solved <- max(abs(scores)) <= 1 / n
  # The epsilon travel, against maxit * delta_epsilon, and the gap between the SE read off the
  # per-unit influence curve and the one tmle3's summary reports (plan 3.4, A2).
  travel <- sum(vapply(epsilons, function(e) sqrt(sum(unlist(e)^2)), numeric(1)))
  summary_se <- as.numeric(fitted$summary$se)
  ic_se <- sqrt(colSums(ic^2)) / n
  se_gap <- if (length(summary_se) == length(ic_se)) max(abs(summary_se - ic_se)) else NA_real_
  message(sprintf(
    paste0("tmle3-survival replicate=%d arm=%d steps=%d nonzero_steps=%d printed_errors=%d ",
           "max_abs_score=%.17g bound=%.17g solved=%s maxit_bound=%s travel=%.17g ",
           "max_se_gap=%.17g"),
    replicate, arm, updater$steps, nonzero, errors, max(abs(scores)), 1 / n, solved,
    updater$steps >= maxit, travel, se_gap
  ))
  if (nonzero == 0 && !solved) {
    stop(sprintf("arm %d took no nonzero step and left the score unsolved (%.3g > 1/n)",
                 arm, max(abs(scores))))
  }
  list(survival = final$psi, initial = fitted$initial_psi, ic = ic)
}

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  frame <- frame[order(frame$row), ]
  data <- data.table::data.table(
    id = seq_len(nrow(frame)), W1 = frame$W1, W2 = frame$W2, A = frame$A,
    T_tilde = frame$time, Delta = as.numeric(frame$event == 1)
  )
  fits <- lapply(arms, function(arm) fit_arm(data, arm, replicate))
  n <- nrow(data)
  rows <- list()
  for (k in seq_along(grid)) {
    t0 <- grid[[k]]
    for (arm in names(arms)) {
      # The risk is one minus the survival tmle3 reports, so its influence curve is negated.
      rows[[length(rows) + 1]] <- row_for(
        replicate, sprintf("risk_regimen[%s @ t=%d]", arm, t0),
        1 - fits[[arm]]$survival[[k]], 1 - fits[[arm]]$initial[[k]], -fits[[arm]]$ic[, k], n
      )
    }
    rows[[length(rows) + 1]] <- row_for(
      replicate, sprintf("ate_regimen[arm1 vs %s @ t=%d]", reference, t0),
      fits$arm0$survival[[k]] - fits$arm1$survival[[k]],
      fits$arm0$initial[[k]] - fits$arm1$initial[[k]],
      fits$arm0$ic[, k] - fits$arm1$ic[, k], n
    )
  }
  do.call(rbind, rows)
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(
  results, expected, paths$output,
  versions = c(study_version("tmle3"), study_version("sl3"), study_version("glmnet"))
)
