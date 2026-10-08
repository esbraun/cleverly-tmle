suppressPackageStartupMessages(library(survtmle))
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments(
  "usage: point_survival_runner.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv"
)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
z <- qnorm(0.975)

# The reported visits of each scenario and the label of each event code.  Transcribed from
# tests/studies/canonical_point_survival.py; a gate test reads these vectors back.
reported <- list(survival = c(1, 3, 5), competing = c(2, 4))
causes <- list(survival = c("1" = ""), competing = c("1" = "relapse", "2" = "death"))
arms <- c(arm0 = 0, arm1 = 1)
reference <- "arm0"

ctime_formula <- function(t0) {
  # One intercept and three slopes per visit 1..t0-1, and no term shared across visits, so the
  # likelihood factorises by visit and the pooled fit is the per-node fit.  survtmle predicts
  # this model at t - 1, so visit k's coefficients are node k + 1's censoring model.
  if (t0 <= 1) return("1")
  terms <- vapply(seq_len(t0 - 1), function(k) {
    sprintf("I(t == %d) + I(t == %d):trt + I(t == %d):W1 + I(t == %d):W2", k, k, k, k)
  }, character(1))
  paste("-1 +", paste(terms, collapse = " + "))
}

check_estimation_rows <- function(frame, t0) {
  # A visit with no estimation row has an unestimable coefficient.  survtmle builds its rows
  # from the observed times, so every visit below t0 must be one.
  observed <- sort(unique(frame$time[frame$time <= t0]))
  missing <- setdiff(seq_len(t0), observed)
  if (length(missing)) stop(sprintf("no estimation rows at visit(s) %s", paste(missing, collapse = ", ")))
}

truth_for <- function(scenario, replicate, estimand) {
  selected <- truths$scenario == scenario & truths$replicate == replicate &
    truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows for %s", sum(selected), estimand))
  truths$truth[selected]
}

level_name <- function(arm, horizon, cause) {
  if (cause == "") sprintf("risk_regimen[%s @ t=%d]", arm, horizon)
  else sprintf("cif_regimen[%s, %s @ t=%d]", arm, cause, horizon)
}

contrast_name <- function(horizon, cause) {
  if (cause == "") sprintf("ate_regimen[arm1 vs %s @ t=%d]", reference, horizon)
  else sprintf("ate_regimen[arm1 vs %s, %s @ t=%d]", reference, cause, horizon)
}

row_for <- function(scenario, replicate, name, estimate, initial, ic, n) {
  truth <- truth_for(scenario, replicate, name)
  standard_error <- sqrt(sum(ic^2)) / n
  low <- estimate - z * standard_error
  high <- estimate + z * standard_error
  data.frame(
    implementation = "survtmle",
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

fit_scenario <- function(frame) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  frame <- frame[order(frame$row), ]
  n <- nrow(frame)
  adjust <- frame[c("W1", "W2")]
  rows <- list()
  for (t0 in reported[[scenario]]) {
    check_estimation_rows(frame, t0)
    arguments <- list(
      ftime = frame$time, ftype = frame$event, trt = frame$A, adjustVars = adjust, t0 = t0,
      method = "mean", glm.trt = "W1 + W2", glm.ftime = "trt*(W1 + W2)",
      glm.ctime = ctime_formula(t0), verbose = FALSE
    )
    fitted <- suppressWarnings(do.call(survtmle, c(arguments, list(returnIC = TRUE))))
    initial <- suppressWarnings(do.call(survtmle, c(arguments, list(Gcomp = TRUE))))
    for (code in names(causes[[scenario]])) {
      cause <- causes[[scenario]][[code]]
      values <- list()
      for (arm in names(arms)) {
        key <- sprintf("%d %s", arms[[arm]], code)
        column <- sprintf("IC%sstar.Z%d", code, arms[[arm]])
        values[[arm]] <- list(
          estimate = fitted$est[key, 1], initial = initial$est[key, 1], ic = fitted$ic[[column]]
        )
        rows[[length(rows) + 1]] <- row_for(
          scenario, replicate, level_name(arm, t0, cause), values[[arm]]$estimate,
          values[[arm]]$initial, values[[arm]]$ic, n
        )
      }
      rows[[length(rows) + 1]] <- row_for(
        scenario, replicate, contrast_name(t0, cause),
        values$arm1$estimate - values$arm0$estimate,
        values$arm1$initial - values$arm0$initial,
        values$arm1$ic - values$arm0$ic, n
      )
    }
  }
  do.call(rbind, rows)
}

fit_one <- function(frame) {
  # One replication's samples of every scenario, which the Python side writes contiguously.
  do.call(rbind, lapply(split(frame, frame$scenario), fit_scenario))
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(results, expected, paths$output, versions = study_version("survtmle"))
