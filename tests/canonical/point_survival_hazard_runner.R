suppressPackageStartupMessages(library(survtmle))
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments(
  "usage: point_survival_hazard_runner.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv"
)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
z <- qnorm(0.975)

# The reported visits of each scenario and the label of each event code.  Transcribed from
# tests/studies/canonical_point_survival.py; a gate test reads these vectors back.
# survtmle's exported default tol = 1 / sqrt(n) stops before the first fluctuation on these
# laws, so the runner passes tol = 1 / n and maxIter = 100.  The gate test reads these lines
# back too.
max_iter <- 100
gtol <- 1e-3
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

ftime_formula <- function(t0) {
  # One intercept and three slopes per visit 1..t0 and per cause, with no term shared across
  # visits, so the pooled hazard fit is the per-node fit on the design [W1, W2, A].
  terms <- vapply(seq_len(t0), function(k) {
    sprintf("I(t == %d) + I(t == %d):trt + I(t == %d):W1 + I(t == %d):W2", k, k, k, k)
  }, character(1))
  paste("-1 +", paste(terms, collapse = " + "))
}

counted_call <- function(arguments, label) {
  # survtmle warns "Fluctuation convergence failure" when one fluctuation glm fails; every
  # warning is counted here rather than suppressed, then muffled so the log stays readable.
  # Every other warning's text is tallied for the run log.  This handler raises nothing, so no
  # warning becomes a record (test_harness_failures.py lists it in COUNTING_HANDLERS).
  # RM39a2: the survtmle call runs inside `study_stage`, so an error survtmle raises costs one
  # group, recorded under `label`.
  failures <- 0L
  texts <- character(0)
  fitted <- withCallingHandlers(
    study_stage(label, do.call(survtmle, arguments)),
    warning = function(w) {
      if (grepl("Fluctuation convergence failure", conditionMessage(w), fixed = TRUE)) {
        failures <<- failures + 1L
      } else {
        texts <<- c(texts, conditionMessage(w))
      }
      invokeRestart("muffleWarning")
    }
  )
  tally <- table(texts)
  list(
    fit = fitted, failures = failures, others = length(texts),
    other_texts = paste(sprintf("%s x%d", names(tally), as.integer(tally)), collapse = "; ")
  )
}

ic_column <- function(fitted, code, arm, frame, t0) {
  # method = "hazard" names the influence curve of F_j(t0 | z) "D.j<j>.z<z>"
  # (hazard_tmle.R), where method = "mean" writes "IC<j>star.Z<z>".
  column <- sprintf("D.j%s.z%d", code, arm)
  if (!column %in% names(fitted$ic)) {
    stop(sprintf("no influence column %s; survtmle returned %s", column,
                 paste(names(fitted$ic), collapse = ", ")))
  }
  ic <- fitted$ic[[column]]
  # The column is the curve of the incidence, not of the survival: it rises with an observed
  # cause-j event by t0 in arm z.  A negated, cause-swapped or permuted curve fails this check;
  # an arm swap does not, and the paired SE rows catch it.
  observed <- as.numeric(frame$A == arm & frame$event == as.numeric(code) & frame$time <= t0)
  if (!(stats::cov(ic, observed) > 0)) {
    stop(sprintf("influence column %s does not rise with an observed event; its sign is wrong",
                 column))
  }
  ic
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
  tol <- 1 / n
  for (t0 in reported[[scenario]]) {
    check_estimation_rows(frame, t0)
    arguments <- list(
      ftime = frame$time, ftype = frame$event, trt = frame$A, adjustVars = adjust, t0 = t0,
      method = "hazard", glm.trt = "W1 + W2", glm.ftime = ftime_formula(t0),
      glm.ctime = ctime_formula(t0), maxIter = max_iter, gtol = gtol, verbose = FALSE
    )
    targeted <- counted_call(
      c(arguments, list(tol = tol, returnIC = TRUE)), sprintf("survtmle t0=%d targeted", t0)
    )
    # At tol = Inf the targeting loop never runs, so est is the initial fit's plug-in.
    untargeted <- counted_call(c(arguments, list(tol = Inf)), sprintf("survtmle t0=%d tol=Inf", t0))
    fitted <- targeted$fit
    initial <- untargeted$fit
    mean_ic <- as.numeric(unlist(fitted$meanIC))
    message(sprintf(
      paste0("survtmle-hazard scenario=%s replicate=%d t0=%d failures=%d other_warnings=%d ",
             "initial_failures=%d max_abs_meanIC=%.17g tol=%.17g converged=%s ",
             "other_texts=[%s]"),
      scenario, replicate, t0, targeted$failures, targeted$others, untargeted$failures,
      max(abs(mean_ic)), tol, all(abs(mean_ic) <= tol), targeted$other_texts
    ))
    for (code in names(causes[[scenario]])) {
      cause <- causes[[scenario]][[code]]
      values <- list()
      for (arm in names(arms)) {
        key <- sprintf("%d %s", arms[[arm]], code)
        values[[arm]] <- list(
          estimate = fitted$est[key, 1], initial = initial$est[key, 1],
          ic = ic_column(fitted, code, arms[[arm]], frame, t0)
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
