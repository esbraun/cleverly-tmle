suppressPackageStartupMessages(library(survtmle))
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments(
  "usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv"
)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
z <- qnorm(0.975)

# survival-node-bounds (X33), the comparator of P1 and P2.  CRAN survtmle 1.1.1 is installed by
# tests/canonical/survtmle/Dockerfile.  The development branch is not used: its bounded-hazard
# path drops -l_j from s_j and indexes rows wrongly.
#
# P1 is `method = "mean"` (scenario `mean_model`) and P2 is `method = "hazard"` (scenario
# `hazard_model`), one `t0` per call.  The valid bounds are transcribed from
# tests/studies/survival_node_bounds_law.py at the factor 1.1 (`mean_bounds`, `hazard_bounds`);
# tests/unit/test_survival_node_bounds_design.py reads these vectors back.  The lower bounds
# are zero at every node.
bound_factor <- 1.1
mean_upper <- c(
  0.21933370796098725, 0.1868519456870831, 0.14963001843473187,
  0.10681660878643615, 0.05736891938625957
)
hazard_upper <- c(
  0.0391283081998998, 0.04308229507644081, 0.047417980435194754,
  0.05216846049532346, 0.057368919386259515
)
# survtmle's default tol = 1 / sqrt(n) stops before the first fluctuation on these laws, so the
# runner passes tol = 1 / n and maxIter = 100, as point_survival_hazard_runner.R does.
max_iter <- 100
gtol <- 1e-3
reported <- c(3, 5)
arms <- c(arm0 = 0, arm1 = 1)
reference <- "arm0"
methods <- list(mean_model = "mean", hazard_model = "hazard")
uppers <- list(mean_model = mean_upper, hazard_model = hazard_upper)
# The conditional-mean model of P1 is per arm on W1 and W2; the hazard model of P2 adds a
# linear visit term (X22-plan.md II-3.4).  Censoring is one pooled model on all visits.
ftime_formulas <- list(mean_model = "trt*(W1 + W2)", hazard_model = "trt*(W1 + W2) + t")
ctime_formula <- "trt + W1 + W2 + t"

bounds_for <- function(scenario, t0) {
  # survtmle reads the bounds of visit t from the row with t == visit, for cause 1.
  data.frame(t = seq_len(t0), l1 = 0, u1 = uppers[[scenario]][seq_len(t0)])
}

check_estimation_rows <- function(frame, t0) {
  # A visit with no estimation row has an unestimable coefficient.  survtmle builds its rows
  # from the observed times, so every visit up to t0 must be one.
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

level_name <- function(arm, horizon) sprintf("risk_regimen[%s @ t=%d]", arm, horizon)

contrast_name <- function(horizon) sprintf("ate_regimen[arm1 vs %s @ t=%d]", reference, horizon)

ic_column <- function(fitted, scenario, arm, frame, t0) {
  # method = "mean" writes the curve of F(t0 | z) "IC1star.Z<z>"; method = "hazard" writes
  # "D.j1.z<z>" (hazard_tmle.R).
  column <- if (scenario == "mean_model") sprintf("IC1star.Z%d", arm) else sprintf("D.j1.z%d", arm)
  if (!column %in% names(fitted$ic)) {
    stop(sprintf("no influence column %s; survtmle returned %s", column,
                 paste(names(fitted$ic), collapse = ", ")))
  }
  ic <- fitted$ic[[column]]
  # The column is the curve of the incidence: it rises with an observed event by t0 in arm z.
  observed <- as.numeric(frame$A == arm & frame$event == 1 & frame$time <= t0)
  if (!(stats::cov(ic, observed) > 0)) {
    stop(sprintf("influence column %s does not rise with an observed event; its sign is wrong",
                 column))
  }
  ic
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

survtmle_call <- function(arguments, label) {
  # The survtmle call runs inside `study_stage`, so an error survtmle raises costs one group,
  # recorded under `label`.  A warning stops the run: `study_fatal_error` is the class a stage
  # never records.  Only the glm message about fitted probabilities at 0 or 1 is muffled,
  # because the censoring and treatment logits sit near the boundary on a sparse draw.
  withCallingHandlers(
    study_stage(label, do.call(survtmle, arguments)),
    warning = function(condition) {
      if (grepl("fitted probabilities numerically 0 or 1", conditionMessage(condition), fixed = TRUE)) {
        invokeRestart("muffleWarning")
      }
      study_fatal_error(sprintf("unexpected survtmle warning: %s", conditionMessage(condition)))
    }
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
  for (t0 in reported) {
    check_estimation_rows(frame, t0)
    arguments <- list(
      ftime = frame$time, ftype = frame$event, trt = frame$A, adjustVars = adjust, t0 = t0,
      method = methods[[scenario]], bounds = bounds_for(scenario, t0), glm.trt = "W1 + W2",
      glm.ftime = ftime_formulas[[scenario]], glm.ctime = ctime_formula,
      maxIter = max_iter, gtol = gtol, tol = tol, verbose = FALSE
    )
    fitted <- survtmle_call(c(arguments, list(returnIC = TRUE)), sprintf("survtmle %s t0=%d", scenario, t0))
    # The initial plug-in: the hazard fit at tol = Inf never enters the targeting loop, and
    # the mean fit with Gcomp = TRUE returns the g-computation estimate.
    initial_arguments <- arguments
    if (scenario == "mean_model") {
      initial_arguments$Gcomp <- TRUE
    } else {
      initial_arguments$tol <- Inf
    }
    initial <- survtmle_call(initial_arguments, sprintf("survtmle %s t0=%d initial", scenario, t0))
    values <- list()
    for (arm in names(arms)) {
      key <- sprintf("%d 1", arms[[arm]])
      values[[arm]] <- list(
        estimate = fitted$est[key, 1], initial = initial$est[key, 1],
        ic = ic_column(fitted, scenario, arms[[arm]], frame, t0)
      )
      rows[[length(rows) + 1]] <- row_for(
        scenario, replicate, level_name(arm, t0), values[[arm]]$estimate,
        values[[arm]]$initial, values[[arm]]$ic, n
      )
    }
    rows[[length(rows) + 1]] <- row_for(
      scenario, replicate, contrast_name(t0),
      values$arm1$estimate - values$arm0$estimate,
      values$arm1$initial - values$arm0$initial,
      values$arm1$ic - values$arm0$ic, n
    )
  }
  do.call(rbind, rows)
}

fit_one <- function(frame) {
  # One replication's samples of both models, which the Python side writes contiguously.
  do.call(rbind, lapply(split(frame, frame$scenario), fit_scenario))
}

expected <- length(unique(truths$replicate))
# A replicate holds 2 * 2,000 lines, so one 64,000-line chunk holds 16 replicates.
results <- study_stream(paths$samples, fit_one, expected, chunk_lines = 64000L)
study_collect(results, expected, paths$output, versions = study_version("survtmle"))
