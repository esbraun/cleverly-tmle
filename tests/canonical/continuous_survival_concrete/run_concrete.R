# The concrete 1.0.5 side of the continuous-time-survival comparator phase (roadmap X6 part f).
#
# Each group is one cell's draw of one replicate, written by
# tests/canonical/continuous_survival_concrete/run_phase.py, so a concrete failure in one cell
# costs that cell's draw and not the replicate's other cells.  Each cell is one formatArguments,
# doConcrete and getOutput call on the cell's models (tests/studies/continuous_survival_concrete.py,
# CONCRETE_CELLS), with concrete's defaults for the update: 500 steps, a first step of 0.1, the
# positivity bound 5 / (sqrt(n) log(n)) and its stopping rule.  The published std_error is
# getOutput's; ic_std_error reads the influence curves, which for a risk difference is the
# difference of the two arms' curves where getOutput adds their variances.
#
# doTmleUpdate.R:106 calls browser() when the new norm of the mean influence curve is NA.  That
# branch is unreachable: the norm is NA only when an influence curve is, and getIC (getEIC.R)
# stops with "IC overflow" on any NA or NaN curve before the norm is formed, which the stage
# records as a failure of the draw.  The container also runs without a terminal on stdin.
suppressPackageStartupMessages({
  library(concrete)
  library(data.table)
  library(survival)
})
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments(
  "usage: run_concrete.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv"
)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
z <- qnorm(0.975)

# Transcribed from tests/studies/continuous_survival_concrete.py; a fast-tier test reads these
# back.
times <- c(0.46, 1.12, 2.32)
arms <- c(treated = "A=1", control = "A=0")
causes <- c("1" = "cause1", "2" = "cause2")
formulas <- c(
  treatment = "Z",
  main = "Z + X1 + X2 + X3 + X4",
  interaction = "Z * (X1 + X2 + X3 + X4)"
)
cells <- list(
  p1_parity = list(
    hazards = c("interaction"), censoring = c("main"), treatment = c("SL.glm"), folds = 2L
  ),
  p2_correct = list(
    hazards = c("treatment", "main", "interaction"),
    censoring = c("treatment", "main", "interaction"),
    treatment = c("SL.glm", "SL.glmnet"), folds = 10L
  ),
  p3_hazard_wrong = list(
    hazards = c("treatment"), censoring = c("main"), treatment = c("SL.glm"), folds = 2L
  )
)

hazard_models <- function(code, members) {
  out <- lapply(members, function(member) {
    as.formula(sprintf("Surv(time, event == %s) ~ %s", code, formulas[[member]]))
  })
  names(out) <- members
  out
}

model_for <- function(spec) {
  list(
    Z = spec$treatment,
    "0" = hazard_models(0, spec$censoring),
    "1" = hazard_models(1, spec$hazards),
    "2" = hazard_models(2, spec$hazards)
  )
}

level_name <- function(arm, cause, time) sprintf("cif_regimen[%s, %s @ t=%g]", arm, cause, time)
difference_name <- function(cause, time) {
  sprintf("ate_regimen[treated vs control, %s @ t=%g]", cause, time)
}

truth_for <- function(scenario, replicate, estimand) {
  selected <- truths$scenario == scenario & truths$replicate == replicate &
    truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows for %s", sum(selected), estimand))
  truths$truth[selected]
}

pick <- function(output, estimator, intervention, code, time) {
  selected <- output$Estimator == estimator & output$Intervention == intervention &
    output$Event == as.numeric(code) & output$Time == time
  if (sum(selected) != 1) {
    stop(sprintf(
      "concrete's output has %d %s rows for %s, event %s, time %g",
      sum(selected), estimator, intervention, code, time
    ))
  }
  output[selected, ]
}

curve_for <- function(estimate, intervention, code, time, n) {
  ic <- as.data.frame(estimate[[intervention]][["IC"]])
  ic <- ic[ic$Event == as.numeric(code) & ic$Time == time, ]
  ic <- ic[order(ic$ID), ]
  if (nrow(ic) != n) stop(sprintf("concrete's influence curve has %d of %d rows", nrow(ic), n))
  ic$IC
}

row_for <- function(scenario, replicate, n, name, point, initial, se, ic, steps, converged,
                    seconds) {
  truth <- truth_for(scenario, replicate, name)
  low <- point - z * se
  high <- point + z * se
  data.frame(
    implementation = "concrete",
    scenario = scenario,
    replicate = replicate,
    n = n,
    estimand = name,
    truth = truth,
    estimate = point,
    inference_estimate = point,
    std_error = se,
    ci_lower = low,
    ci_upper = high,
    inference_scale = "identity",
    covered = as.integer(low <= truth && truth <= high),
    initial_estimate = initial,
    ic_std_error = sd(ic) / sqrt(n),
    steps = steps,
    converged = converged,
    seconds = seconds,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

fit_cell <- function(frame) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  if (!(scenario %in% names(cells))) stop(sprintf("no concrete cell %s", scenario))
  frame <- frame[order(frame$row), ]
  n <- nrow(frame)
  data <- data.table::as.data.table(frame[c("time", "event", "Z", "X1", "X2", "X3", "X4")])
  spec <- cells[[scenario]]
  model <- model_for(spec)
  bound <- 5 / sqrt(n) / log(n)
  folds <- list(V = spec$folds)
  start <- proc.time()[["elapsed"]]
  arguments <- study_stage("concrete", suppressWarnings(suppressMessages(formatArguments(
    DataTable = data, EventTime = "time", EventType = "event", Treatment = "Z",
    TargetTime = times, TargetEvent = c(1, 2), Intervention = makeITT(), CVArg = folds,
    Model = model, MaxUpdateIter = 500, OneStepEps = 0.1, MinNuisance = bound,
    Verbose = FALSE, GComp = TRUE, ReturnModels = FALSE, RenameCovs = FALSE
  ))))
  estimate <- study_stage("concrete", suppressWarnings(suppressMessages(doConcrete(arguments))))
  output <- study_stage("concrete", suppressWarnings(suppressMessages(getOutput(
    estimate, Estimand = c("Risk", "RD"), Intervention = 1:2, GComp = TRUE, Simultaneous = FALSE
  ))))
  seconds <- proc.time()[["elapsed"]] - start
  output <- as.data.frame(output)
  status <- attr(estimate, "TmleConverged")
  steps <- as.numeric(status$step)
  converged <- as.numeric(isTRUE(status$converged))
  rows <- list()
  for (code in names(causes)) {
    for (time in times) {
      curves <- list()
      for (arm in names(arms)) {
        tmle <- pick(output, "tmle", arms[[arm]], code, time)
        gcomp <- pick(output, "gcomp", arms[[arm]], code, time)
        curves[[arm]] <- curve_for(estimate, arms[[arm]], code, time, n)
        rows[[length(rows) + 1]] <- row_for(
          scenario, replicate, n, level_name(arm, causes[[code]], time), tmle[["Pt Est"]],
          gcomp[["Pt Est"]], tmle[["se"]], curves[[arm]], steps, converged, seconds
        )
      }
      contrast <- sprintf("[%s] - [%s]", arms[["treated"]], arms[["control"]])
      tmle <- pick(output, "tmle", contrast, code, time)
      gcomp <- pick(output, "gcomp", contrast, code, time)
      rows[[length(rows) + 1]] <- row_for(
        scenario, replicate, n, difference_name(causes[[code]], time), tmle[["Pt Est"]],
        gcomp[["Pt Est"]], tmle[["se"]], curves$treated - curves$control, steps, converged,
        seconds
      )
    }
  }
  do.call(rbind, rows)
}

versions <- c(
  study_version("concrete"), study_version("survival"), study_version("SuperLearner"),
  study_version("glmnet"), study_version("data.table")
)
writeLines(c(versions, R.version.string), file.path(dirname(paths$output), "concrete-versions.txt"))
samples <- read.csv(paths$samples, stringsAsFactors = FALSE)
groups <- split(samples, interaction(samples$scenario, samples$replicate, drop = TRUE))
cores <- study_cores(groups)
# Forced here, in the parent: with `mc.preschedule = TRUE`, `mclapply` would force it in each
# child, and `study_collect` would then find no registered groups to retry or assemble.
fit_group <- study_fitter(groups, fit_cell)
results <- parallel::mclapply(seq_along(groups), fit_group, mc.cores = cores, mc.preschedule = TRUE)
study_collect(
  results,
  expected = length(groups),
  output = paths$output,
  versions = versions,
  key = c("scenario", "replicate")
)
