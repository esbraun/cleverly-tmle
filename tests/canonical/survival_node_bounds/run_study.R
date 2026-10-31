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
# `hazard_model`), one `t0` per call.  The calls were checked against the survtmle 1.1.1 source
# (tmle_mean.R, tmle_hazard.R, makeDataList.R, censoring_estimate.R, hazards_estimate.R).  The valid bounds are transcribed from
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
# linear visit term (X22-plan.md II-3.4).  Censoring is one pooled model on visits 1 to 4.
ftime_formulas <- list(mean_model = "trt*(W1 + W2)", hazard_model = "trt*(W1 + W2) + t")
ctime_formula <- "trt + W1 + W2 + t"
# The law's visits.  A censoring at the last visit is the study's end, not a dropout.
visits <- 5

bounds_for <- function(scenario) {
  # survtmle reads the bounds of visit t from the row with t == visit, for cause 1, and fills
  # 0 and 1 at a visit the frame does not name.  The hazard fit pools every visit of the long
  # data, also those after t0, so every visit carries its bound: with bounds up to t0 only, the
  # later rows would enter one pooled logit on another scale.  The iterated mean reads visits up
  # to t0 only, so the extra rows change nothing there.
  data.frame(t = seq_len(visits), l1 = 0, u1 = uppers[[scenario]])
}

check_estimation_rows <- function(frame) {
  # survtmle numbers its long rows by the rank of the observed time among the distinct times
  # (makeDataList.R), so the visit index is right only when every visit is an observed time.
  observed <- sort(unique(frame$time))
  if (!identical(as.numeric(observed), as.numeric(seq_len(visits)))) {
    stop(sprintf("observed times %s are not the visits 1 to %d",
                 paste(observed, collapse = ", "), visits))
  }
}

censoring_rows <- function(frame) {
  # The pooled censoring regression of the subject: one row per unit and visit s = 1..4 at
  # which the unit is still observed after its event check, C = 1 for a dropout after visit s.
  # This is the subject's node s + 1 retention, and the model is linear in the visit, so the
  # visit offset moves the intercept only.
  rows <- lapply(seq_len(visits - 1), function(s) {
    keep <- frame$time >= s & !(frame$time == s & frame$event == 1)
    data.frame(
      C = as.numeric(frame$time[keep] == s & frame$event[keep] == 0),
      trt = frame$A[keep], W1 = frame$W1[keep], W2 = frame$W2[keep], t = s
    )
  })
  do.call(rbind, rows)
}

fit_censoring <- function(frame, label) {
  # survtmle fits glm.ctime on its long rows with an exclusion rule that depends on t0.  At
  # t0 = 3 it keeps the visit-5 rows, where every unit still followed reads C = 1 (the study's
  # end), and the visit-4 event rows with C = 0; one logit linear in t then misses the dropout
  # hazard at the visits t0 reads.  survtmle accepts a fitted glm as glm.ctime and predicts it
  # at t - 1 with G_dC = 1 at t = 1 (censoring_estimate.R).  So the runner fits the declared
  # formula once per sample on the subject's row set and passes the fit to both t0 calls.
  # The glm runs inside `study_stage`, so an error in it costs one group; the row builder
  # stays outside, so its checks stay fatal.
  rows <- censoring_rows(frame)
  formula <- stats::as.formula(paste("C ~", ctime_formula))
  study_stage(label, stats::glm(formula, data = rows, family = stats::binomial()))
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

#: survtmle's own convergence warnings (hazards_fluctuate.R and tmle_hazard.R).  The fit still
#: returns its estimate, so each one is counted and logged, and the run continues.
convergence_texts <- c("Fluctuation convergence failure", "TMLE fluctuations did not converge")

survtmle_call <- function(arguments, label) {
  # The initial hazard call passes tol = Inf on purpose, so checkInputs.R's advice to lower a
  # tol above 1/sqrt(n) is expected there and only there.
  untargeted <- is.infinite(arguments$tol)
  # The survtmle call runs inside `study_stage`, so an error survtmle raises costs one group,
  # recorded under `label`.  Any other warning stops the run: `study_fatal_error` is the class
  # a stage never records.  The glm message about fitted probabilities at 0 or 1 is muffled,
  # because the logits sit near the boundary on a sparse draw.
  counted <- 0L
  fitted <- withCallingHandlers(
    study_stage(label, do.call(survtmle, arguments)),
    warning = function(condition) {
      text <- conditionMessage(condition)
      if (grepl("fitted probabilities numerically 0 or 1", text, fixed = TRUE)) {
        invokeRestart("muffleWarning")
      }
      # checkInputs.R parses `paste0("C ~", glm.ctime)` to check a formula string and discards
      # the result.  For the fitted glm the runner passes, that paste is a vector, so R warns
      # that formula() of a vector is deprecated.  Nothing reads the parsed value.
      if (grepl("Using formula(x) is deprecated when x is a character vector", text, fixed = TRUE)) {
        invokeRestart("muffleWarning")
      }
      if (untargeted && grepl("tol is larger than 1/sqrt(n)", text, fixed = TRUE)) {
        invokeRestart("muffleWarning")
      }
      if (any(vapply(convergence_texts, grepl, logical(1), x = text, fixed = TRUE))) {
        counted <<- counted + 1L
        invokeRestart("muffleWarning")
      }
      study_fatal_error(sprintf("unexpected survtmle warning: %s", text))
    }
  )
  list(fit = fitted, warnings = counted)
}

fit_scenario <- function(frame) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  frame <- frame[order(frame$row), ]
  n <- nrow(frame)
  adjust <- frame[c("W1", "W2")]
  rows <- list()
  tol <- 1 / n
  check_estimation_rows(frame)
  censoring <- fit_censoring(frame, sprintf("censoring glm %s", scenario))
  for (t0 in reported) {
    arguments <- list(
      ftime = frame$time, ftype = frame$event, trt = frame$A, adjustVars = adjust, t0 = t0,
      method = methods[[scenario]], bounds = bounds_for(scenario), glm.trt = "W1 + W2",
      glm.ftime = ftime_formulas[[scenario]], glm.ctime = censoring,
      maxIter = max_iter, gtol = gtol, tol = tol, verbose = FALSE
    )
    targeted <- survtmle_call(c(arguments, list(returnIC = TRUE)), sprintf("survtmle %s t0=%d", scenario, t0))
    fitted <- targeted$fit
    # The initial plug-in.  The hazard fit at tol = Inf never enters the targeting loop, so its
    # est is the initial fit's plug-in.  The mean fit has none: survtmle 1.1.1's `Gcomp = TRUE`
    # with `bounds` stops with "object 'Qj.t' not found" (iterated_mean_fluctuate.R defines
    # Qj.t only in the targeted branch), so the mean rows carry NA, as the npcausal runners do.
    # The column feeds only the subject's targeting displacement.
    initial <- NULL
    if (scenario == "hazard_model") {
      initial_arguments <- arguments
      initial_arguments$tol <- Inf
      initial <- survtmle_call(initial_arguments, sprintf("survtmle %s t0=%d initial", scenario, t0))$fit
    }
    # The mean method fluctuates once per visit and never reads tol; the hazard method loops
    # until every |meanIC| <= tol or maxIter + 1 rounds.  The run log reads both.
    mean_ic <- as.numeric(unlist(fitted$meanIC))
    message(sprintf(
      "survtmle-%s replicate=%d t0=%d convergence_warnings=%d max_abs_meanIC=%.17g tol=%.17g converged=%s",
      scenario, replicate, t0, targeted$warnings, max(abs(mean_ic)), tol,
      scenario == "mean_model" || all(abs(mean_ic) <= tol)
    ))
    values <- list()
    for (arm in names(arms)) {
      key <- sprintf("%d 1", arms[[arm]])
      values[[arm]] <- list(
        estimate = fitted$est[key, 1],
        initial = if (is.null(initial)) NA_real_ else initial$est[key, 1],
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
