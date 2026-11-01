# Comparator runner of the long-horizon ordinary LTMLE study (X59, S1): R ltmle 1.3-0.
#
# The fits are ltmle_horizon_adapter.R's.  The plans of a scenario are read from its abar
# columns and the estimands from the truth table, so nothing here restates a plan.  Each row
# carries one extra column after the declared fourteen, gcomp_estimate: ltmle's untargeted
# value (gcomp = TRUE) of the same estimand, which gate G-C reads (tests/canonical/regenerate.py,
# parity_inputs).

suppressPackageStartupMessages(library(ltmle))
source("/fixture/study_harness.R")
source("/fixture/ltmle_horizon_adapter.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
z <- qnorm(0.975)

typed <- function(frame) {
  # Reads the text columns back to their types and drops a column that is empty throughout.
  frame[] <- lapply(frame, function(column) {
    if (is.character(column)) type.convert(column, as.is = TRUE, na.strings = c("", "NA")) else column
  })
  frame[, colSums(!is.na(frame)) > 0, drop = FALSE]
}

plans_of <- function(frame) {
  sub("^abar__(.*)__1$", "\\1", grep("^abar__.*__1$", names(frame), value = TRUE))
}

parse_estimand <- function(name) {
  # ey_regimen[P], ate_regimen[P vs Q] or msm_regimen[term].
  inner <- sub("^[a-z_]+\\[(.*)\\]$", "\\1", name)
  if (startsWith(name, "ate_regimen[")) {
    sides <- strsplit(inner, " vs ", fixed = TRUE)[[1]]
    return(list(left = sides[[1]], right = sides[[2]]))
  }
  list(left = inner, right = NULL)
}

row_for <- function(scenario, replicate, name, truth, estimate, initial, gcomp, ic, n) {
  standard_error <- sd(ic) / sqrt(n)
  low <- estimate - z * standard_error
  high <- estimate + z * standard_error
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
    ci_lower = low,
    ci_upper = high,
    inference_scale = "identity",
    covered = as.integer(low <= truth && truth <= high),
    initial_estimate = initial,
    gcomp_estimate = gcomp,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

fit_scenario <- function(frame, scenario) {
  replicate <- frame$replicate[[1]]
  frame <- typed(frame[order(frame$row), ])
  wanted <- truths[truths$scenario == scenario & truths$replicate == replicate, ]
  if (!nrow(wanted)) stop(sprintf("no truth rows for %s replication %s", scenario, replicate))
  plans <- plans_of(frame)
  if (!length(plans)) stop(sprintf("%s has no deterministic plan to fit", scenario))
  fits <- setNames(lapply(plans, function(plan) {
    fit <- horizon_ltmle_plan(frame, plan)
    # Only the plan's estimates and curve outlive its two ltmle fits; a collection here frees
    # those fits before the next plan allocates its own.
    invisible(gc())
    fit
  }), plans)
  if (isTRUE(as.logical(unique(frame$design__msm)))) {
    durations <- vapply(plans, function(plan) unique(frame[[sprintf("msmdesign__%s", plan)]]), numeric(1))
    fits <- c(fits, horizon_msm(fits, durations))
  }
  rows <- lapply(seq_len(nrow(wanted)), function(index) {
    name <- wanted$estimand[[index]]
    parsed <- parse_estimand(name)
    key <- if (startsWith(name, "msm_regimen[")) name else parsed$left
    left <- fits[[key]]
    if (is.null(left)) stop(sprintf("%s: no fit answers %s", scenario, name))
    if (is.null(parsed$right) || startsWith(name, "msm_regimen[")) {
      return(row_for(
        scenario, replicate, name, wanted$truth[[index]],
        left$estimate, left$initial, left$gcomp, left$ic, nrow(frame)
      ))
    }
    right <- fits[[parsed$right]]
    if (is.null(right)) stop(sprintf("%s: no fit answers %s", scenario, name))
    row_for(
      scenario, replicate, name, wanted$truth[[index]],
      left$estimate - right$estimate, left$initial - right$initial, left$gcomp - right$gcomp,
      left$ic - right$ic, nrow(frame)
    )
  })
  do.call(rbind, rows)
}

fit_one <- function(frame) {
  # A reference-only run holds one scenario per archive (regenerate.py), which is fitted
  # without a split copy of the replicate; a frame of several scenarios is split.
  scenarios <- unique(frame$scenario)
  if (length(scenarios) == 1L) return(fit_scenario(frame, scenarios[[1]]))
  pieces <- split(frame, frame$scenario)
  do.call(rbind, lapply(names(pieces), function(scenario) fit_scenario(pieces[[scenario]], scenario)))
}

expected <- length(unique(truths$replicate))
results <- study_stream(
  paths$samples, horizon_timed(fit_one), expected, chunk_lines = horizon_chunk_lines()
)
study_collect(results, expected, paths$output, versions = study_version("ltmle"))
