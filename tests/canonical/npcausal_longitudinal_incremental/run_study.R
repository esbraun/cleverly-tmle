suppressPackageStartupMessages(library(npcausal))
source("/fixture/npcausal_ipsi_adapter.R")
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
z <- qnorm(0.975)
# Built once, before any group: the check that only the fold draw moved is fatal, not a
# group's failure.
ipsi_copy <- ipsi_with_folds()

# The plans of tests/studies/canonical_longitudinal_incremental.py, in the grid order of one
# call.  A gate test reads these back.
N_TIMES <- 3L
DELTAS <- c("odds x0.5" = 0.5, "natural course" = 1, "odds x2" = 2)
SCENARIOS <- c("ipsi_folds_5")

parse_estimand <- function(name) {
  inner <- sub("^[a-z_]+\\[(.*)\\]$", "\\1", name)
  if (startsWith(name, "ate_regimen[")) {
    sides <- strsplit(inner, " vs ", fixed = TRUE)[[1]]
    return(list(left = sides[[1]], right = sides[[2]]))
  }
  list(left = inner, right = NULL)
}

row_for <- function(scenario, replicate, name, truth, estimate, ic, n) {
  # `sd(ic) / sqrt(n)`, npcausal's own standard error (R/ipsi.R lines 212 and 214).
  standard_error <- sd(ic) / sqrt(n)
  low <- estimate - z * standard_error
  high <- estimate + z * standard_error
  data.frame(
    implementation = "npcausal",
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
    # `ipsi` forms no plug-in, so there is none to publish (see the point runner).
    initial_estimate = NA_real_,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

fit_scenario <- function(frame, scenario) {
  if (!scenario %in% SCENARIOS) stop(sprintf("no design for scenario %s", scenario))
  replicate <- frame$replicate[[1]]
  frame <- frame[order(frame$row), ]
  wanted <- truths[truths$scenario == scenario & truths$replicate == replicate, ]
  if (!nrow(wanted)) stop(sprintf("no truth rows for %s replicate %s", scenario, replicate))
  arguments <- ipsi_arguments(frame, unname(DELTAS), frame$fold + 1L, N_TIMES)
  fitted <- NULL
  invisible(capture.output(fitted <- study_stage("ipsi", do.call(ipsi_copy, arguments))))
  fit <- ipsi_grid_result(fitted)
  names(fit$estimate) <- names(DELTAS)
  colnames(fit$ic) <- names(DELTAS)
  rows <- lapply(seq_len(nrow(wanted)), function(index) {
    name <- wanted$estimand[[index]]
    parsed <- parse_estimand(name)
    estimate <- fit$estimate[[parsed$left]]
    ic <- fit$ic[, parsed$left]
    if (!is.null(parsed$right)) {
      estimate <- estimate - fit$estimate[[parsed$right]]
      ic <- ic - fit$ic[, parsed$right]
    }
    row_for(scenario, replicate, name, wanted$truth[[index]], estimate, ic, nrow(frame))
  })
  do.call(rbind, rows)
}

fit_one <- function(frame) {
  # One replicate holds both scenarios' samples, contiguous in the archive.
  pieces <- split(frame, frame$scenario)
  do.call(rbind, lapply(names(pieces), function(scenario) fit_scenario(pieces[[scenario]], scenario)))
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(results, expected, paths$output, versions = study_version("npcausal"))
