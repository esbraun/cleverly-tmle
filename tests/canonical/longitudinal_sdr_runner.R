suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_sdr_adapter.R")
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments(
  "usage: longitudinal_sdr_runner.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv"
)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
z <- qnorm(0.975)

# The design of each scenario.  The plans and the estimands are read from the truth table, and
# each plan's shifted values and ratios from the sample, so nothing here restates a plan.
# Transcribed from tests/studies/canonical_longitudinal_sdr.py and
# tests/canonical/longitudinal_sdr/gates.py; a gate test reads these lists back.
censored <- list(
  baseline = c("W1", "W2"), time_vary = list(NULL, "L2"), cens = c("C1", "C2"),
  outcome = "Y", survival = FALSE, copies = 1L, as_label = FALSE, learners = "SL.glm"
)
survival <- modifyList(censored, list(outcome = c("Y1", "Y2"), survival = TRUE))
discrete <- modifyList(censored, list(baseline = "W", learners = "SL.cellmeans"))
discrete_survival <- modifyList(discrete, list(outcome = c("Y1", "Y2"), survival = TRUE))
uncensored <- list(
  baseline = "W", time_vary = list(NULL, "L2"), cens = NULL, outcome = "Y", survival = FALSE,
  copies = 1L, as_label = FALSE, learners = "SL.glm"
)
designs <- list(
  sdr_end_of_study = censored,
  sdr_end_of_study_crossfit = censored,
  sdr_survival = survival,
  sdr_mtp = uncensored,
  sdr_policy = modifyList(uncensored, list(copies = 4L, as_label = TRUE)),
  gate_end_of_study_1 = discrete,
  gate_end_of_study_5 = discrete,
  gate_survival_1 = discrete_survival,
  gate_survival_5 = discrete_survival
)

slug <- function(label) gsub("^_+|_+$", "", gsub("[^a-z0-9]+", "_", tolower(label)))

typed <- function(frame) {
  # One archive holds every scenario, so a column that is a label in one scenario is read as
  # text in all of them.  Re-read each scenario's columns on their own, and drop the columns
  # another scenario wrote.
  frame[] <- lapply(frame, function(column) {
    if (is.character(column)) type.convert(column, as.is = TRUE, na.strings = c("", "NA")) else column
  })
  frame[, colSums(!is.na(frame)) > 0, drop = FALSE]
}

fit_plan <- function(frame, design, label, horizon) {
  nodes <- seq_len(horizon)
  outcome <- design$outcome
  outcome_type <- "binomial"
  if (design$survival) {
    outcome <- design$outcome[nodes]
    # lmtp requires two event nodes for outcome_type = "survival".  At the first horizon the
    # cumulative risk is the one-node binary mean, which its binomial path estimates.
    outcome_type <- if (horizon == 1) "binomial" else "survival"
  }
  fit <- lmtp_sdr_with_folds(
    frame,
    slug(label),
    trt = c("A1", "A2")[nodes],
    outcome = outcome,
    baseline = design$baseline,
    time_vary = design$time_vary[nodes],
    cens = if (is.null(design$cens)) NULL else design$cens[nodes],
    outcome_type = outcome_type,
    copies = design$copies,
    as_label = design$as_label,
    learners_outcome = design$learners
  )
  if (outcome_type == "survival") {
    # lmtp's survival path reports event-free survival; cleverly reports cumulative risk.
    fit <- list(estimate = 1 - fit$estimate, ic = -fit$ic, initial = 1 - fit$initial)
  }
  fit
}

parse_estimand <- function(name) {
  # ey_regimen[L], risk_regimen[L @ t=h], ate_regimen[L vs R] or ate_regimen[L vs R @ t=h].
  inner <- sub("^[a-z_]+\\[(.*)\\]$", "\\1", name)
  horizon <- 2L
  if (grepl(" @ t=", inner, fixed = TRUE)) {
    horizon <- as.integer(sub(".* @ t=", "", inner))
    inner <- sub(" @ t=[0-9]+$", "", inner)
  }
  if (startsWith(name, "ate_regimen[")) {
    sides <- strsplit(inner, " vs ", fixed = TRUE)[[1]]
    return(list(left = sides[[1]], right = sides[[2]], horizon = horizon))
  }
  list(left = inner, right = NULL, horizon = horizon)
}

row_for <- function(scenario, replicate, name, truth, estimate, initial, ic, n) {
  standard_error <- sd(ic) / sqrt(n)
  low <- estimate - z * standard_error
  high <- estimate + z * standard_error
  data.frame(
    implementation = "lmtp",
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

fit_scenario <- function(frame, scenario) {
  design <- designs[[scenario]]
  if (is.null(design)) stop(sprintf("no design for scenario %s", scenario))
  replicate <- frame$replicate[[1]]
  frame <- typed(frame[order(frame$row), ])
  wanted <- truths[truths$scenario == scenario & truths$replicate == replicate, ]
  if (!nrow(wanted)) stop(sprintf("no truth rows for %s replicate %s", scenario, replicate))
  fits <- list()
  get_fit <- function(label, horizon) {
    key <- paste(label, horizon, sep = " @ ")
    if (is.null(fits[[key]])) fits[[key]] <<- fit_plan(frame, design, label, horizon)
    fits[[key]]
  }
  rows <- lapply(seq_len(nrow(wanted)), function(index) {
    name <- wanted$estimand[[index]]
    parsed <- parse_estimand(name)
    left <- get_fit(parsed$left, parsed$horizon)
    if (is.null(parsed$right)) {
      return(row_for(
        scenario, replicate, name, wanted$truth[[index]],
        left$estimate, left$initial, left$ic, nrow(frame)
      ))
    }
    right <- get_fit(parsed$right, parsed$horizon)
    row_for(
      scenario, replicate, name, wanted$truth[[index]],
      left$estimate - right$estimate, left$initial - right$initial, left$ic - right$ic,
      nrow(frame)
    )
  })
  do.call(rbind, rows)
}

fit_one <- function(frame) {
  # One replicate holds every scenario's sample, contiguous in the archive.
  pieces <- split(frame, frame$scenario)
  do.call(rbind, lapply(names(pieces), function(scenario) fit_scenario(pieces[[scenario]], scenario)))
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(results, expected, paths$output, versions = study_version("lmtp"))
