suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_sdr_repeated_adapter.R")
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments(
  "usage: longitudinal_repeated_sdr_runner.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv"
)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
z <- qnorm(0.975)

# The reference side of `longitudinal-repeated-sdr` (X32b plan sections 5.3 and K12).  The
# repeated outcome Y_1, Y_2, Y_3 has no non-absorbing mode in lmtp, so each (plan, horizon) is
# one `lmtp_sdr_with_folds` call on the frame truncated after node h:
#   trt A_1..A_h, cens C_1..C_h, outcome Y_h (binomial), baseline W,
#   time-varying blocks [[], [Y_1, L_2], [Y_2, L_3]] up to h, k = Inf, .trim = 1.
# The shifted values (`shift__<plan>__<node>__1`), the per-node ratios (`ratio__<plan>__<node>`)
# and the fold assignment come from the cleverly fit that wrote the sample, so no side
# estimates a ratio the other does not see.  The learner is `SL.cellmeans.outcomes`
# (`lmtp_sdr_repeated_adapter.R`).  The plans and estimands are read from the truth table, and
# nothing here restates a plan.  `lmtp_sdr_with_folds` also returns the first-node plug-in,
# which is the `initial_estimate` column.
blocks <- list(NULL, c("Y1", "L2"), c("Y2", "L3"))
horizons <- 3L

slug <- function(label) gsub("^_+|_+$", "", gsub("[^a-z0-9]+", "_", tolower(label)))

fit_plan <- function(frame, label, horizon) {
  nodes <- seq_len(horizon)
  # A promoted warning raises `study_fatal_error`, the class `study_stage` never records, so a
  # warning from lmtp or from the learner stops the run.  The first replicates show it.
  withCallingHandlers(
    lmtp_sdr_with_folds(
      frame,
      slug(label),
      trt = sprintf("A%d", nodes),
      outcome = sprintf("Y%d", horizon),
      baseline = "W",
      time_vary = blocks[nodes],
      cens = sprintf("C%d", nodes),
      outcome_type = "binomial",
      learners_outcome = "SL.cellmeans.outcomes"
    ),
    warning = function(condition) {
      study_fatal_error(sprintf(
        "unexpected lmtp warning for %s at t=%d: %s", label, horizon, conditionMessage(condition)
      ))
    }
  )
}

parse_estimand <- function(name) {
  # ey_regimen[L @ t=h] or ate_regimen[L vs R @ t=h].
  inner <- sub("^[a-z_]+\\[(.*)\\]$", "\\1", name)
  if (!grepl(" @ t=", inner, fixed = TRUE)) stop(sprintf("estimand %s has no horizon", name))
  horizon <- as.integer(sub(".* @ t=", "", inner))
  if (is.na(horizon) || horizon < 1L || horizon > horizons) {
    stop(sprintf("estimand %s has horizon %s", name, horizon))
  }
  inner <- sub(" @ t=[0-9]+$", "", inner)
  if (startsWith(name, "ate_regimen[")) {
    sides <- strsplit(inner, " vs ", fixed = TRUE)[[1]]
    if (length(sides) != 2L) stop(sprintf("estimand %s is not a contrast of two plans", name))
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

write_curves <- function(scenario, replicate, curves) {
  # The unit-level influence curves, for gate G1r (tests/canonical/sdr_repeated/gates.py).  A
  # declared run leaves CLEVERLY_IC_DIR unset and writes nothing.
  directory <- study_env("CLEVERLY_IC_DIR")
  if (!nzchar(directory)) return(invisible(NULL))
  dir.create(directory, showWarnings = FALSE, recursive = TRUE)
  table <- as.data.frame(curves, check.names = FALSE)
  table$row <- seq_len(nrow(table)) - 1L
  target <- file.path(directory, sprintf("ic-%s-%d.csv.gz", scenario, as.integer(replicate)))
  connection <- gzfile(target, open = "wt")
  on.exit(close(connection), add = TRUE)
  write.csv(table, connection, row.names = FALSE)
}

fit_scenario <- function(frame, scenario) {
  replicate <- frame$replicate[[1]]
  frame <- frame[order(frame$row), ]
  wanted <- truths[truths$scenario == scenario & truths$replicate == replicate, ]
  if (!nrow(wanted)) stop(sprintf("no truth rows for %s replicate %s", scenario, replicate))
  fits <- list()
  get_fit <- function(label, horizon) {
    key <- paste(label, horizon, sep = " @ ")
    if (is.null(fits[[key]])) fits[[key]] <<- fit_plan(frame, label, horizon)
    fits[[key]]
  }
  curves <- list()
  rows <- lapply(seq_len(nrow(wanted)), function(index) {
    name <- wanted$estimand[[index]]
    parsed <- parse_estimand(name)
    left <- get_fit(parsed$left, parsed$horizon)
    if (is.null(parsed$right)) {
      curves[[name]] <<- left$ic
      return(row_for(
        scenario, replicate, name, wanted$truth[[index]],
        left$estimate, left$initial, left$ic, nrow(frame)
      ))
    }
    right <- get_fit(parsed$right, parsed$horizon)
    curves[[name]] <<- left$ic - right$ic
    row_for(
      scenario, replicate, name, wanted$truth[[index]],
      left$estimate - right$estimate, left$initial - right$initial, left$ic - right$ic,
      nrow(frame)
    )
  })
  write_curves(scenario, replicate, curves)
  do.call(rbind, rows)
}

fit_one <- function(frame) {
  # One replicate holds both scenarios' samples, contiguous in the archive.
  pieces <- split(frame, frame$scenario)
  do.call(rbind, lapply(names(pieces), function(scenario) fit_scenario(pieces[[scenario]], scenario)))
}

# A replicate is 24,000 rows (two scenarios of 12,000), so this chunk holds 24 replicates and
# the harness fits them across its workers.  study_stream carries the last replicate of a chunk
# forward, so a chunk must hold several replicates for the workers to have work.
expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected, chunk_lines = 576000L)
study_collect(results, expected, paths$output, versions = study_version("lmtp"))
