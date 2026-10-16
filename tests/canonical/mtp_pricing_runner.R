suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_mtp_adapter.R")
source("/fixture/lmtp_sdr_adapter.R")
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments(
  "usage: mtp_pricing_runner.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv"
)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
z <- qnorm(0.975)

# The estimator of each scenario.  The plans and the estimands are read from the truth table,
# and each plan's shifted values and ratios from the sample, so nothing here restates a plan.
# Transcribed from tests/studies/canonical_mtp_pricing.py (SCENARIOS); a gate test reads
# these vectors back.
estimators <- c(
  pricing_ltmle_tau2 = "ltmle", pricing_sdr_tau2 = "sdr",
  pricing_ltmle_tau3 = "ltmle", pricing_sdr_tau3 = "sdr",
  pricing_ltmle_tau5 = "ltmle", pricing_sdr_tau5 = "sdr"
)
horizons <- c(
  pricing_ltmle_tau2 = 2L, pricing_sdr_tau2 = 2L,
  pricing_ltmle_tau3 = 3L, pricing_sdr_tau3 = 3L,
  pricing_ltmle_tau5 = 5L, pricing_sdr_tau5 = 5L
)
baseline <- "W1"

slug <- function(label) gsub("^_+|_+$", "", gsub("[^a-z0-9]+", "_", tolower(label)))

typed <- function(frame) {
  # One archive holds every horizon, so the columns of a longer horizon are empty in a
  # shorter one's rows.  Drop the columns this scenario did not write.
  frame[, colSums(!is.na(frame)) > 0, drop = FALSE]
}

fit_plan <- function(frame, estimator, tau, label) {
  trt <- paste0("A", seq_len(tau))
  time_vary <- as.list(paste0("L", seq_len(tau)))
  if (estimator == "ltmle") {
    # The study's exactness gate judges every pair (exactness.csv), so a prediction at lmtp's
    # bound moves that pair rather than stopping the run.
    return(lmtp_mtp_tmle(
      frame, slug(label),
      trt = trt, outcome = "Y", baseline = baseline, time_vary = time_vary,
      refuse_bound = FALSE
    ))
  }
  lmtp_sdr_with_folds(
    frame, slug(label),
    trt = trt, outcome = "Y", baseline = baseline, time_vary = time_vary
  )
}

parse_estimand <- function(name) {
  # ey_regimen[L] or ate_regimen[L vs R].
  inner <- sub("^[a-z_]+\\[(.*)\\]$", "\\1", name)
  if (startsWith(name, "ate_regimen[")) {
    sides <- strsplit(inner, " vs ", fixed = TRUE)[[1]]
    return(list(left = sides[[1]], right = sides[[2]]))
  }
  list(left = inner, right = NULL)
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
  estimator <- estimators[[scenario]]
  tau <- horizons[[scenario]]
  replicate <- frame$replicate[[1]]
  frame <- typed(frame[order(frame$row), ])
  wanted <- truths[truths$scenario == scenario & truths$replicate == replicate, ]
  if (!nrow(wanted)) stop(sprintf("no truth rows for %s replicate %s", scenario, replicate))
  fits <- list()
  get_fit <- function(label) {
    # The adapters run each lmtp call inside its own `study_stage`, so an lmtp error is a
    # recorded failure under the failure bound, and their own checks stay fatal (M7).
    if (is.null(fits[[label]])) {
      fits[[label]] <<- fit_plan(frame, estimator, tau, label)
    }
    fits[[label]]
  }
  rows <- lapply(seq_len(nrow(wanted)), function(index) {
    name <- wanted$estimand[[index]]
    parsed <- parse_estimand(name)
    left <- get_fit(parsed$left)
    if (is.null(parsed$right)) {
      return(row_for(
        scenario, replicate, name, wanted$truth[[index]],
        left$estimate, left$initial, left$ic, nrow(frame)
      ))
    }
    right <- get_fit(parsed$right)
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
# Several replicates per read, so the local workers stay busy (design review m5).
results <- study_stream(paths$samples, fit_one, expected, chunk_lines = 256000L)
study_collect(results, expected, paths$output, versions = study_version("lmtp"))
