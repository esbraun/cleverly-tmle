suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_mtp_adapter.R")
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments(
  "usage: longitudinal_mtp_runner.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv"
)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
z <- qnorm(0.975)

# The plans of each scenario, as cleverly labels and the column slugs of the shifted values and
# ratios.  Transcribed from tests/studies/canonical_longitudinal_mtp.py; a gate test reads
# these vectors back.
plans <- list(
  mtp_continuous = c(
    natural = "natural", up = "up", "scale at 2" = "scale_at_2",
    "up then history" = "up_then_history"
  ),
  mtp_continuous_crossfit = c(
    natural = "natural", up = "up", "scale at 2" = "scale_at_2",
    "up then history" = "up_then_history"
  ),
  mtp_categorical = c(natural = "natural", "minus one" = "minus_one", gated = "gated"),
  rr_tilt = c(natural = "natural", "rr 0.5" = "rr_0_5")
)
copies <- c(mtp_continuous = 1L, mtp_continuous_crossfit = 1L, mtp_categorical = 1L, rr_tilt = 2L)
as_label <- c(
  mtp_continuous = FALSE, mtp_continuous_crossfit = FALSE, mtp_categorical = TRUE, rr_tilt = FALSE
)

truth_for <- function(scenario, replicate, estimand) {
  selected <- truths$replicate == replicate & truths$scenario == scenario &
    truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows", sum(selected)))
  truths$truth[selected]
}

row_for <- function(scenario, replicate, name, estimate, initial, ic, n) {
  truth <- truth_for(scenario, replicate, name)
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
  replicate <- frame$replicate[[1]]
  frame <- frame[order(frame$row), ]
  labels <- plans[[scenario]]
  fits <- lapply(names(labels), function(label) {
    lmtp_mtp_tmle(
      frame,
      labels[[label]],
      trt = c("A1", "A2"),
      outcome = "Y",
      baseline = "W",
      time_vary = list(NULL, "L2"),
      copies = copies[[scenario]],
      as_label = as_label[[scenario]]
    )
  })
  names(fits) <- names(labels)
  rows <- lapply(names(labels), function(label) {
    fit <- fits[[label]]
    row_for(
      scenario, replicate, sprintf("ey_regimen[%s]", label),
      fit$estimate, fit$initial, fit$ic, nrow(frame)
    )
  })
  for (label in setdiff(names(labels), "natural")) {
    left <- fits[[label]]
    right <- fits[["natural"]]
    rows[[length(rows) + 1]] <- row_for(
      scenario,
      replicate,
      sprintf("ate_regimen[%s vs natural]", label),
      left$estimate - right$estimate,
      left$initial - right$initial,
      left$ic - right$ic,
      nrow(frame)
    )
  }
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
