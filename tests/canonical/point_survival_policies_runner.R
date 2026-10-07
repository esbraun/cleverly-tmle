suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_held_survival_adapter.R")
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments(
  "usage: point_survival_policies_runner.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv"
)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
z <- qnorm(0.975)

# The plans of each scenario, the copies that realise each, and the reported visits.
# Transcribed from tests/studies/canonical_point_survival_policies.py; a gate test reads these
# lines back.
plans <- list(policy = c("natural", "policy"), mtp = c("natural", "mtp"))
copies <- c(natural = 1L, policy = 4L, mtp = 1L)
reported <- c(1L, 3L, 5L)
reference <- "natural"
nodes <- 5L

truth_for <- function(scenario, replicate, estimand) {
  selected <- truths$scenario == scenario & truths$replicate == replicate &
    truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows for %s", sum(selected), estimand))
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

carry_events <- function(frame) {
  # lmtp reads an absorbing event as 1 at every later node; the sample holds it missing.
  happened <- rep(FALSE, nrow(frame))
  for (t in seq_len(nodes)) {
    column <- sprintf("Y%d", t)
    frame[[column]][happened] <- 1
    happened <- happened | (!is.na(frame[[column]]) & frame[[column]] == 1)
  }
  frame
}

fit_plan <- function(frame, scenario, label, horizon) {
  as_label <- scenario == "mtp"
  data <- frame[c("W1", "W2", "A", sprintf("C%d", seq_len(horizon)), sprintf("Y%d", seq_len(horizon)))]
  draws <- as.matrix(frame[sprintf("shift__%s__%d", label, seq_len(copies[[label]]))])
  if (as_label) {
    data$A <- as.character(data$A)
    draws <- matrix(as.character(draws), nrow = nrow(draws))
  }
  ratios <- as.matrix(frame[sprintf("ratio__%s__%d", label, seq_len(horizon))])
  fit <- lmtp_held_survival_tmle(
    data, draws, ratios,
    trt = "A",
    outcome = sprintf("Y%d", seq_len(horizon)),
    cens = sprintf("C%d", seq_len(horizon)),
    baseline = c("W1", "W2"),
    copies = copies[[label]],
    # lmtp needs two event nodes for its survival path.  At visit 1 the risk is the binary
    # mean, so the binomial path is the same parameter.
    outcome_type = if (horizon == 1) "binomial" else "survival"
  )
  if (horizon == 1) {
    fit
  } else {
    # lmtp's survival path reports event-free survival; cleverly reports the risk.
    list(estimate = 1 - fit$estimate, initial = 1 - fit$initial, ic = -fit$ic)
  }
}

fit_scenario <- function(frame) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  frame <- carry_events(frame[order(frame$row), ])
  n <- nrow(frame)
  rows <- list()
  plan <- setdiff(plans[[scenario]], reference)
  for (horizon in reported) {
    fits <- lapply(plans[[scenario]], function(label) fit_plan(frame, scenario, label, horizon))
    names(fits) <- plans[[scenario]]
    for (label in plans[[scenario]]) {
      rows[[length(rows) + 1]] <- row_for(
        scenario, replicate, sprintf("risk_regimen[%s @ t=%d]", label, horizon),
        fits[[label]]$estimate, fits[[label]]$initial, fits[[label]]$ic, n
      )
    }
    rows[[length(rows) + 1]] <- row_for(
      scenario, replicate, sprintf("ate_regimen[%s vs %s @ t=%d]", plan, reference, horizon),
      fits[[plan]]$estimate - fits[[reference]]$estimate,
      fits[[plan]]$initial - fits[[reference]]$initial,
      fits[[plan]]$ic - fits[[reference]]$ic, n
    )
  }
  do.call(rbind, rows)
}

fit_one <- function(frame) {
  do.call(rbind, lapply(split(frame, frame$scenario), fit_scenario))
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(results, expected, paths$output, versions = study_version("lmtp"))
