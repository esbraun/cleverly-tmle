suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_mtp_adapter.R")
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments(
  "usage: policy_point_mtp_runner.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv"
)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
z <- qnorm(0.975)

# The policies of the study, as cleverly labels and the column slugs of the policy doses and
# ratios.  Transcribed from tests/studies/canonical_policy_point_mtp.py; a gate test reads
# this vector back.
policies <- c(
  "natural course" = "natural_course", "x1.25" = "x1_25", piecewise = "piecewise",
  "halve below 3" = "halve_below_3"
)
reference <- "natural course"

square_dose <- function(frame) {
  # The outcome design cleverly fits (QuadraticOutcome): the columns and the squared dose.
  # lmtp reads the policy dose under the treatment's own name.
  if (!("A" %in% names(frame))) stop("the outcome design has no column A")
  frame$A_squared <- frame$A^2
  frame
}

SL.glm.quadratic <- function(Y, X, newX, family, obsWeights, ...) {
  inner <- SuperLearner::SL.glm(Y, square_dose(X), square_dose(newX), family, obsWeights, ...)
  inner$fit <- structure(list(inner = inner$fit), class = "SL.glm.quadratic")
  inner
}

predict.SL.glm.quadratic <- function(object, newdata, ...) {
  predict(object$inner, newdata = square_dose(newdata), ...)
}

truth_for <- function(replicate, estimand) {
  selected <- truths$replicate == replicate & truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows", sum(selected)))
  truths$truth[selected]
}

row_for <- function(scenario, replicate, name, estimate, initial, ic, n) {
  truth <- truth_for(replicate, name)
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

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  scenario <- frame$scenario[[1]]
  frame <- frame[order(frame$row), ]
  fits <- lapply(names(policies), function(label) {
    lmtp_mtp_tmle(
      frame,
      policies[[label]],
      trt = "A",
      outcome = "Y",
      baseline = c("W1", "W2"),
      time_vary = list(NULL),
      learners_outcome = "SL.glm.quadratic"
    )
  })
  names(fits) <- names(policies)
  rows <- lapply(names(policies), function(label) {
    fit <- fits[[label]]
    row_for(
      scenario, replicate, sprintf("ey_policy[%s]", label),
      fit$estimate, fit$initial, fit$ic, nrow(frame)
    )
  })
  for (label in setdiff(names(policies), reference)) {
    left <- fits[[label]]
    right <- fits[[reference]]
    rows[[length(rows) + 1]] <- row_for(
      scenario,
      replicate,
      sprintf("ate_policy[%s vs %s]", label, reference),
      left$estimate - right$estimate,
      left$initial - right$initial,
      left$ic - right$ic,
      nrow(frame)
    )
  }
  do.call(rbind, rows)
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(results, expected, paths$output, versions = study_version("lmtp"))
