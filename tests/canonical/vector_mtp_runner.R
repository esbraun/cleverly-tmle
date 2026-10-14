suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_vector_adapter.R")
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments(
  "usage: vector_mtp_runner.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv"
)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
z <- qnorm(0.975)
reference <- "natural"

# The plans of each scenario, as cleverly labels and the column slugs of the shifted
# components and ratios.  Transcribed from tests/studies/canonical_vector_mtp.py; a gate test
# reads these vectors back.
plans <- list(
  point_vector_mixed = c(natural = "natural", "dose up" = "dose_up", "flag scaled" = "flag_scaled"),
  point_vector_categorical = c(natural = "natural", "a2 off" = "a2_off", "a1 on" = "a1_on"),
  longitudinal_vector_mixed = c(
    natural = "natural", "dose up" = "dose_up", "flag scaled" = "flag_scaled"
  ),
  longitudinal_vector_ordered_classifier = c(natural = "natural", ordered = "ordered", full = "full")
)
kinds <- c(
  point_vector_mixed = "policy",
  point_vector_categorical = "policy",
  longitudinal_vector_mixed = "regimen",
  longitudinal_vector_ordered_classifier = "regimen"
)

# The outcome designs cleverly fits, as SL.glm on the columns plus the squared doses, and the
# product A1 * A2 on the categorical law, which with A1 and A2 spans the three joint-level
# indicators.  A term whose columns a node's design lacks is left out, so the two-node law's
# first regression is SL.glm on its main terms, as cleverly's least squares is.
augmenter <- function(squares, product = character()) {
  force(squares)
  force(product)
  function(frame) {
    for (column in intersect(squares, names(frame))) {
      frame[[paste0(column, "_squared")]] <- frame[[column]]^2
    }
    if (length(product) == 2 && all(product %in% names(frame))) {
      frame$product <- frame[[product[[1]]]] * frame[[product[[2]]]]
    }
    frame
  }
}

augmented_glm <- function(augment) {
  force(augment)
  function(Y, X, newX, family, obsWeights, ...) {
    inner <- SuperLearner::SL.glm(Y, augment(X), augment(newX), family, obsWeights, ...)
    inner$fit <- structure(list(inner = inner$fit, augment = augment), class = "SL.glm.augmented")
    inner
  }
}

predict.SL.glm.augmented <- function(object, newdata, ...) {
  predict(object$inner, newdata = object$augment(newdata), ...)
}

SL.glm.point <- augmented_glm(augmenter("D"))
SL.glm.categorical <- augmented_glm(augmenter(character(), c("A1", "A2")))
SL.glm.long <- augmented_glm(augmenter("D2"))
SL.glm.ordered <- augmented_glm(augmenter(c("D1", "D2")))

designs <- list(
  point_vector_mixed = list(
    trt = list(c("F", "D")), baseline = c("W1", "W2"), time_vary = list(NULL),
    learner = "SL.glm.point"
  ),
  point_vector_categorical = list(
    trt = list(c("A1", "A2")), baseline = c("W1", "W2"), time_vary = list(NULL),
    learner = "SL.glm.categorical"
  ),
  longitudinal_vector_mixed = list(
    trt = list(c("F1", "D1"), c("F2", "D2")), baseline = "W", time_vary = list(NULL, "L2"),
    learner = "SL.glm.long"
  ),
  longitudinal_vector_ordered_classifier = list(
    trt = list(c("F", "D1", "D2")), baseline = c("W1", "W2"), time_vary = list(NULL),
    learner = "SL.glm.ordered"
  )
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
  design <- designs[[scenario]]
  fitter <- if (scenario == "longitudinal_vector_ordered_classifier") {
    lmtp_vector_classifier_tmle
  } else {
    lmtp_vector_tmle
  }
  fits <- lapply(names(labels), function(label) {
    fitter(
      frame,
      labels[[label]],
      trt = design$trt,
      outcome = "Y",
      baseline = design$baseline,
      time_vary = design$time_vary,
      learners_outcome = design$learner
    )
  })
  names(fits) <- names(labels)
  kind <- kinds[[scenario]]
  rows <- lapply(names(labels), function(label) {
    fit <- fits[[label]]
    row_for(
      scenario, replicate, sprintf("ey_%s[%s]", kind, label),
      fit$estimate, fit$initial, fit$ic, nrow(frame)
    )
  })
  for (label in setdiff(names(labels), reference)) {
    left <- fits[[label]]
    right <- fits[[reference]]
    rows[[length(rows) + 1]] <- row_for(
      scenario,
      replicate,
      sprintf("ate_%s[%s vs %s]", kind, label, reference),
      left$estimate - right$estimate,
      left$initial - right$initial,
      left$ic - right$ic,
      nrow(frame)
    )
  }
  do.call(rbind, rows)
}

fit_one <- function(frame) {
  # One replicate holds every scenario's sample, contiguous in the archive.  The laws differ in
  # their columns, so each scenario keeps the columns it fills.
  pieces <- split(frame, frame$scenario)
  do.call(rbind, lapply(names(pieces), function(scenario) {
    piece <- pieces[[scenario]]
    piece <- piece[, colSums(!is.na(piece)) > 0, drop = FALSE]
    fit_scenario(piece, scenario)
  }))
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(results, expected, paths$output, versions = study_version("lmtp"))
