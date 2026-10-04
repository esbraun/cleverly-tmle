suppressPackageStartupMessages(library(npcausal))
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
scenario <- "stratified_incremental"
z <- qnorm(0.975)

# npcausal has no stratified incremental estimator, so the stratum parameters are the marginal
# ones of the law given V = s: one `ipsi` call per stratum subset and multiplier.  The marginal
# names of the stratified fit are one `ipsi` call per multiplier on every row.
#
# One call per multiplier, not one call with `delta.seq = c(1, 2, 0.5)`, for the reason
# tests/canonical/npcausal_incremental/run_study.R gives: with k > 1 `return_ifvals = TRUE`
# recycles the subtrahend down the flattened matrix, and with k = 1 it is a scalar.
DELTAS <- list(
  list(delta = 1, name = "natural course"),
  list(delta = 2, name = "odds x2"),
  list(delta = 0.5, name = "odds x0.5")
)

truth_for <- function(replicate, estimand) {
  selected <- truths$replicate == replicate & truths$scenario == scenario & truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows for %s", sum(selected), estimand))
  truths$truth[selected]
}

indicators <- function(frame, columns) {
  # One indicator per non-reference level of each column, so `SL.glm.interaction` spans every
  # cell of the columns and the treatment.  A level absent from the rows adds no column.
  design <- list()
  for (column in columns) {
    for (level in c(1, 2)) {
      values <- as.numeric(frame[[column]] == level)
      if (any(values > 0)) design[[sprintf("%s%d", column, level)]] <- values
    }
  }
  as.data.frame(design)
}

fit_delta <- function(frame, design, delta, seed) {
  # Reseeded to one value for every multiplier of one subset, so the three calls share their
  # folds and nuisance fits, as the marginal runner does.  `nsplits = 2`: the documented
  # single-split path selects no training rows at this commit.
  set.seed(seed)
  n <- nrow(frame)
  fit <- ipsi(
    y = frame$Y,
    a = frame$A,
    x.trt = design,
    x.out = design,
    time = rep(1L, n),
    id = seq_len(n),
    delta.seq = delta,
    nsplits = 2,
    progress_bar = FALSE,
    return_ifvals = TRUE,
    fit = "sl",
    sl.lib = c("SL.glm.interaction")
  )
  influence <- as.numeric(fit$ifvals[, 1])
  if (abs(sd(influence) - fit$res.ptwise$se[[1]]) > 1e-9 * max(1, fit$res.ptwise$se[[1]])) {
    stop("npcausal influence values disagree with its own published standard error")
  }
  list(estimate = fit$res.ptwise$est[[1]], ic = influence)
}

row_for <- function(replicate, name, fit, n) {
  truth <- truth_for(replicate, name)
  standard_error <- sd(fit$ic) / sqrt(n)
  low <- fit$estimate - z * standard_error
  high <- fit$estimate + z * standard_error
  data.frame(
    implementation = "npcausal-stratified",
    scenario = scenario,
    replicate = replicate,
    n = n,
    estimand = name,
    truth = truth,
    estimate = fit$estimate,
    inference_estimate = fit$estimate,
    std_error = standard_error,
    ci_lower = low,
    ci_upper = high,
    inference_scale = "identity",
    covered = as.integer(low <= truth && truth <= high),
    initial_estimate = NA_real_,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

# The stratum estimate's curve is on the subset's own scale, so its standard error is
# sd(ic) / sqrt(n_s), the same number the stratified fit's embedded curve gives on the full
# sample.  The published n is the replication's n, which the schema requires of every row.
rows_for <- function(frame, design, suffix, seed, replicate, n_total) {
  fits <- lapply(DELTAS, function(entry) fit_delta(frame, design, entry$delta, seed))
  names(fits) <- vapply(DELTAS, function(entry) entry$name, character(1))
  rows <- list()
  scale <- sqrt(n_total / nrow(frame))
  standardised <- function(fit) list(estimate = fit$estimate, ic = fit$ic * scale)
  for (name in names(fits)) {
    rows[[length(rows) + 1]] <- row_for(
      replicate, sprintf("ey_ipsi[%s]%s", name, suffix), standardised(fits[[name]]), n_total
    )
  }
  natural <- fits[["natural course"]]
  for (name in c("odds x2", "odds x0.5")) {
    contrast <- list(
      estimate = fits[[name]]$estimate - natural$estimate,
      ic = fits[[name]]$ic - natural$ic
    )
    rows[[length(rows) + 1]] <- row_for(
      replicate,
      sprintf("ate_ipsi[%s vs natural course]%s", name, suffix),
      standardised(contrast),
      n_total
    )
  }
  do.call(rbind, rows)
}

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  n_total <- nrow(frame)
  seed <- 10000 * replicate + 7
  out <- list(rows_for(frame, indicators(frame, c("W", "V")), "", seed, replicate, n_total))
  for (stratum in c(0, 1, 2)) {
    subset <- frame[frame$V == stratum, , drop = FALSE]
    out[[length(out) + 1]] <- rows_for(
      subset, indicators(subset, "W"), sprintf("[V=%d]", stratum), seed + stratum + 1,
      replicate, n_total
    )
  }
  result <- do.call(rbind, out)
  stopifnot(nrow(result) == 20L, !anyDuplicated(result$estimand))
  result
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(results, expected, paths$output, versions = study_version("npcausal"))
