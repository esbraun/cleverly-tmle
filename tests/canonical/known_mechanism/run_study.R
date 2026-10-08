suppressPackageStartupMessages(library(tmle))
source("/fixture/study_harness.R")
source("/fixture/tmle_point_adapter.R")
options(digits = 17)

# R tmle 2.1.1 with the design's known mechanism as g1W, on the samples and the initial
# outcome regression the Python phase wrote.  gbound = 0.1 bounds each arm's total from below,
# and every declared g lies in [0.25, 0.70], so no bound binds.

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(paths$samples), stringsAsFactors = FALSE, check.names = FALSE)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)

required <- c("scenario", "replicate", "Y", "A", "W1", "W2", "W3", "qn0", "qn1", "gn1")
missing_columns <- setdiff(required, names(samples))
if (length(missing_columns)) {
  stop(sprintf("samples omitted columns: %s", paste(missing_columns, collapse = ", ")))
}
if (anyNA(samples[required])) stop("required sample columns contain missing values")
if (any(samples$gn1 < 0.25 - 1e-12 | samples$gn1 > 0.70 + 1e-12)) {
  stop("a declared mechanism value lies outside [0.25, 0.70], so the bound could bind")
}

conditional_row <- function(fit, frame, scenario, replicate, estimand) {
  value <- fit$estimates[[toupper(estimand)]]
  if (is.null(value)) stop(sprintf("tmle fit omitted %s", estimand))
  selected <- (
    truths$scenario == scenario & truths$replicate == replicate & truths$estimand == estimand
  )
  if (sum(selected) != 1L) stop(sprintf("truth join failed for %s/%s/%s", scenario, replicate, estimand))
  truth <- as.numeric(truths$truth[selected])
  interval <- as.numeric(value$CI)
  variance <- as.numeric(value$var.psi)
  blip <- frame$qn1 - frame$qn0
  initial <- if (estimand == "att") mean(blip[frame$A == 1]) else mean(blip[frame$A == 0])
  if (length(interval) != 2L || !is.finite(variance) || variance <= 0) {
    stop(sprintf("tmle returned an invalid %s for %s/%s", estimand, scenario, replicate))
  }
  data.frame(
    implementation = "tmle-r-known-g",
    scenario = scenario,
    replicate = replicate,
    n = nrow(frame),
    estimand = estimand,
    truth = truth,
    estimate = as.numeric(value$psi),
    inference_estimate = as.numeric(value$psi),
    std_error = sqrt(variance),
    ci_lower = interval[[1]],
    ci_upper = interval[[2]],
    inference_scale = "identity",
    covered = as.integer(interval[[1]] <= truth && truth <= interval[[2]]),
    initial_estimate = initial,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

fit_one <- function(frame) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  if (length(unique(frame$scenario)) != 1L || length(unique(frame$replicate)) != 1L) {
    stop("a fit group contains more than one pairing key")
  }
  qn <- cbind(frame$qn0, frame$qn1)
  fit <- tmle::tmle(
    Y = frame$Y,
    A = frame$A,
    W = frame[c("W1", "W2", "W3")],
    Q = qn,
    g1W = frame$gn1,
    gbound = 0.1,
    family = "binomial",
    fluctuation = "logistic",
    Qbounds = c(0.001, 0.999),
    cvQinit = FALSE,
    verbose = FALSE
  )
  rows <- tmle_point_rows(
    fit, qn, rep(1, nrow(frame)), truths, scenario, replicate,
    implementation = "tmle-r-known-g"
  )
  if (scenario == "binary_q_correct") {
    rows <- rbind(
      rows,
      conditional_row(fit, frame, scenario, replicate, "att"),
      conditional_row(fit, frame, scenario, replicate, "atc")
    )
  }
  rows
}

groups <- split(samples, interaction(samples$scenario, samples$replicate, drop = TRUE))
expected <- length(groups)
rm(samples)
invisible(gc())
results <- parallel::mclapply(
  seq_along(groups),
  study_fitter(groups, fit_one),
  mc.cores = study_cores(groups),
  mc.preschedule = TRUE
)
study_collect(
  results,
  expected = expected,
  output = paths$output,
  versions = c(
    study_version("tmle"),
    paste("source sha256", Sys.getenv("TMLE_SHA256"))
  ),
  key = c("scenario", "replicate")
)
