suppressPackageStartupMessages(library(tmle))
source("/fixture/study_harness.R")
source("/fixture/tmle_continuous_point_adapter.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(paths$samples), stringsAsFactors = FALSE, check.names = FALSE)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)

required_sample_columns <- c("scenario", "replicate", "Y", "A", "W1", "W2", "obs_weight")
missing_sample_columns <- setdiff(required_sample_columns, names(samples))
if (length(missing_sample_columns)) {
  stop(sprintf("samples omitted columns: %s", paste(missing_sample_columns, collapse = ", ")))
}
required_truth_columns <- c("scenario", "replicate", "estimand", "truth")
missing_truth_columns <- setdiff(required_truth_columns, names(truths))
if (length(missing_truth_columns)) {
  stop(sprintf("truths omitted columns: %s", paste(missing_truth_columns, collapse = ", ")))
}
if (anyNA(samples[required_sample_columns])) stop("required sample columns contain missing values")
if (anyNA(truths[required_truth_columns])) stop("required truth columns contain missing values")
if (any(!samples$A %in% c(0, 1))) stop("tmle comparison requires binary treatment")
if (any(!is.finite(samples$obs_weight)) || any(samples$obs_weight <= 0)) {
  stop("observation weights must be finite and strictly positive")
}
if (!nrow(samples)) stop("samples contain no observations")
fit_one <- function(frame) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  if (length(unique(frame$scenario)) != 1L || length(unique(frame$replicate)) != 1L) {
    stop("a fit group contains more than one pairing key")
  }
  fit <- tmle::tmle(
    Y = frame$Y,
    A = frame$A,
    W = frame[c("W1", "W2")],
    obsWeights = frame$obs_weight,
    family = "gaussian",
    fluctuation = "linear",
    Qform = "Y~A+W1+W2",
    gform = "A~W1+W2",
    gbound = c(0.01, 0.99),
    cvQinit = FALSE,
    prescreenW.g = FALSE,
    verbose = FALSE
  )
  qn <- fit$Qinit$Q
  if (!is.matrix(qn) || ncol(qn) != 2L || nrow(qn) != nrow(frame)) {
    stop("tmle did not retain an aligned two-arm initial outcome regression")
  }
  tmle_continuous_point_rows(
    fit,
    qn,
    frame$obs_weight,
    truths,
    scenario,
    replicate,
    implementation = "tmle-r-learned-weighted"
  )
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
