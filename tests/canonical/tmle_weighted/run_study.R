suppressPackageStartupMessages(library(tmle))
source("/fixture/study_harness.R")
source("/fixture/tmle_point_adapter.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(paths$samples), stringsAsFactors = FALSE, check.names = FALSE)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)

required_sample_columns <- c(
  "scenario", "replicate", "Y", "A", "qn0", "qn1", "gn1", "obs_weight"
)
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
if (any(!samples$A %in% c(0, 1)) || any(!samples$Y %in% c(0, 1))) {
  stop("tmle weighted comparison requires binary A and Y")
}
covariates <- grep("^W", names(samples), value = TRUE)
if (!length(covariates)) stop("samples must contain at least one W covariate")
if (!nrow(samples)) stop("samples contain no observations")

fit_one <- function(frame) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  if (length(unique(frame$scenario)) != 1L || length(unique(frame$replicate)) != 1L) {
    stop("a fit group contains more than one pairing key")
  }
  if (any(!is.finite(frame$gn1)) || any(frame$gn1 <= 0 | frame$gn1 >= 1)) {
    stop(sprintf("invalid treatment nuisance for %s/%s", scenario, replicate))
  }
  qn <- cbind(frame$qn0, frame$qn1)
  fit <- tmle::tmle(
    Y = frame$Y,
    A = frame$A,
    W = frame[covariates],
    Q = qn,
    g1W = frame$gn1,
    obsWeights = frame$obs_weight,
    family = "binomial",
    fluctuation = "logistic",
    Qbounds = c(0.001, 0.999),
    gbound = c(0.01, 0.99),
    cvQinit = FALSE,
    verbose = FALSE
  )
  tmle_point_rows(
    fit,
    qn,
    frame$obs_weight,
    truths,
    scenario,
    replicate,
    implementation = "tmle-r-weighted"
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
