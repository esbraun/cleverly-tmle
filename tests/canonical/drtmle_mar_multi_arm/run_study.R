suppressPackageStartupMessages(library(drtmle))
source("/fixture/study_harness.R")
source("/fixture/multi_arm_helpers.R")
options(digits = 17)

args <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(args$samples), stringsAsFactors = FALSE, check.names = FALSE)
truths <- read.csv(args$truths, stringsAsFactors = FALSE, check.names = FALSE)
# The labels in arm-code order. multi_arm_rows_from_moments takes the first as the
# reference, and "high" is arm code 0.
labels <- c("high", "low", "mid")

fit_one <- function(frame) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  rows <- truths[truths$scenario == scenario & truths$replicate == replicate, ]
  if (nrow(rows) == 0) stop(sprintf("no truth for %s replicate %s", scenario, replicate))
  truth <- rows$truth
  names(truth) <- rows$estimand
  # cvFolds = 1 fits in-sample, so no fold vector is installed.
  fit <- drtmle::drtmle(
    Y = frame$Y,
    A = frame$A_code,
    W = frame["W"],
    DeltaY = frame$Delta,
    a_0 = 0:2,
    family = binomial(),
    Qn = list(frame$qn0, frame$qn1, frame$qn2),
    gn = list(frame$gn0, frame$gn1, frame$gn2),
    glm_Qr = "gn",
    glm_gr = "Qn",
    guard = c("Q", "g"),
    reduction = "univariate",
    maxIter = 100,
    tolIC = 1e-8,
    tolg = 0.01,
    Qsteps = 2,
    cvFolds = 1,
    se_cv = "none",
    returnModels = FALSE,
    returnNuisance = TRUE,
    use_future = FALSE
  )
  means <- as.numeric(fit$drtmle$est)
  covariance <- fit$drtmle$cov
  initial <- c(mean(frame$qn0), mean(frame$qn1), mean(frame$qn2))
  multi_arm_rows_from_moments(
    means, covariance, initial, labels, truth,
    "drtmle-r-multi-arm-mar", scenario, replicate, nrow(frame)
  )
}

groups <- split(samples, interaction(samples$scenario, samples$replicate, drop = TRUE))
cores <- study_cores(groups)
results <- parallel::mclapply(
  seq_along(groups),
  study_fitter(groups, fit_one),
  mc.cores = cores,
  mc.preschedule = TRUE
)
study_collect(
  results,
  expected = length(groups),
  output = args$output,
  versions = c(
    study_version("drtmle"),
    paste("commit", Sys.getenv("DRTMLE_COMMIT"))
  )
)
