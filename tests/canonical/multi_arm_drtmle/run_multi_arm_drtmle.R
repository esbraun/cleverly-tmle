suppressPackageStartupMessages(library(drtmle))
source("/fixture/study_harness.R")
source("/fixture/multi_arm_helpers.R")
options(digits = 17)

paths <- study_arguments(
  "usage: run_multi_arm_drtmle.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv"
)
samples <- read.csv(gzfile(paths$samples), stringsAsFactors = FALSE, check.names = FALSE)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
truth_key <- paste(truths$scenario, truths$replicate, sep = "|")
truth_columns <- grep("^truth_", names(truths), value = TRUE)
labels <- c("high", "low", "medium")

# drtmle 1.1.2 documents a fold vector, but an early scalar branch tests it in ``if``.
# Install the supplied assignment at the later constructor and pass only its count through
# that branch.  This changes fold plumbing, not the estimator.
install_folds <- function(folds) {
  namespace <- asNamespace("drtmle")
  original <- get("make_validRows", envir = namespace)
  utils::assignInNamespace("make_validRows", function(cvFolds, n, ...) {
    stopifnot(n == length(folds))
    original(folds, n = n, ...)
  }, ns = "drtmle")
  function() utils::assignInNamespace("make_validRows", original, ns = "drtmle")
}

fit_one <- function(frame) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  row <- match(paste(scenario, replicate, sep = "|"), truth_key)
  if (is.na(row)) stop(sprintf("no truth for %s replicate %s", scenario, replicate))
  truth <- as.numeric(truths[row, truth_columns])
  names(truth) <- sub("^truth_", "", truth_columns)
  folds <- as.integer(frame$fold)
  restore <- install_folds(folds)
  on.exit(restore(), add = TRUE)
  fit <- drtmle::drtmle(
    Y = frame$Y,
    A = frame$A_code,
    W = frame[c("W1", "W2", "W3")],
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
    tolg = 0.025,
    Qsteps = 2,
    cvFolds = length(unique(folds)),
    se_cv = "none",
    returnModels = FALSE,
    returnNuisance = TRUE,
    use_future = FALSE
  )
  scores <- unlist(fit$nuisance_drtmle$meanIC)
  if (any(!is.finite(scores))) stop("the R correction score is non-finite")
  means <- as.numeric(fit$drtmle$est)
  covariance <- fit$drtmle$cov
  initial <- c(mean(frame$qn0), mean(frame$qn1), mean(frame$qn2))
  rows <- multi_arm_rows_from_moments(
    means, covariance, initial, labels, truth,
    "drtmle-r-multi-arm", scenario, replicate, nrow(frame)
  )
  rows$score_max <- max(abs(scores))
  rows$solver_reported <- FALSE
  rows$solver_passed <- NA
  rows$bound_active <- FALSE
  rows
}

groups <- split(samples, interaction(samples$scenario, samples$replicate, drop = TRUE))
expected <- length(groups)
rm(samples)
invisible(gc())
results <- parallel::mclapply(
  seq_along(groups), study_fitter(groups, fit_one),
  mc.cores = study_cores(groups), mc.preschedule = TRUE
)
study_collect(
  results, expected, paths$output,
  versions = c(study_version("drtmle"), paste("commit", Sys.getenv("DRTMLE_COMMIT"))),
  key = c("scenario", "replicate"), na = "NA"
)
