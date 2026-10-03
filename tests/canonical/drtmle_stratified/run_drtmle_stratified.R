suppressPackageStartupMessages(library(drtmle))
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments("usage: run_drtmle_stratified.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
scenario <- "stratified_both_correct"
critical <- qnorm(0.975)

# drtmle 1.1.2 documents vector cvFolds, but an early scalar branch evaluates the vector in an
# if statement.  Install the documented assignment at make_validRows and pass its count through
# that scalar branch, as tests/canonical/drtmle/run_drtmle.R does.  The wrapper changes only fold
# construction.
install_folds <- function(folds) {
  ns <- asNamespace("drtmle")
  original <- get("make_validRows", envir = ns)
  utils::assignInNamespace("make_validRows", function(cvFolds, n, ...) {
    stopifnot(n == length(folds))
    original(folds, n = n, ...)
  }, ns = "drtmle")
  function() utils::assignInNamespace("make_validRows", original, ns = "drtmle")
}

truth_for <- function(replicate, estimand) {
  selected <- truths$replicate == replicate & truths$scenario == scenario & truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows for %s", sum(selected), estimand))
  truths$truth[selected]
}

# drtmle has no stratified form, so a stratum is drtmle on the stratum's rows, with the
# subject's own initial arrays and the rows' own fold labels.  Its reduced regressions are then
# fitted on the stratum's rows, which is the construction the stratified subject uses.
fit_subset <- function(frame, suffix, replicate, n_total) {
  folds <- as.integer(frame$fold)
  # Renumber the labels present in the subset to 1..K, so a stratum that holds no row of some
  # fold still passes drtmle a contiguous assignment.  Every row keeps its fold.
  folds <- as.integer(factor(folds))
  restore <- install_folds(folds)
  on.exit(restore(), add = TRUE)
  fit <- drtmle::drtmle(
    Y = frame$Y,
    A = frame$A,
    W = frame[c("W1", "W2", "W12")],
    a_0 = c(0, 1),
    family = binomial(),
    Qn = list(frame$qn0, frame$qn1),
    gn = list(1 - frame$gn1, frame$gn1),
    glm_Qr = "gn",
    glm_gr = "Qn",
    guard = c("Q", "g"),
    reduction = "univariate",
    maxIter = 100,
    tolIC = 1e-8,
    tolg = 0.01,
    Qsteps = 2,
    cvFolds = length(unique(folds)),
    se_cv = "none",
    returnModels = FALSE,
    use_future = FALSE
  )
  estimates <- as.numeric(fit$drtmle$est)
  covariance <- fit$drtmle$cov
  psi <- c(estimates[[1]], estimates[[2]], estimates[[2]] - estimates[[1]])
  # The subset's standard errors are on the subset's scale, sqrt(var / n_s): the number the
  # stratified fit's embedded curve gives on the full sample.
  se <- c(
    sqrt(covariance[1, 1]),
    sqrt(covariance[2, 2]),
    sqrt(covariance[1, 1] + covariance[2, 2] - 2 * covariance[1, 2])
  )
  names <- c(sprintf("ey[0]%s", suffix), sprintf("ey[1]%s", suffix), sprintf("ate%s", suffix))
  truth <- vapply(names, function(name) truth_for(replicate, name), numeric(1))
  initial <- c(mean(frame$qn0), mean(frame$qn1), mean(frame$qn1 - frame$qn0))
  low <- psi - critical * se
  high <- psi + critical * se
  data.frame(
    implementation = "drtmle-r-stratified",
    scenario = scenario,
    replicate = replicate,
    n = n_total,
    estimand = names,
    truth = unname(truth),
    estimate = psi,
    inference_estimate = psi,
    std_error = se,
    ci_lower = low,
    ci_upper = high,
    inference_scale = "identity",
    covered = as.integer(low <= truth & truth <= high),
    initial_estimate = initial,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  n_total <- nrow(frame)
  out <- list()
  for (stratum in c(0, 1, 2)) {
    subset <- frame[frame$V == stratum, , drop = FALSE]
    out[[length(out) + 1]] <- fit_subset(subset, sprintf("[V=%d]", stratum), replicate, n_total)
  }
  result <- do.call(rbind, out)
  stopifnot(nrow(result) == 9L, !anyDuplicated(result$estimand))
  result
}

samples <- read.csv(gzfile(paths$samples), stringsAsFactors = FALSE)
groups <- split(samples, samples$replicate)
expected <- length(groups)
rm(samples)
invisible(gc())
results <- parallel::mclapply(
  seq_along(groups), study_fitter(groups, fit_one),
  mc.cores = study_cores(groups), mc.preschedule = FALSE
)
study_collect(results, expected, paths$output, versions = study_version("drtmle"))
