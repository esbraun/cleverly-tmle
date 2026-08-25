suppressPackageStartupMessages(library(drtmle))
options(digits = 17)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: run_drtmle.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(args[[1]]), stringsAsFactors = FALSE)
truths <- read.csv(args[[2]], stringsAsFactors = FALSE)
output <- args[[3]]
truth_key <- paste(truths$scenario, truths$replicate, sep = "|")
critical <- qnorm(0.975)

# drtmle 1.1.2 documents vector cvFolds, but an early scalar branch evaluates the vector in
# an if statement.  Install the documented assignment at make_validRows and pass its count
# through that scalar branch.  The wrapper changes only fold construction.
install_folds <- function(folds) {
  ns <- asNamespace("drtmle")
  original <- get("make_validRows", envir = ns)
  utils::assignInNamespace("make_validRows", function(cvFolds, n, ...) {
    stopifnot(n == length(folds))
    original(folds, n = n, ...)
  }, ns = "drtmle")
  function() utils::assignInNamespace("make_validRows", original, ns = "drtmle")
}

fit_one <- function(frame) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  truth_row <- match(paste(scenario, replicate, sep = "|"), truth_key)
  if (is.na(truth_row)) stop(sprintf("missing truth for %s/%s", scenario, replicate))
  folds <- as.integer(frame$fold)
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
    returnNuisance = TRUE,
    use_future = FALSE
  )
  score <- unlist(fit$nuisance_drtmle$meanIC)
  # The raw score only. Whether it passes is decided once, in Python, at score_check's own
  # bar and from this fit's own reported standard errors -- so both implementations answer to
  # one rule instead of each applying its own constant.
  score_max <- max(abs(score))
  if (any(!is.finite(score))) stop(sprintf("non-finite score for %s/%s", scenario, replicate))
  if (any(frame$gn1 <= 0.01 | frame$gn1 >= 0.99)) {
    stop(sprintf("propensity bound active for %s/%s", scenario, replicate))
  }
  estimates <- as.numeric(fit$drtmle$est)
  covariance <- fit$drtmle$cov
  psi <- c(estimates[[1]], estimates[[2]], estimates[[2]] - estimates[[1]])
  se <- c(
    sqrt(covariance[1, 1]),
    sqrt(covariance[2, 2]),
    sqrt(covariance[1, 1] + covariance[2, 2] - 2 * covariance[1, 2])
  )
  names(psi) <- names(se) <- c("ey0", "ey1", "ate")
  initial <- c(mean(frame$qn0), mean(frame$qn1), mean(frame$qn1 - frame$qn0))
  names(initial) <- names(psi)
  truth <- c(
    ey0 = truths$truth_ey0[[truth_row]],
    ey1 = truths$truth_ey1[[truth_row]],
    ate = truths$truth_ate[[truth_row]]
  )
  low <- psi - critical * se
  high <- psi + critical * se
  data.frame(
    implementation = "drtmle-r",
    scenario = scenario,
    replicate = replicate,
    n = nrow(frame),
    estimand = names(psi),
    truth = unname(truth[names(psi)]),
    estimate = unname(psi),
    inference_estimate = unname(psi),
    std_error = unname(se),
    ci_lower = unname(low),
    ci_upper = unname(high),
    inference_scale = "identity",
    covered = as.integer(low <= truth[names(psi)] & truth[names(psi)] <= high),
    initial_estimate = unname(initial),
    score_max = score_max,
    # drtmle returns no convergence flag, so this runner has no honest value to write and
    # writes none.  It used to write TRUE, and the study then published "24 Cleverly solver
    # failures against 0" off a column the reference could not fail.  NA travels through to
    # an empty cell beside solver_reported = FALSE.
    solver_reported = FALSE,
    solver_passed = NA,
    # This one the runner does check: the fit above stops when the propensity bound is
    # active, so a row that exists is a row the bound did not bind on.
    bound_active = FALSE,
    stringsAsFactors = FALSE
  )
}

groups <- split(samples, interaction(samples$scenario, samples$replicate, drop = TRUE))
requested <- suppressWarnings(as.integer(Sys.getenv("CLEVERLY_R_CORES", "1")))
cores <- if (is.na(requested) || requested < 1L) 1L else requested
results <- if (.Platform$OS.type == "unix" && cores > 1L) {
  # Keep one fork per core.  Each child processes its assigned replicate stream, avoiding
  # thousands of expensive forks of the multi-gigabyte shared input frame.
  parallel::mclapply(groups, fit_one, mc.cores = cores, mc.preschedule = TRUE)
} else {
  lapply(groups, fit_one)
}
failed <- !vapply(results, is.data.frame, logical(1))
if (any(failed)) stop(sprintf("%d R worker(s) returned no result", sum(failed)))
out <- do.call(rbind, results)
expected <- length(groups)
observed <- length(unique(paste(out$scenario, out$replicate)))
if (observed != expected) stop(sprintf("wrote %d of %d replications", observed, expected))
write.csv(out, output, row.names = FALSE)
cat("drtmle ", as.character(packageVersion("drtmle")), "\n", sep = "")
cat("commit ", Sys.getenv("DRTMLE_COMMIT"), "\n", sep = "")
cat("R ", R.version.string, "\n", sep = "")
