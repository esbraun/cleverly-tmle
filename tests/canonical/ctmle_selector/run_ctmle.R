suppressPackageStartupMessages(library(ctmle))
options(digits = 17)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: run_ctmle.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(args[[1]]), stringsAsFactors = FALSE)
truths <- read.csv(args[[2]], stringsAsFactors = FALSE)
output <- args[[3]]
truth_key <- paste(truths$scenario, truths$replicate, sep = "|")

# The selection split arrives from the Python side, taken off the fit that produced the
# subject's own row.  Rebuild it here and assert what a partition has to be before using
# it: a fold column that stopped meaning what it says would move this reference's answer
# silently, and the paired gates would go on passing against the wrong comparison.
selection_folds <- function(frame) {
  assignment <- frame$selection_fold
  if (anyNA(assignment)) stop("the selection fold column has missing values")
  codes <- sort(unique(assignment))
  if (!identical(as.integer(codes), seq_along(codes) - 1L)) {
    stop(sprintf("selection folds must be 0..V-1; got %s", paste(codes, collapse = ",")))
  }
  folds <- lapply(codes, function(k) which(assignment == k))
  if (sum(lengths(folds)) != nrow(frame) || anyDuplicated(unlist(folds)) != 0L) {
    stop("the selection folds do not partition the sample")
  }
  for (index in seq_along(folds)) {
    if (length(unique(frame$A[folds[[index]]])) != 2L) {
      stop(sprintf("selection fold %s does not hold both treatment arms", codes[[index]]))
    }
  }
  folds
}

fit_one <- function(frame, scenario, replicate) {
  truth_row <- match(paste(scenario, replicate, sep = "|"), truth_key)
  if (is.na(truth_row)) stop(sprintf("no truth for %s replicate %s", scenario, replicate))
  truth <- truths$truth_ate[[truth_row]]
  W <- frame[c("W1", "W2", "W3")]
  q_fit <- glm(Y ~ A + W1 + W2 + W3, data = frame, family = binomial())
  q0 <- predict(q_fit, transform(frame, A = 0), type = "response")
  q1 <- predict(q_fit, transform(frame, A = 1), type = "response")
  folds <- selection_folds(frame)
  # `ordered` and `discrete` reach the same R entry point.  This package has one
  # pre-ordered mode, and the subject's discrete candidate list is exactly the nested
  # prefix ladder that mode enumerates, so the two correspond; the strategies differ on
  # the Cleverly side, not here.
  ordered <- endsWith(scenario, "ordered") || endsWith(scenario, "discrete")
  fit <- ctmleDiscrete(
    Y = frame$Y,
    A = frame$A,
    W = W,
    Wg = W,
    Q = cbind(q0, q1),
    preOrder = ordered,
    order = if (ordered) seq_len(ncol(W)) else NULL,
    family = "binomial",
    like_type = "loglike",
    gbound = 0.025,
    PEN = FALSE,
    V = length(folds),
    folds = folds,
    detailed = TRUE
  )
  estimate <- as.numeric(fit$est)
  std_error <- sqrt(as.numeric(fit$var.psi))
  ci <- as.numeric(fit$CI)
  initial <- mean(q1 - q0)
  data.frame(
    implementation = "r-ctmle",
    scenario = scenario,
    replicate = replicate,
    n = nrow(frame),
    estimand = "ate",
    truth = truth,
    estimate = estimate,
    inference_estimate = estimate,
    std_error = std_error,
    ci_lower = ci[[1]],
    ci_upper = ci[[2]],
    inference_scale = "identity",
    covered = as.integer(ci[[1]] <= truth && truth <= ci[[2]]),
    initial_estimate = initial,
    stringsAsFactors = FALSE
  )
}

groups <- split(samples, interaction(samples$scenario, samples$replicate, drop = TRUE))
scenarios <- sort(unique(samples$scenario))

# Seeded per replication, inside the worker, and deliberately not left to the parallel
# machinery.  `mclapply` derives each forked worker's seed from scheduling, so with
# `mc.set.seed = TRUE` the stream a replication gets depends on how many cores the run had.
# The pre-ordered search consumes no randomness and is unaffected; the greedy search does,
# and without this 13 of 30 replications changed between a two-core and an eight-core run --
# which would make these committed rows a record of one machine rather than of the method.
# Seeding from the scenario and replication instead makes a replication's answer a function
# of its data alone.
fit_group <- function(frame) {
  scenario <- frame$scenario[[1]]
  set.seed(7919L * match(scenario, scenarios) + frame$replicate[[1]] + 1L)
  tryCatch(
    fit_one(frame, scenario, frame$replicate[[1]]),
    error = function(condition) stop(sprintf(
      "%s replicate %s: %s",
      scenario, frame$replicate[[1]], conditionMessage(condition)
    ), call. = FALSE)
  )
}
requested <- suppressWarnings(as.integer(Sys.getenv("CLEVERLY_R_CORES", "1")))
cores <- if (is.na(requested) || requested < 1L) 1L else requested
results <- parallel::mclapply(groups, fit_group, mc.cores = cores, mc.preschedule = FALSE)
# `mclapply` returns a `try-error` for a worker that failed rather than propagating it, so
# above one core a failed replication reaches `rbind` as a value.  The row count alone does
# not catch that; without this the study can publish a mangled frame instead of stopping.
failed <- vapply(results, inherits, logical(1), what = "try-error")
if (any(failed)) stop(paste(vapply(results[failed], as.character, character(1)), collapse = "\n"))
out <- do.call(rbind, results)
if (nrow(out) != length(groups)) stop("a selector reference replication was silently dropped")
write.csv(out, output, row.names = FALSE)
cat("ctmle ", as.character(packageVersion("ctmle")), "\n", sep = "")
cat("R ", R.version.string, "\n", sep = "")
