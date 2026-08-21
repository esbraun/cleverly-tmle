suppressPackageStartupMessages(library(ctmle))
options(digits = 17)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: run_ctmle.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(args[[1]]), stringsAsFactors = FALSE)
truths <- read.csv(args[[2]], stringsAsFactors = FALSE)
output <- args[[3]]
truth_key <- paste(truths$scenario, truths$replicate, sep = "|")

fit_one <- function(frame, scenario, replicate) {
  truth_row <- match(paste(scenario, replicate, sep = "|"), truth_key)
  if (is.na(truth_row)) stop(sprintf("no truth for %s replicate %s", scenario, replicate))
  truth <- truths$truth_ate[[truth_row]]
  binary <- startsWith(scenario, "binary")
  W <- frame[c("W1", "W2", "W3")]
  q_family <- if (binary) binomial() else gaussian()
  q_fit <- glm(Y ~ A + W1 + W2 + W3, data = frame, family = q_family)
  q0 <- predict(q_fit, transform(frame, A = 0), type = "response")
  q1 <- predict(q_fit, transform(frame, A = 1), type = "response")
  folds <- lapply(sort(unique(frame$selection_fold)), function(k) which(frame$selection_fold == k))
  ordered <- endsWith(scenario, "ordered") || endsWith(scenario, "discrete")
  fit <- ctmleDiscrete(
    Y = frame$Y,
    A = frame$A,
    W = W,
    Wg = W,
    Q = cbind(q0, q1),
    preOrder = ordered,
    order = if (ordered) seq_len(ncol(W)) else NULL,
    family = if (binary) "binomial" else "gaussian",
    like_type = if (binary) "loglike" else "RSS",
    gbound = 0.025,
    PEN = FALSE,
    V = 5,
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
fit_group <- function(frame) {
  tryCatch(
    fit_one(frame, frame$scenario[[1]], frame$replicate[[1]]),
    error = function(condition) stop(sprintf(
      "%s replicate %s: %s",
      frame$scenario[[1]], frame$replicate[[1]], conditionMessage(condition)
    ), call. = FALSE)
  )
}
requested <- suppressWarnings(as.integer(Sys.getenv("CLEVERLY_R_CORES", "1")))
cores <- if (is.na(requested) || requested < 1L) 1L else requested
results <- parallel::mclapply(groups, fit_group, mc.cores = cores, mc.preschedule = FALSE)
out <- do.call(rbind, results)
if (nrow(out) != length(groups)) stop("a selector reference replication was silently dropped")
write.csv(out, output, row.names = FALSE)
cat("ctmle ", as.character(packageVersion("ctmle")), "\n", sep = "")
cat("R ", R.version.string, "\n", sep = "")
