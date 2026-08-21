suppressPackageStartupMessages({
  library(data.table)
  library(sl3)
  library(tmle3)
  library(ctmle3)
})
options(digits = 17)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: run_ctmle3_oat.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(args[[1]]), stringsAsFactors = FALSE)
truths <- read.csv(args[[2]], stringsAsFactors = FALSE)
output <- args[[3]]
truth_key <- paste(truths$scenario, truths$replicate, sep = "|")

wald <- function(estimate, influence, transform = "identity") {
  n <- length(influence)
  se <- stats::sd(influence) / sqrt(n)
  z <- stats::qnorm(0.975)
  if (transform == "identity") {
    ci <- estimate + c(-1, 1) * z * se
    return(list(estimate = estimate, inference = estimate, se = se, ci = ci))
  }
  log_estimate <- log(estimate)
  ci <- exp(log_estimate + c(-1, 1) * z * se)
  list(estimate = estimate, inference = log_estimate, se = se, ci = ci)
}

fit_one <- function(frame, scenario, replicate) {
  row <- match(paste(scenario, replicate, sep = "|"), truth_key)
  if (is.na(row)) stop(sprintf("no truth for %s replicate %s", scenario, replicate))
  covariates <- c("W1", "W2", "W3")
  data <- frame[c(covariates, "A", "Y")]
  nodes <- list(W = covariates, A = "A", Y = "Y")
  learners <- list(A = Lrnr_glm$new(), Y = Lrnr_glm$new())
  fit <- tmle3(tmle_oat_TSM_all(), data, nodes, learners)
  estimates <- fit$estimates
  if (length(estimates) != 2) stop("the two-arm OAT comparison did not return two means")
  psi <- vapply(estimates, `[[`, numeric(1), "psi")
  influence <- do.call(cbind, lapply(estimates, `[[`, "IC"))
  initial <- as.numeric(fit$initial_psi)
  truth <- function(name) truths[[paste0("truth_", name)]][[row]]
  make_row <- function(name, value, curve, native, initial_value) {
    result <- wald(value, curve, native)
    reference <- truth(name)
    data.frame(
      implementation = "tlverse-ctmle3-oat",
      scenario = scenario,
      replicate = replicate,
      n = nrow(frame),
      estimand = name,
      truth = reference,
      estimate = result$estimate,
      inference_estimate = result$inference,
      std_error = result$se,
      ci_lower = result$ci[[1]],
      ci_upper = result$ci[[2]],
      inference_scale = native,
      covered = as.integer(result$ci[[1]] <= reference && reference <= result$ci[[2]]),
      initial_estimate = initial_value,
      stringsAsFactors = FALSE
    )
  }
  rows <- list(
    make_row("ey0", psi[[1]], influence[, 1], "identity", initial[[1]]),
    make_row("ey1", psi[[2]], influence[, 2], "identity", initial[[2]]),
    make_row("ate", psi[[2]] - psi[[1]], influence[, 2] - influence[, 1], "identity", initial[[2]] - initial[[1]])
  )
  if (scenario == "binary") {
    rr <- psi[[2]] / psi[[1]]
    rr_curve <- influence[, 2] / psi[[2]] - influence[, 1] / psi[[1]]
    odds <- function(x) x / (1 - x)
    or <- odds(psi[[2]]) / odds(psi[[1]])
    or_curve <- influence[, 2] / (psi[[2]] * (1 - psi[[2]])) -
      influence[, 1] / (psi[[1]] * (1 - psi[[1]]))
    rows[[length(rows) + 1]] <- make_row("rr", rr, rr_curve, "log", initial[[2]] / initial[[1]])
    rows[[length(rows) + 1]] <- make_row("or", or, or_curve, "log", odds(initial[[2]]) / odds(initial[[1]]))
  }
  do.call(rbind, rows)
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
failed <- vapply(results, inherits, logical(1), what = "try-error")
if (any(failed)) stop(paste(vapply(results[failed], as.character, character(1)), collapse = "\n"))
out <- do.call(rbind, results)
expected <- 5L * length(groups)
if (nrow(out) != expected) stop("an OAT reference replication was silently dropped")
write.csv(out, output, row.names = FALSE)
cat("ctmle3 ", as.character(packageVersion("ctmle3")), "\n", sep = "")
cat("tmle3 ", as.character(packageVersion("tmle3")), "\n", sep = "")
cat("sl3 ", as.character(packageVersion("sl3")), "\n", sep = "")
cat("R ", R.version.string, "\n", sep = "")
