# Shared transcription of a continuous-outcome R tmle point-treatment fit.

tmle_continuous_point_rows <- function(
    fit, qn, weights, truths, scenario, replicate, implementation = "tmle-r") {
  estimate_names <- c(ey0 = "EY0", ey1 = "EY1", ate = "ATE")
  missing <- setdiff(unname(estimate_names), names(fit$estimates))
  if (length(missing)) {
    stop(sprintf("tmle fit omitted estimates: %s", paste(missing, collapse = ", ")))
  }
  if (!is.matrix(qn) || ncol(qn) != 2L || nrow(qn) != length(weights)) {
    stop("qn must be an n by 2 matrix aligned with weights")
  }
  if (any(!is.finite(qn))) stop("continuous-outcome nuisance predictions must be finite")
  if (any(!is.finite(weights)) || any(weights <= 0)) {
    stop("observation weights must be finite and strictly positive")
  }

  initial_means <- c(
    ey0 = stats::weighted.mean(qn[, 1], weights),
    ey1 = stats::weighted.mean(qn[, 2], weights)
  )
  initial <- c(initial_means, ate = initial_means[["ey1"]] - initial_means[["ey0"]])

  make_row <- function(estimand) {
    value <- fit$estimates[[estimate_names[[estimand]]]]
    selected <- (
      truths$scenario == scenario &
        truths$replicate == replicate &
        truths$estimand == estimand
    )
    if (sum(selected) != 1L) {
      stop(sprintf(
        "truth join found %d rows for %s/%s/%s",
        sum(selected), scenario, replicate, estimand
      ))
    }
    truth <- as.numeric(truths$truth[selected])
    estimate <- as.numeric(value$psi)
    variance <- as.numeric(value$var.psi)
    interval <- as.numeric(value$CI)
    numeric_values <- c(estimate, variance, interval, truth, initial[[estimand]])
    if (length(interval) != 2L || any(!is.finite(numeric_values)) || variance <= 0) {
      stop(sprintf("tmle returned an invalid %s result for %s/%s", estimand, scenario, replicate))
    }
    if (interval[[1]] > interval[[2]]) {
      stop(sprintf("tmle returned a reversed %s interval for %s/%s", estimand, scenario, replicate))
    }
    data.frame(
      implementation = implementation,
      scenario = scenario,
      replicate = replicate,
      n = nrow(qn),
      estimand = estimand,
      truth = truth,
      estimate = estimate,
      inference_estimate = estimate,
      std_error = sqrt(variance),
      ci_lower = interval[[1]],
      ci_upper = interval[[2]],
      inference_scale = "identity",
      covered = as.integer(interval[[1]] <= truth && truth <= interval[[2]]),
      initial_estimate = unname(initial[[estimand]]),
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  }

  do.call(rbind, lapply(names(estimate_names), make_row))
}
