# Shared transcription of an R tmle point-treatment fit into the evidence schema.

tmle_point_rows <- function(
    fit, qn, weights, truths, scenario, replicate, implementation = "tmle-r") {
  required <- c("EY0", "EY1", "ATE", "RR", "OR")
  missing <- setdiff(required, names(fit$estimates))
  if (length(missing)) {
    stop(sprintf("tmle fit omitted estimates: %s", paste(missing, collapse = ", ")))
  }
  if (!is.matrix(qn) || ncol(qn) != 2L || nrow(qn) != length(weights)) {
    stop("qn must be an n by 2 matrix aligned with weights")
  }
  if (any(!is.finite(qn)) || any(qn <= 0 | qn >= 1)) {
    stop("binary-outcome nuisance predictions must be finite and strictly between zero and one")
  }
  if (any(!is.finite(weights)) || any(weights <= 0)) {
    stop("observation weights must be finite and strictly positive")
  }

  initial_means <- c(
    ey0 = stats::weighted.mean(qn[, 1], weights),
    ey1 = stats::weighted.mean(qn[, 2], weights)
  )
  odds <- function(value) value / (1 - value)
  initial <- c(
    initial_means,
    ate = initial_means[["ey1"]] - initial_means[["ey0"]],
    rr = initial_means[["ey1"]] / initial_means[["ey0"]],
    or = odds(initial_means[["ey1"]]) / odds(initial_means[["ey0"]])
  )
  estimate_names <- c(ey0 = "EY0", ey1 = "EY1", ate = "ATE", rr = "RR", or = "OR")
  scales <- c(ey0 = "identity", ey1 = "identity", ate = "identity", rr = "log", or = "log")

  make_row <- function(estimand) {
    value <- fit$estimates[[estimate_names[[estimand]]]]
    scale <- scales[[estimand]]
    estimate <- as.numeric(value$psi)
    inference_estimate <- if (scale == "log") as.numeric(value$log.psi) else estimate
    variance <- if (scale == "log") as.numeric(value$var.log.psi) else as.numeric(value$var.psi)
    interval <- as.numeric(value$CI)
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
    numeric_values <- c(estimate, inference_estimate, variance, interval, truth, initial[[estimand]])
    if (length(interval) != 2L || any(!is.finite(numeric_values)) || variance <= 0) {
      stop(sprintf("tmle returned an invalid %s result for %s/%s", estimand, scenario, replicate))
    }
    if (interval[[1]] > interval[[2]]) {
      stop(sprintf("tmle returned a reversed %s interval for %s/%s", estimand, scenario, replicate))
    }
    if (scale == "log" && (estimate <= 0 || abs(log(estimate) - inference_estimate) > 1e-10)) {
      stop(sprintf("tmle returned inconsistent native log inference for %s", estimand))
    }
    data.frame(
      implementation = implementation,
      scenario = scenario,
      replicate = replicate,
      n = nrow(qn),
      estimand = estimand,
      truth = truth,
      estimate = estimate,
      inference_estimate = inference_estimate,
      std_error = sqrt(variance),
      ci_lower = interval[[1]],
      ci_upper = interval[[2]],
      inference_scale = scale,
      covered = as.integer(interval[[1]] <= truth && truth <= interval[[2]]),
      initial_estimate = unname(initial[[estimand]]),
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  }

  do.call(rbind, lapply(names(estimate_names), make_row))
}
