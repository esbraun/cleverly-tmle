# Shared extraction of three arm means and their reference-arm contrasts.

multi_arm_rows_from_moments <- function(
    means, covariance, initial, labels, truth, implementation, scenario, replicate, n) {
  stopifnot(length(means) == length(labels), nrow(covariance) == length(labels))
  names(means) <- names(initial) <- labels
  rownames(covariance) <- colnames(covariance) <- labels
  reference <- labels[[1]]
  critical <- stats::qnorm(0.975)

  make_row <- function(name, estimate, gradient, initial_estimate, scale) {
    variance <- as.numeric(t(gradient) %*% covariance %*% gradient)
    standard_error <- sqrt(max(variance, 0))
    inference <- if (scale == "log") log(estimate) else estimate
    interval <- if (scale == "log") {
      exp(inference + c(-1, 1) * critical * standard_error)
    } else {
      estimate + c(-1, 1) * critical * standard_error
    }
    target <- unname(truth[[name]])
    data.frame(
      implementation = implementation,
      scenario = scenario,
      replicate = replicate,
      n = n,
      estimand = name,
      truth = target,
      estimate = estimate,
      inference_estimate = inference,
      std_error = standard_error,
      ci_lower = interval[[1]],
      ci_upper = interval[[2]],
      inference_scale = scale,
      covered = as.integer(interval[[1]] <= target && target <= interval[[2]]),
      initial_estimate = initial_estimate,
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
  }

  rows <- list()
  for (index in seq_along(labels)) {
    gradient <- rep(0, length(labels))
    gradient[[index]] <- 1
    rows[[length(rows) + 1]] <- make_row(
      paste0("ey[", labels[[index]], "]"),
      means[[index]], gradient, initial[[index]], "identity"
    )
  }
  reference_index <- match(reference, labels)
  for (label in labels[-reference_index]) {
    index <- match(label, labels)
    ate_gradient <- rep(0, length(labels))
    ate_gradient[[index]] <- 1
    ate_gradient[[reference_index]] <- -1
    rows[[length(rows) + 1]] <- make_row(
      paste0("ate[", label, " vs ", reference, "]"),
      means[[index]] - means[[reference_index]],
      ate_gradient,
      initial[[index]] - initial[[reference_index]],
      "identity"
    )

    rr_gradient <- rep(0, length(labels))
    rr_gradient[[index]] <- 1 / means[[index]]
    rr_gradient[[reference_index]] <- -1 / means[[reference_index]]
    rows[[length(rows) + 1]] <- make_row(
      paste0("rr[", label, " vs ", reference, "]"),
      means[[index]] / means[[reference_index]],
      rr_gradient,
      initial[[index]] / initial[[reference_index]],
      "log"
    )

    or_gradient <- rep(0, length(labels))
    or_gradient[[index]] <- 1 / (means[[index]] * (1 - means[[index]]))
    or_gradient[[reference_index]] <- -1 / (
      means[[reference_index]] * (1 - means[[reference_index]])
    )
    odds <- function(value) value / (1 - value)
    rows[[length(rows) + 1]] <- make_row(
      paste0("or[", label, " vs ", reference, "]"),
      odds(means[[index]]) / odds(means[[reference_index]]),
      or_gradient,
      odds(initial[[index]]) / odds(initial[[reference_index]]),
      "log"
    )
  }
  do.call(rbind, rows)
}

multi_arm_rows_from_tmle <- function(
    fit, labels, truth, implementation, scenario, replicate, n) {
  estimates <- fit$estimates
  if (length(estimates) != length(labels)) {
    stop(sprintf("expected %s treatment-specific means, got %s", length(labels), length(estimates)))
  }
  means <- vapply(estimates, `[[`, numeric(1), "psi")
  influence <- do.call(cbind, lapply(estimates, `[[`, "IC"))
  covariance <- stats::cov(influence) / nrow(influence)
  initial <- as.numeric(fit$initial_psi)
  multi_arm_rows_from_moments(
    means, covariance, initial, labels, truth, implementation, scenario, replicate, n
  )
}
