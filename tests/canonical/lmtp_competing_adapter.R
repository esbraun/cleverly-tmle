# Competing-risk extension of the pinned lmtp 1.5.4 fold adapter.
#
# The shared adapter deliberately remains unchanged because its bytes are part of four
# committed evidence manifests.  This wrapper reproduces its orchestration with D populated,
# and supports the one-fold training-equals-validation convention used by the ordinary row.

# With one learner, SuperLearner's final SL.glm or SL.mean fit is the direct fit below.  Its inner
# folds estimate a library weight that is identically one and do not change the prediction.
# The adapter removes that redundant cross-validation while retaining lmtp's own outer folds,
# competing-risk recursion, fluctuation, and influence-curve implementation.
lmtp_direct_learner <- function(data, y, learners, outcome_type, id, folds) {
  if (!learners %in% c("SL.glm", "SL.mean")) {
    stop("the direct adapter only supports one SL.glm or SL.mean learner")
  }
  if (folds < 2L) stop("the declared learner-fold count must be at least two")
  features <- setdiff(names(data), c(id, y))
  response <- data[[y]]
  binomial_outcome <- outcome_type == "binomial"
  if (identical(learners, "SL.mean")) {
    return(structure(
      list(
        mean = mean(response),
        features = character(0),
        binomial_outcome = binomial_outcome
      ),
      class = "lmtp_direct_learner"
    ))
  }
  x <- cbind("(Intercept)" = 1, as.matrix(data[, features, drop = FALSE]))
  fit <- if (binomial_outcome) {
    glm.fit(x, response, family = binomial())
  } else {
    lm.fit(x, response)
  }
  if (anyNA(fit$coefficients)) stop("the direct SL.glm fit is rank deficient")
  structure(
    list(
      coefficients = fit$coefficients,
      features = features,
      binomial_outcome = binomial_outcome,
      mean = NULL
    ),
    class = "lmtp_direct_learner"
  )
}

predict.lmtp_direct_learner <- function(object, newdata, tol = .Machine$double.eps, ...) {
  prediction <- if (!is.null(object$mean)) {
    rep(object$mean, nrow(newdata))
  } else {
    x <- cbind("(Intercept)" = 1, as.matrix(newdata[, object$features, drop = FALSE]))
    linear <- drop(x %*% object$coefficients)
    if (object$binomial_outcome) plogis(linear) else linear
  }
  if (is.null(tol)) return(prediction)
  pmin(1 - tol, pmax(tol, prediction))
}

lmtp_namespace <- asNamespace("lmtp")
unlockBinding("run_ensemble", lmtp_namespace)
assign("run_ensemble", lmtp_direct_learner, envir = lmtp_namespace)
lockBinding("run_ensemble", lmtp_namespace)

lmtp_competing_tmle_with_folds <- function(
  data,
  shifted,
  trt,
  outcome,
  compete,
  baseline = NULL,
  time_vary = NULL,
  cens = NULL,
  outcome_type = "survival",
  fold_assignment,
  learners_outcome = "SL.glm",
  learners_trt = "SL.glm",
  density_ratios = NULL,
  control = lmtp::lmtp_control(
    .trim = 1,
    .learners_outcome_folds = 2,
    .learners_trt_folds = 2,
    .return_full_fits = TRUE
  )
) {
  if (nrow(data) != length(fold_assignment)) stop("fold assignment has the wrong length")
  variables <- c(unlist(trt), outcome, compete, unlist(time_vary), baseline, cens)
  variables <- unique(variables[!is.na(variables)])
  natural <- data[, variables, drop = FALSE]
  intervention <- shifted[, variables, drop = FALSE]

  labels <- sort(unique(as.integer(fold_assignment)))
  if (!identical(labels, seq.int(0L, length(labels) - 1L))) {
    stop("fold assignment must use contiguous zero-based labels")
  }
  if (length(labels) == 1L) {
    folds <- list(list(training_set = seq_len(nrow(data)), validation_set = seq_len(nrow(data))))
  } else {
    folds <- fold_list(fold_assignment)
  }

  Task <- lmtp_internal("LmtpTask")
  task <- Task$new(
    data = natural,
    shifted = intervention,
    A = trt,
    Y = outcome,
    L = time_vary,
    W = baseline,
    C = cens,
    D = compete,
    k = Inf,
    id = NULL,
    outcome_type = outcome_type,
    bounds = NULL,
    folds = length(folds),
    weights = NULL
  )
  task$folds <- folds
  progress <- function(...) invisible(NULL)

  if (is.null(density_ratios)) {
    density <- lmtp_internal("cf_density_ratios")(task, learners_trt, FALSE, control, progress)
  } else {
    if (nrow(density_ratios) != nrow(data) || ncol(density_ratios) != length(trt)) {
      stop("supplied density ratios have the wrong dimensions")
    }
    if (!all(is.finite(density_ratios)) || any(density_ratios < 0)) {
      stop("supplied density ratios are not finite nonnegative values")
    }
    density <- list(density_ratios = density_ratios, fits = NULL)
  }

  regressions <- lmtp_internal("cf_tmle")(
    task, density$density_ratios, learners_outcome, control, progress
  )
  if (!isTRUE(control$.return_full_fits)) {
    stop("the evidence runner requires .return_full_fits=TRUE")
  }

  initial <- rep(NA_real_, nrow(data))
  for (fold in seq_along(task$folds)) {
    natural_fold <- lmtp_internal("get_folded_data")(task$natural, task$folds, fold)
    shifted_fold <- lmtp_internal("get_folded_data")(task$shifted, task$folds, fold)
    y0 <- task$is_outcome_free(natural_fold$valid, 0)
    c0 <- task$observed(natural_fold$valid, 0)
    d0 <- task$is_competing_risk_free(natural_fold$valid, 0)
    valid <- c0 & y0 & d0
    history <- task$vars$history("L", 2)
    first_treatment <- lmtp_internal("current_trt")(task$vars$A, 1)
    under_shift <- natural_fold$valid[valid, c("..i..lmtp_id", history)]
    under_shift[, first_treatment] <- shifted_fold$valid[valid, first_treatment]
    if (length(regressions$fits[[fold]]) != length(task$vars$A)) {
      stop("the sequential-regression count does not match the treatment nodes")
    }
    predicted <- predict(regressions$fits[[fold]][[1]], under_shift, 1e-05)
    if (anyNA(predicted) || any(predicted < 0) || any(predicted > 1)) {
      stop("the initial plug-in is outside [0, 1]")
    }
    held_out <- task$folds[[fold]]$validation_set[valid]
    initial[held_out] <- predicted
  }
  if (anyNA(initial)) stop("the initial plug-in did not cover every row")

  result <- lmtp_internal("theta_dr")(
    task = task,
    sequential_regressions = list(
      natural = regressions$natural,
      shifted = regressions$shifted
    ),
    density_ratios = density$density_ratios,
    fits_m = regressions$fits,
    fits_r = density$fits,
    shift = "supplied competing-risk panel with exact folds",
    is_sdr = FALSE
  )
  result$initial <- initial
  result$fold_assignment <- as.integer(fold_assignment)
  result
}
