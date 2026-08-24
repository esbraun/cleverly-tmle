# A pinned adapter for lmtp 1.5.4 at f04a2b47f46debc515ce4ae778e05ebfde922c44.
#
# The public function accepts only a fold count.  A paired study needs the exact realized
# Python assignment, so this function reproduces lmtp_tmle's short orchestration layer and
# replaces only task$folds.  Estimation remains in lmtp's cf_density_ratios, cf_tmle, and
# theta_dr functions.  Every internal locator is checked before a study starts.

lmtp_internal <- function(name) {
  value <- getFromNamespace(name, "lmtp")
  if (is.null(value)) stop(sprintf("lmtp internal %s is unavailable", name))
  value
}

fold_list <- function(assignment) {
  labels <- sort(unique(as.integer(assignment)))
  if (!identical(labels, seq.int(0L, length(labels) - 1L))) {
    stop("fold assignment must use contiguous zero-based labels")
  }
  lapply(labels, function(label) {
    validation <- which(assignment == label)
    training <- which(assignment != label)
    if (!length(validation) || !length(training)) stop("every fold needs training and validation rows")
    list(training_set = training, validation_set = validation)
  })
}

lmtp_tmle_with_folds <- function(
  data,
  shifted,
  trt,
  outcome,
  baseline = NULL,
  time_vary = NULL,
  cens = NULL,
  outcome_type = "binomial",
  fold_assignment,
  learners_outcome = "SL.glm",
  learners_trt = "SL.glm",
  control = lmtp::lmtp_control(.trim = 1, .learners_outcome_folds = 5, .learners_trt_folds = 5)
) {
  if (nrow(data) != length(fold_assignment)) stop("fold assignment has the wrong length")
  variables <- c(unlist(trt), outcome, unlist(time_vary), baseline, cens)
  natural <- data[, variables, drop = FALSE]
  intervention <- shifted[, variables, drop = FALSE]

  Task <- lmtp_internal("LmtpTask")
  task <- Task$new(
    data = natural,
    shifted = intervention,
    A = trt,
    Y = outcome,
    L = time_vary,
    W = baseline,
    C = cens,
    D = NULL,
    k = Inf,
    id = NULL,
    outcome_type = outcome_type,
    bounds = NULL,
    folds = length(unique(fold_assignment)),
    weights = NULL
  )
  task$folds <- fold_list(fold_assignment)
  progress <- function(...) invisible(NULL)

  density <- lmtp_internal("cf_density_ratios")(
    task, learners_trt, FALSE, control, progress
  )
  regressions <- lmtp_internal("cf_tmle")(
    task, density$density_ratios, learners_outcome, control, progress
  )
  if (!isTRUE(control$.return_full_fits)) {
    stop("the evidence runner requires .return_full_fits=TRUE for its targeting witness")
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
    predicted <- predict(regressions$fits[[fold]][[1]], under_shift, 1e-05)
    held_out <- task$folds[[fold]]$validation_set[valid]
    initial[held_out] <- predicted
  }
  if (anyNA(initial)) stop("the initial plug-in did not cover every held-out row")

  result <- lmtp_internal("theta_dr")(
    task = task,
    sequential_regressions = list(
      natural = regressions$natural,
      shifted = regressions$shifted
    ),
    density_ratios = density$density_ratios,
    fits_m = regressions$fits,
    fits_r = density$fits,
    shift = "supplied shifted data with exact folds",
    is_sdr = FALSE
  )
  result$initial <- initial
  result$fold_assignment <- as.integer(fold_assignment)
  result
}
