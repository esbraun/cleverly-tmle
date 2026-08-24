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
  lmtp_internal("theta_dr")(
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
}
