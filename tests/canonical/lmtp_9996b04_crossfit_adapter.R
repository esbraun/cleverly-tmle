# The supplied-fold lmtp_tmle adapter of lmtp_crossfit_adapter.R, for lmtp 9996b04 (1.5.5).
#
# At 9996b04, R/tmle.R stores summary(fit, time, fold) for each node regression and returns
# rbindlist(fits), a data.table, whatever .return_full_fits says (X39-plan-review.md, finding 3).
# The shipped adapter predicts the first-node regression from regressions$fits to publish R's
# untargeted plug-in, so it cannot run on this commit.  This adapter is that function with the
# plug-in block removed: it never reads regressions$fits, it checks only that the object has the
# 9996b04 structure (so a commit that changed it again stops here), and it returns initial = NA.
# The long-horizon studies read the untargeted value from cleverly's own sequential regression
# (tests/studies/long_horizon_reference.py, `plugin`) and from G-C, so no published reading
# needs R's.
#
# Everything else is the shipped adapter's: the same task, the supplied fold assignment, the
# supplied ratios behind the same screen, and lmtp's own cf_tmle and theta_dr.  The shipped file
# is sourced for lmtp_internal, fold_list and check_supplied_density_ratios, and is not edited.

source("/fixture/lmtp_crossfit_adapter.R")

lmtp_9996b04_fits_guard <- function(fits) {
  # The 9996b04 shape of regressions$fits: one data.table of per-node, per-fold summaries.
  if (!data.table::is.data.table(fits) || !nrow(fits)) {
    stop("cf_tmle returned fits that are not the 9996b04 data.table of node summaries")
  }
  invisible(TRUE)
}

lmtp_9996b04_tmle_with_folds <- function(
  data,
  shifted,
  trt,
  outcome,
  baseline = NULL,
  time_vary = NULL,
  cens = NULL,
  id = NULL,
  outcome_type = "binomial",
  fold_assignment,
  learners_outcome = "SL.glm",
  learners_trt = "SL.glm",
  density_ratios = NULL,
  control = lmtp::lmtp_control(.trim = 1, .learners_outcome_folds = 2, .learners_trt_folds = 2)
) {
  # RM39a2 (M7): every argument is evaluated here, outside any stage.
  invisible(list(
    data, shifted, trt, outcome, baseline, time_vary, cens, id, outcome_type,
    fold_assignment, learners_outcome, learners_trt, density_ratios, control
  ))
  if (nrow(data) != length(fold_assignment)) stop("fold assignment has the wrong length")
  if (!is.null(id) && (!is.character(id) || length(id) != 1L || !id %in% names(data))) {
    stop("id must be NULL or the name of one unit identifier column")
  }
  variables <- unique(c(unlist(trt), outcome, unlist(time_vary), baseline, cens, id))
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
    id = id,
    outcome_type = outcome_type,
    bounds = NULL,
    folds = length(unique(fold_assignment)),
    weights = NULL
  )
  task$folds <- fold_list(fold_assignment)
  progress <- function(...) invisible(NULL)
  cf_density_ratios <- lmtp_internal("cf_density_ratios")
  cf_tmle <- lmtp_internal("cf_tmle")
  theta_dr <- lmtp_internal("theta_dr")

  density <- study_stage("lmtp density ratios", cf_density_ratios(
    task, learners_trt, FALSE, control, progress
  ))
  if (!is.null(density_ratios)) {
    check_supplied_density_ratios(density_ratios, density$density_ratios)
    density$density_ratios <- density_ratios
  }
  regressions <- study_stage("lmtp tmle", cf_tmle(
    task, density$density_ratios, learners_outcome, control, progress
  ))
  lmtp_9996b04_fits_guard(regressions$fits)
  predictions <- cbind(regressions$natural[, seq_along(trt)], regressions$shifted[, seq_along(trt)])
  result <- study_stage("lmtp estimate", theta_dr(
    task = task,
    sequential_regressions = list(natural = regressions$natural, shifted = regressions$shifted),
    density_ratios = density$density_ratios,
    fits_m = regressions$fits,
    fits_r = density$fits,
    shift = "supplied shifted data with exact folds",
    is_sdr = FALSE
  ))
  list(
    estimate = result$estimate@x,
    ic = as.numeric(result$estimate@eif),
    # Not read from lmtp at 9996b04: see the header.
    initial = NA_real_,
    predictions = predictions,
    fold_assignment = as.integer(fold_assignment)
  )
}
