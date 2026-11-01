# The known-policy lmtp_tmle adapter of lmtp_policy_adapter.R, for lmtp 9996b04 (1.5.5).
#
# The shipped adapter realises a known stochastic policy by proportional replication (its header
# states why the replicated fit is the integrated estimator), and it publishes R's untargeted
# plug-in by predicting regressions$fits[[1]][[1]].  At 9996b04 that object is a data.table of
# node summaries (lmtp_9996b04_crossfit_adapter.R), so this adapter is the shipped function
# without that prediction: it never reads regressions$fits beyond a structure guard, and it
# returns initial = NA.  The shipped file is sourced for screen_policy_ratios and, through it,
# lmtp_crossfit_adapter.R; neither is edited.
#
# Unlike the shipped adapter it takes the supplied fold assignment at any fold count, because
# the cross-fitted long-horizon scenarios fit a policy plan at five folds.

source("/fixture/lmtp_9996b04_crossfit_adapter.R")
source("/fixture/lmtp_policy_adapter.R")

lmtp_9996b04_policy_tmle <- function(
  data,
  draws,
  ratios,
  trt,
  outcome,
  baseline,
  time_vary,
  cens = NULL,
  fold_assignment,
  copies,
  learners_outcome = "SL.glm",
  control = lmtp::lmtp_control(.trim = 1, .learners_outcome_folds = 2)
) {
  # data: one row per unit, natural columns.  draws: one n x copies matrix per node, the
  # shifted arm of each copy.  ratios: n x T per-node ratios at the natural arm.
  # RM39a2 (M7): every argument is evaluated here, outside any stage.
  invisible(list(
    data, draws, ratios, trt, outcome, baseline, time_vary, cens, fold_assignment, copies,
    learners_outcome, control
  ))
  n <- nrow(data)
  if (length(draws) != length(trt)) stop("one draw matrix per treatment node is required")
  for (node in seq_along(draws)) {
    if (!identical(dim(draws[[node]]), c(n, as.integer(copies)))) {
      stop(sprintf("node %d draws are not %d x %d", node, n, copies))
    }
  }
  # The plan's mass at the observed arm, built from the draws alone: the draws give each arm
  # its quarters of the copies, so some copy takes the observed arm exactly where the plan
  # gives that arm mass.  A unit keeps it only while observed and uncensored at the node.  The
  # supplied ratios are computed apart, so the screen compares two constructions of one zero
  # pattern (tests/unit/test_x59_long_horizon_reference.py holds the same construction).
  support <- vapply(seq_along(trt), function(node) {
    observed <- data[[trt[[node]]]]
    kept <- !is.na(observed)
    if (!is.null(cens)) kept <- kept & !is.na(data[[cens[[node]]]]) & data[[cens[[node]]]] == 1
    taken <- rowSums(draws[[node]] == ifelse(is.na(observed), -1, observed)) > 0
    as.numeric(kept & taken)
  }, numeric(n))
  screen_policy_ratios(ratios, matrix(support, nrow = n))

  unit <- rep(seq_len(n), each = copies)
  natural <- data[unit, , drop = FALSE]
  natural$..unit <- unit
  shifted <- natural
  for (node in seq_along(trt)) {
    shifted[[trt[[node]]]] <- as.vector(t(draws[[node]]))
  }
  replicated <- ratios[unit, , drop = FALSE]
  variables <- unique(c(unlist(trt), outcome, unlist(time_vary), baseline, cens, "..unit"))
  natural <- natural[, variables, drop = FALSE]
  shifted <- shifted[, variables, drop = FALSE]
  rownames(natural) <- NULL
  rownames(shifted) <- NULL

  Task <- lmtp_internal("LmtpTask")
  task <- Task$new(
    data = natural,
    shifted = shifted,
    A = trt,
    Y = outcome,
    L = time_vary,
    W = baseline,
    C = cens,
    D = NULL,
    k = Inf,
    id = "..unit",
    outcome_type = "binomial",
    bounds = NULL,
    folds = length(unique(fold_assignment)),
    weights = NULL
  )
  task$folds <- fold_list(rep(as.integer(fold_assignment), each = copies))
  progress <- function(...) invisible(NULL)
  cf_tmle <- lmtp_internal("cf_tmle")
  theta_dr <- lmtp_internal("theta_dr")
  regressions <- study_stage(
    "lmtp tmle", cf_tmle(task, replicated, learners_outcome, control, progress)
  )
  lmtp_9996b04_fits_guard(regressions$fits)
  predictions <- cbind(regressions$natural[, seq_along(trt)], regressions$shifted[, seq_along(trt)])
  result <- study_stage("lmtp estimate", theta_dr(
    task = task,
    sequential_regressions = list(natural = regressions$natural, shifted = regressions$shifted),
    density_ratios = replicated,
    fits_m = regressions$fits,
    fits_r = NULL,
    shift = "known policy by proportional replication",
    is_sdr = FALSE
  ))
  unit_ic <- as.numeric(tapply(result$estimate@eif, unit, mean))
  if (anyNA(unit_ic)) stop("the influence curve has a gap")
  list(
    estimate = result$estimate@x,
    ic = unit_ic,
    # Not read from lmtp at 9996b04: see lmtp_9996b04_crossfit_adapter.R.
    initial = NA_real_,
    predictions = predictions
  )
}
