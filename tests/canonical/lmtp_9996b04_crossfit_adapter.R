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
# is sourced for lmtp_internal and fold_list, and is not edited; the ratio screen is the
# shipped one with its correlation replaced by an exact moment check (check_horizon_density_ratios,
# below).

horizon_ratio_moments <- function(density_ratios, unscreened = integer(0), bound = 5) {
  # Each supplied per-node ratio r_t = q_t(A_t | H_t) 1{C_t = 1} / (g_t(A_t | H_t) c_t) has
  # conditional mean one given the history: E[r_t | H_t] = sum_a q_t(a | H_t) = 1.  So over the
  # units still on the path at t (every earlier ratio positive, an H_t-measurable set) its mean
  # is one within sampling error, at every node, whatever the plan.  A wrong arm's g, a missing
  # censoring factor or a node's g read at another node moves some node's mean by many standard
  # errors (tests/unit/test_x59_long_horizon_reference.py holds the identity and those
  # controls).  `unscreened` names the nodes a declared mutation changes on purpose.
  prior <- rep(TRUE, nrow(density_ratios))
  for (t in seq_len(ncol(density_ratios))) {
    values <- density_ratios[prior, t]
    if (!(t %in% unscreened) && length(values) >= 2L) {
      spread <- stats::sd(values) / sqrt(length(values))
      distance <- abs(mean(values) - 1)
      if (!is.finite(spread) || distance > bound * spread) {
        stop(sprintf(
          "the supplied node-%d density ratio averages %.4f over the %d units on the path, %.1f standard errors from one; a correct ratio averages one there",
          t, mean(values), length(values), if (is.finite(spread) && spread > 0) distance / spread else Inf
        ))
      }
    }
    prior <- prior & density_ratios[, t] > 0
  }
  invisible(TRUE)
}

check_horizon_density_ratios <- function(density_ratios, estimated, unscreened = integer(0)) {
  # The shipped screen (check_supplied_density_ratios in lmtp_crossfit_adapter.R) keeps its
  # shape and zero-pattern checks.  Its correlation with lmtp's own estimate is dropped: at
  # three to five nodes of a dynamic plan, lmtp's stacked classifier with main-effects GLM
  # (learners_trt = "SL.glm", arms entered as numbers) cannot represent 1{a = d(H)} / g(a | H),
  # so its estimate is not a reference.  Emulated on S2 `respond` at K = 5, its per-node rank
  # correlation with the exact ratio on the followers is 1.00 at node 1 and 0.02 to 0.25 at
  # nodes 2 to 5, and the cumulative products' is -0.01 over the 2262 units both keep (row 50's R
  # read 0.0090); over all units it is 0.9999 only because the zeros tie.  In its place every
  # node's ratio is held to its exact mean (horizon_ratio_moments).
  if (!identical(dim(density_ratios), dim(estimated))) {
    stop(sprintf(
      "supplied density ratios are %s; lmtp's are %s",
      paste(dim(density_ratios), collapse = " x "),
      paste(dim(estimated), collapse = " x ")
    ))
  }
  disagreeing <- sum(xor(density_ratios == 0, estimated == 0))
  if (disagreeing > 0) {
    stop(sprintf(
      "the supplied density ratios follow a different path from lmtp's: %d of %d cells disagree on whether the unit is still on the regimen at that node",
      disagreeing, length(estimated)
    ))
  }
  if (!all(is.finite(density_ratios)) || any(density_ratios < 0)) {
    stop("the supplied density ratios contain a negative or non-finite value")
  }
  horizon_ratio_moments(density_ratios, unscreened)
}

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
  control = lmtp::lmtp_control(.trim = 1, .learners_outcome_folds = 2, .learners_trt_folds = 2),
  unscreened = integer(0)
) {
  # RM39a2 (M7): every argument is evaluated here, outside any stage.
  invisible(list(
    data, shifted, trt, outcome, baseline, time_vary, cens, id, outcome_type,
    fold_assignment, learners_outcome, learners_trt, density_ratios, control, unscreened
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
    check_horizon_density_ratios(density_ratios, density$density_ratios, unscreened)
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
