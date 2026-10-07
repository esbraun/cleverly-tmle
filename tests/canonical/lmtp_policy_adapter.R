# A pinned adapter for a known stochastic policy in lmtp 1.5.4 at
# f04a2b47f46debc515ce4ae778e05ebfde922c44, by proportional replication.
#
# lmtp's public interface takes one shifted value per unit and node.  A known policy
# q_t(a | H_t) draws one of several arms.  When every policy probability is a multiple of
# 1 / copies, the policy is realised exactly by replicating each unit `copies` times: copy c
# takes its shifted arm at node t from row c of that node's allocation table, so a unit's
# copies hold the arms in proportion copies * q_t(. | H_t), with H_t holding the natural
# earlier arms.  Three facts of lmtp's estimate_tmle (R/tmle.R) make the replicated fit the
# integrated estimator:
#
# * node t predicts the shifted value by replacing only the current arm, so the joint law
#   of the copies across nodes never enters;
# * the regression and the fluctuation are fitted on copies that share their design row,
#   and both scores are linear in the outcome, so node t - 1 regresses the copy mean
#   sum_j q_t(j | H_t) m*_t(j, H_t);
# * theta is the mean of the shifted first-node prediction, and the unit mean of the
#   per-copy eif (R/eif.R) is the integrated influence curve.
#
# The shared adapter is sourced for lmtp_internal and fold_list only.  Its bytes are
# unchanged, because its sha256 is in the reference hashes of registered studies.  This
# adapter never calls cf_density_ratios: the supplied ratio q / g at the natural arm differs
# from lmtp's own followed-rule ratio on purpose, so that function's screen does not apply.
# Its own screens run on one row per unit, because replicated copies understate a spread by
# sqrt(copies).

source("/fixture/lmtp_crossfit_adapter.R")

screen_policy_ratios <- function(ratios, density_at_observed) {
  # (i) A supplied ratio is zero exactly where the plan gives the observed arm no mass.
  if (!identical(dim(ratios), dim(density_at_observed))) {
    stop("the supplied policy ratios and the policy densities differ in shape")
  }
  disagreeing <- sum(xor(ratios == 0, density_at_observed == 0))
  if (disagreeing > 0) {
    stop(sprintf(
      "%d of %d supplied policy ratios are zero where the plan has mass, or the reverse",
      disagreeing, length(ratios)
    ))
  }
  if (!all(is.finite(ratios)) || any(ratios < 0)) {
    stop("the supplied policy ratios contain a negative or non-finite value")
  }
  # (ii) The first node has no event upstream of it, so its ratio averages one.
  first <- ratios[, 1]
  spread <- stats::sd(first) / sqrt(length(first))
  distance <- abs(mean(first) - 1)
  if (!is.finite(spread) || distance > 5 * spread) {
    stop(sprintf(
      "the first-node policy ratio averages %.4f, %.1f unit-level standard errors from one",
      mean(first), if (spread > 0) distance / spread else Inf
    ))
  }
  invisible(TRUE)
}

allocate_copies <- function(probabilities, arms, copies) {
  # One row of a policy table, as the arm each of `copies` copies takes.  Refuses a row that
  # is not a multiple of 1 / copies, because then no replication realises it exactly.
  counts <- probabilities * copies
  if (any(abs(counts - round(counts)) > 1e-12) || abs(sum(counts) - copies) > 1e-12) {
    stop(sprintf(
      "policy row (%s) is not a multiple of 1/%d",
      paste(probabilities, collapse = ", "), copies
    ))
  }
  rep(arms, times = as.integer(round(counts)))
}

lmtp_policy_tmle <- function(
  data,
  draws,
  ratios,
  density_at_observed,
  trt,
  outcome,
  baseline,
  time_vary,
  fold_assignment,
  copies,
  learners_outcome = "SL.glm",
  control = lmtp::lmtp_control(
    .trim = 1,
    .learners_outcome_folds = 2,
    .return_full_fits = TRUE
  )
) {
  # data: one row per unit, natural columns.  draws: one n x copies character matrix per
  # node, the shifted arm of each copy.  ratios and density_at_observed: n x T, per node,
  # at the natural arm.  Returns the estimate, the unit-level influence curve and the
  # untargeted plug-in.
  n <- nrow(data)
  if (length(draws) != length(trt)) stop("one draw matrix per treatment node is required")
  for (node in seq_along(draws)) {
    if (!identical(dim(draws[[node]]), c(n, as.integer(copies)))) {
      stop(sprintf("node %d draws are not %d x %d", node, n, copies))
    }
  }
  screen_policy_ratios(ratios, density_at_observed)

  unit <- rep(seq_len(n), each = copies)
  natural <- data[unit, , drop = FALSE]
  natural$..unit <- unit
  shifted <- natural
  for (node in seq_along(trt)) {
    shifted[[trt[[node]]]] <- as.vector(t(draws[[node]]))
  }
  replicated <- ratios[unit, , drop = FALSE]
  # (iii) Every copy of a unit carries that unit's ratio row.
  for (node in seq_len(ncol(replicated))) {
    if (any(tapply(replicated[, node], unit, function(x) length(unique(x))) != 1L)) {
      stop(sprintf("the copies of a unit carry different ratios at node %d", node))
    }
  }
  variables <- unique(c(unlist(trt), outcome, unlist(time_vary), baseline, "..unit"))
  natural <- natural[, variables, drop = FALSE]
  shifted <- shifted[, variables, drop = FALSE]

  Task <- lmtp_internal("LmtpTask")
  task <- Task$new(
    data = natural,
    shifted = shifted,
    A = trt,
    Y = outcome,
    L = time_vary,
    W = baseline,
    C = NULL,
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
  regressions <- lmtp_internal("cf_tmle")(task, replicated, learners_outcome, control, progress)

  # No prediction may reach lmtp's bound of 1e-5, or cleverly's [0, 1] clip and lmtp's bound
  # would be two different regularisations of one regression.
  predictions <- cbind(regressions$natural[, seq_along(trt)], regressions$shifted[, seq_along(trt)])
  if (any(predictions <= 1e-5 | predictions >= 1 - 1e-5, na.rm = TRUE)) {
    stop("a sequential regression reached lmtp's prediction bound")
  }

  # The untargeted plug-in: the first regression at the copy's shifted first arm, averaged.
  history <- task$vars$history("L", 2)
  first_treatment <- lmtp_internal("current_trt")(task$vars$A, 1)
  under_shift <- task$natural[, c("..i..lmtp_id", history)]
  under_shift[, first_treatment] <- task$shifted[, first_treatment]
  initial <- predict(regressions$fits[[1]][[1]], under_shift, 1e-05)

  result <- lmtp_internal("theta_dr")(
    task = task,
    sequential_regressions = list(natural = regressions$natural, shifted = regressions$shifted),
    density_ratios = replicated,
    fits_m = regressions$fits,
    fits_r = NULL,
    shift = "known policy by proportional replication",
    is_sdr = FALSE
  )
  copy_eif <- result$estimate@eif
  unit_ic <- as.numeric(tapply(copy_eif, unit, mean))
  list(
    estimate = result$estimate@x,
    ic = unit_ic,
    initial = mean(as.numeric(tapply(initial, unit, mean)))
  )
}
