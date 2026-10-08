# A pinned adapter for a held point treatment over survival nodes in lmtp 1.5.4 at
# f04a2b47f46debc515ce4ae778e05ebfde922c44: K event nodes, K censoring nodes, and a policy or a
# modified treatment policy at the baseline decision only.
#
# The treatment is passed as K copied columns A1..AK, each present while the unit is in the
# study, with the policy's shifted value at A1 and the observed value at every later column.
# That is Díaz et al. (2023), Definition 1, with the identity policy d_t(a_t, h_t) = a_t for
# t >= 2, which is cleverly's held design: each later regression is evaluated at the natural
# history, so the shift enters once.  lmtp's own form with trt of length one shifts the one
# treatment column at every node's prediction, which applies a policy that is not idempotent
# (minus one, or a stochastic draw) once per node; that is a different estimand.
#
# The caller supplies the shifted value of each copy of a unit and the per-node density ratio,
# both computed from the law's own mechanism, so neither side estimates a ratio the other does
# not see.  Like lmtp_policy_adapter.R, this builds an LmtpTask with the shifted frame and calls
# cf_tmle and theta_dr directly, never cf_density_ratios; a policy with probabilities that are
# multiples of 1 / copies is realised by proportional replication, and the unit-level influence
# curve is the mean of its copies'.  The shared adapters are sourced and not edited.

source("/fixture/lmtp_policy_adapter.R")

lmtp_held_survival_tmle <- function(
  data,
  draws,
  ratios,
  trt,
  outcome,
  cens,
  baseline,
  copies,
  outcome_type,
  learners_outcome = "SL.glm"
) {
  # data: one row per unit, natural columns.  draws: an n x copies matrix of the shifted
  # treatment of each copy.  ratios: n x K per-node ratios on the unit's own row.  Returns the
  # estimate, the unit-level influence curve and the untargeted plug-in.
  n <- nrow(data)
  if (!identical(dim(draws), c(n, as.integer(copies)))) stop("the draws are not n x copies")
  if (ncol(ratios) != length(outcome)) stop("one ratio column per event node is required")
  first <- ratios[, 1]
  spread <- stats::sd(first) / sqrt(n)
  if (!is.finite(spread) || abs(mean(first) - 1) > 5 * spread) {
    stop(sprintf("the first-node ratio averages %.4f, which is not one", mean(first)))
  }
  if (!all(is.finite(ratios)) || any(ratios < 0)) stop("a supplied ratio is negative or not finite")

  # The copied treatment columns: present while the unit is in the study before the node.
  nodes <- length(outcome)
  treatment <- sprintf("%s%d", trt, seq_len(nodes))
  for (t in seq_len(nodes)) {
    present <- if (t == 1) rep(TRUE, n) else !is.na(data[[cens[[t]]]])
    data[[treatment[[t]]]] <- ifelse(present, data[[trt]], NA)
  }
  unit <- rep(seq_len(n), each = copies)
  natural <- data[unit, , drop = FALSE]
  natural$..unit <- unit
  shifted <- natural
  # The policy acts at the decision; every later node is the identity.
  shifted[[treatment[[1]]]] <- as.vector(t(draws))
  replicated <- ratios[unit, , drop = FALSE]
  trt <- treatment
  variables <- unique(c(trt, outcome, cens, baseline, "..unit"))
  natural <- natural[, variables, drop = FALSE]
  shifted <- shifted[, variables, drop = FALSE]

  Task <- lmtp_internal("LmtpTask")
  task <- Task$new(
    data = natural,
    shifted = shifted,
    A = trt,
    Y = outcome,
    L = NULL,
    W = baseline,
    C = cens,
    D = NULL,
    k = Inf,
    id = "..unit",
    outcome_type = outcome_type,
    bounds = NULL,
    folds = 1,
    weights = NULL
  )
  task$folds <- fold_list(rep(0L, n * copies))
  progress <- function(...) invisible(NULL)
  control <- lmtp::lmtp_control(
    .trim = 1, .learners_outcome_folds = 2, .return_full_fits = TRUE
  )
  regressions <- lmtp_internal("cf_tmle")(task, replicated, learners_outcome, control, progress)

  # The untargeted plug-in: the first regression at the copy's shifted treatment, averaged.
  history <- task$vars$history("L", 2)
  first_treatment <- lmtp_internal("current_trt")(task$vars$A, 1)
  at_start <- task$observed(task$natural, 0) & task$is_outcome_free(task$natural, 0)
  under_shift <- task$natural[at_start, c("..i..lmtp_id", history)]
  under_shift[, first_treatment] <- task$shifted[at_start, first_treatment]
  initial <- rep(NA_real_, nrow(task$natural))
  initial[at_start] <- predict(regressions$fits[[1]][[1]], under_shift, 1e-05)
  if (anyNA(initial)) stop("the initial plug-in did not cover every copy")

  result <- lmtp_internal("theta_dr")(
    task = task,
    sequential_regressions = list(natural = regressions$natural, shifted = regressions$shifted),
    density_ratios = replicated,
    fits_m = regressions$fits,
    fits_r = NULL,
    shift = "held point treatment, supplied shifted values and ratios",
    is_sdr = FALSE
  )
  list(
    estimate = result$estimate@x,
    ic = as.numeric(tapply(result$estimate@eif, unit, mean)),
    initial = mean(as.numeric(tapply(initial, unit, mean)))
  )
}
