# A pinned adapter for a modified treatment policy in lmtp 1.5.4 at
# f04a2b47f46debc515ce4ae778e05ebfde922c44, with the shifted values and the density ratios
# supplied by the caller.
#
# The longitudinal-mtp study hands lmtp the per-node ratio cleverly computed and the value each
# policy assigns, both written beside the replicate data.  So neither side estimates a ratio the
# other does not see.  This adapter reads those columns and calls lmtp_policy_tmle from
# lmtp_policy_adapter.R, which builds an LmtpTask with the shifted frame and calls cf_tmle and
# theta_dr directly, with .trim = 1 and k = Inf, and never cf_density_ratios.  A deterministic
# policy is one copy per unit.  A randomized policy with known branch probabilities that are
# multiples of 1 / copies is realised by proportional replication, as lmtp_policy_adapter.R
# states.  Neither shared adapter is edited; both are hashed by the studies that ran them.

source("/fixture/lmtp_policy_adapter.R")

mtp_draws <- function(frame, plan, node, copies, as_label) {
  # The n x copies matrix of values each copy of a unit takes at `node`.
  columns <- sprintf("shift__%s__%d__%d", plan, node, seq_len(copies))
  missing <- setdiff(columns, names(frame))
  if (length(missing)) stop(sprintf("the sample has no column %s", paste(missing, collapse = ", ")))
  values <- as.matrix(frame[columns])
  if (anyNA(values)) stop(sprintf("plan %s has a missing shifted value at node %d", plan, node))
  if (as_label) {
    values <- matrix(as.character(values), nrow = nrow(values))
  }
  values
}

mtp_ratios <- function(frame, plan, nodes) {
  # The n x T per-node ratios cleverly computed, node_ratio of its fit.
  columns <- sprintf("ratio__%s__%d", plan, seq_len(nodes))
  missing <- setdiff(columns, names(frame))
  if (length(missing)) stop(sprintf("the sample has no column %s", paste(missing, collapse = ", ")))
  ratios <- as.matrix(frame[columns])
  if (anyNA(ratios)) stop(sprintf("plan %s has a missing ratio", plan))
  ratios
}

lmtp_mtp_tmle <- function(
  frame,
  plan,
  trt,
  outcome,
  baseline,
  time_vary,
  copies = 1L,
  as_label = FALSE,
  learners_outcome = "SL.glm"
) {
  # frame: one row per unit, the natural columns plus the study's shift__ and ratio__ columns.
  # as_label: code the treatment columns as character, so SL.glm reads one dummy per level and
  # spans the drop-first indicators cleverly's design holds.
  natural <- frame[c(baseline, unlist(time_vary), trt, outcome)]
  if (as_label) {
    for (column in trt) natural[[column]] <- as.character(natural[[column]])
  }
  draws <- lapply(seq_along(trt), function(node) mtp_draws(frame, plan, node, copies, as_label))
  ratios <- mtp_ratios(frame, plan, length(trt))
  lmtp_policy_tmle(
    natural,
    draws,
    ratios,
    ratios,
    trt = trt,
    outcome = outcome,
    baseline = baseline,
    time_vary = time_vary,
    fold_assignment = frame$fold,
    copies = copies,
    learners_outcome = learners_outcome,
    control = lmtp::lmtp_control(.trim = 1, .learners_outcome_folds = 2, .return_full_fits = TRUE)
  )
}
