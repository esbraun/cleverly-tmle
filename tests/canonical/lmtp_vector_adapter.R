# A pinned adapter for a vector modified treatment policy in lmtp 1.5.4 at
# f04a2b47f46debc515ce4ae778e05ebfde922c44.
#
# lmtp_policy_tmle (lmtp_policy_adapter.R) writes one shifted column per node, so it cannot
# carry a node of several components.  This adapter builds the LmtpTask with vector nodes
# (A = list(c("F", "D")), or list(c("F1", "D1"), c("F2", "D2"))), writes every shifted component
# from the columns shift__<plan>__<node>__<component>.  lmtp_vector_tmle calls cf_tmle and
# theta_dr with the supplied ratios ratio__<plan>__<node>, .trim = 1 and k = Inf, and never
# calls cf_density_ratios.  A deterministic plan is one copy per unit, so no replication is needed.
#
# lmtp_vector_classifier_tmle is the unsupplied pair.  It builds the same vector task and calls
# cf_density_ratios with mtp = TRUE, so lmtp fits its own stacked classifier with learners_trt.
# The third argument of cf_density_ratios is mtp.  With mtp = FALSE lmtp 1.5.4 multiplies each
# ratio by followed_rule, the indicator that every component kept its natural value
# (R/density_ratios.R:58-61, R/utils.R:24-39), which is the static-regime weight and is zero on
# every row a modified treatment policy moves.  lmtp_tmle itself defaults to mtp = TRUE for this
# reason (R/estimators.R).  tests/unit/test_vector_mtp_design.py reads every vector-path call of
# cf_density_ratios and requires mtp = TRUE.  The pair then calls cf_tmle, the unfluctuated
# plug-in and theta_dr with fits_r = the classifier fits, as lmtp_vector_tmle does with a
# supplied ratio.
#
# The shared adapters are sourced and not edited.  Every categorical component stays a binary
# numeric column.  RM39a2: each lmtp estimation call runs inside `study_stage`
# (tests/canonical/study_harness.R, which every caller sources); every screen stays fatal.

source("/fixture/lmtp_policy_adapter.R")

vector_shifted <- function(frame, plan, trt) {
  # The natural frame with every component of every node replaced by the plan's value.
  shifted <- frame
  for (node in seq_along(trt)) {
    for (column in trt[[node]]) {
      name <- sprintf("shift__%s__%d__%s", plan, node, column)
      if (!(name %in% names(frame))) stop(sprintf("the sample has no column %s", name))
      values <- frame[[name]]
      if (anyNA(values)) stop(sprintf("plan %s has a missing value in %s", plan, name))
      shifted[[column]] <- values
    }
  }
  shifted
}

vector_ratios <- function(frame, plan, nodes) {
  # The n x T per-node ratios cleverly computed.
  columns <- sprintf("ratio__%s__%d", plan, seq_len(nodes))
  missing <- setdiff(columns, names(frame))
  if (length(missing)) stop(sprintf("the sample has no column %s", paste(missing, collapse = ", ")))
  ratios <- as.matrix(frame[columns])
  if (anyNA(ratios)) stop(sprintf("plan %s has a missing ratio", plan))
  ratios
}

vector_task <- function(natural, shifted, trt, outcome, baseline, time_vary, fold_assignment) {
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
  task$folds <- fold_list(as.integer(fold_assignment))
  task
}

lmtp_vector_frames <- function(frame, plan, trt, outcome, baseline, time_vary) {
  # The natural and shifted frames of one plan, with a unit id.
  variables <- unique(c(unlist(trt), outcome, unlist(time_vary), baseline))
  natural <- frame[, variables, drop = FALSE]
  shifted <- vector_shifted(frame, plan, trt)[, variables, drop = FALSE]
  natural$..unit <- seq_len(nrow(frame))
  shifted$..unit <- natural$..unit
  list(natural = natural, shifted = shifted)
}

# The ratio at which a fitted classifier probability reaches lmtp's clip:
# pred / (1 - pmin(pred, 0.999)) is at least this exactly when pred is at least 0.999.
VECTOR_RATIO_SCREEN <- 0.999 / (1 - 0.999)

lmtp_vector_density_ratios <- function(task, learners_trt, control) {
  # lmtp's stacked classifier ratios of a vector modified treatment policy: mtp = TRUE.
  progress <- function(...) invisible(NULL)
  # Resolved outside the stage, so a missing internal is a harness error and not a recorded
  # comparator failure.
  cf_density_ratios <- lmtp_internal("cf_density_ratios")
  density <- study_stage(
    "lmtp density ratios", cf_density_ratios(task, learners_trt, TRUE, control, progress)
  )
  ratios <- density$density_ratios
  if (anyNA(ratios) || any(ratios < 0)) stop("lmtp returned a missing or negative ratio")
  if (any(ratios >= VECTOR_RATIO_SCREEN)) {
    stop("a fitted classifier probability reached lmtp's clip at 0.999")
  }
  density
}

lmtp_vector_from_ratios <- function(task, trt, ratios, fits_r, learners_outcome, control, shift) {
  # cf_tmle, the unfluctuated plug-in and theta_dr at the given per-node ratios.
  progress <- function(...) invisible(NULL)
  cf_tmle <- lmtp_internal("cf_tmle")
  theta_dr <- lmtp_internal("theta_dr")
  regressions <- study_stage(
    "lmtp tmle", cf_tmle(task, ratios, learners_outcome, control, progress)
  )
  # No prediction may reach lmtp's bound of 1e-5, or cleverly's [0, 1] clip and lmtp's bound
  # would be two different regularisations of one regression.
  predictions <- cbind(regressions$natural[, seq_along(trt)], regressions$shifted[, seq_along(trt)])
  if (any(predictions <= 1e-5 | predictions >= 1 - 1e-5, na.rm = TRUE)) {
    stop("a sequential regression reached lmtp's prediction bound")
  }

  # The untargeted plug-in: the first regression at the shifted first node, averaged.
  history <- task$vars$history("L", 2)
  first_treatment <- lmtp_internal("current_trt")(task$vars$A, 1)
  if (!setequal(first_treatment, trt[[1]])) {
    stop("lmtp's first treatment node is not the declared vector node")
  }
  under_shift <- task$natural[, c("..i..lmtp_id", history)]
  under_shift[, first_treatment] <- task$shifted[, first_treatment]
  initial <- study_stage("lmtp estimate", predict(regressions$fits[[1]][[1]], under_shift, 1e-05))

  result <- study_stage("lmtp estimate", theta_dr(
    task = task,
    sequential_regressions = list(natural = regressions$natural, shifted = regressions$shifted),
    density_ratios = ratios,
    fits_m = regressions$fits,
    fits_r = fits_r,
    shift = shift,
    is_sdr = FALSE
  ))
  list(estimate = result$estimate@x, ic = result$estimate@eif, initial = mean(initial))
}

lmtp_vector_tmle <- function(
  frame,
  plan,
  trt,
  outcome,
  baseline,
  time_vary,
  learners_outcome = "SL.glm",
  control = lmtp::lmtp_control(.trim = 1, .learners_outcome_folds = 2, .return_full_fits = TRUE)
) {
  # frame: one row per unit, the natural columns plus the study's shift__ and ratio__ columns.
  # trt: one character vector per node.  Returns the estimate, the unit-level influence curve
  # and the untargeted plug-in.
  invisible(list(frame, plan, trt, outcome, baseline, time_vary, learners_outcome, control))
  if (!is.list(trt)) stop("trt must be a list with one character vector per node")
  frames <- lmtp_vector_frames(frame, plan, trt, outcome, baseline, time_vary)
  ratios <- vector_ratios(frame, plan, length(trt))
  screen_policy_ratios(ratios, ratios)
  task <- vector_task(
    frames$natural, frames$shifted, trt, outcome, baseline, time_vary, frame$fold
  )
  lmtp_vector_from_ratios(
    task, trt, ratios, NULL, learners_outcome, control, "supplied vector policy"
  )
}

lmtp_vector_classifier_tmle <- function(
  frame,
  plan,
  trt,
  outcome,
  baseline,
  time_vary,
  learners_outcome = "SL.glm",
  learners_trt,
  control = lmtp::lmtp_control(
    .trim = 1,
    .learners_outcome_folds = 2,
    .learners_trt_folds = 2,
    .return_full_fits = TRUE
  )
) {
  # The unsupplied pair: lmtp fits its own stacked classifier on the natural and shifted
  # frames of the vector task, with mtp = TRUE.  Returns the estimate, the influence curve, the
  # plug-in and lmtp's per-node ratios.
  invisible(list(frame, plan, trt, outcome, baseline, time_vary, learners_outcome, learners_trt))
  if (!is.list(trt)) stop("trt must be a list with one character vector per node")
  frames <- lmtp_vector_frames(frame, plan, trt, outcome, baseline, time_vary)
  task <- vector_task(
    frames$natural, frames$shifted, trt, outcome, baseline, time_vary, frame$fold
  )
  density <- lmtp_vector_density_ratios(task, learners_trt, control)
  out <- lmtp_vector_from_ratios(
    task, trt, density$density_ratios, density$fits, learners_outcome, control,
    "vector policy, lmtp's stacked classifier"
  )
  out$ratios <- density$density_ratios
  out
}
