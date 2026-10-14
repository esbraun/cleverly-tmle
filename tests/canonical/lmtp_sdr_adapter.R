# A pinned adapter for the sequentially doubly robust estimator of lmtp 1.5.4 at
# f04a2b47f46debc515ce4ae778e05ebfde922c44, with the shifted values and the per-node ratios
# supplied by the caller.
#
# The longitudinal-sdr study writes, beside each replicate, the per-node ratio cleverly divides
# by (columns ratio__<plan>__<node>) and the value each copy of a unit takes at each node
# (columns shift__<plan>__<node>__<copy>).  This adapter builds an LmtpTask from them, replaces
# task$folds with the Python assignment, and calls lmtp's own cf_sdr and theta_dr(is_sdr =
# TRUE).  It never calls cf_density_ratios, so no side estimates a ratio the other does not see,
# and it trims nothing (.trim = 1, k = Inf).
#
# A known policy is realised by proportional replication, as lmtp_policy_adapter.R states:
# copy c of a unit takes its shifted arm from column c.  The SDR transform (R/eif.R) is linear
# in each node's shifted prediction, and every other term (the natural prediction and the
# ratio) is shared by the copies of a unit.  So least squares on copies that share their design
# row regresses the copy mean, which is the integrated transform, and the unit mean of the
# per-copy eif is the integrated influence curve.
#
# The shared adapter is sourced for lmtp_internal and fold_list only, and is not edited.

source("/fixture/lmtp_crossfit_adapter.R")

sdr_draws <- function(frame, plan, node, copies, as_label) {
  # The n x copies matrix of the values each copy of a unit takes at `node`.
  columns <- sprintf("shift__%s__%d__%d", plan, node, seq_len(copies))
  missing <- setdiff(columns, names(frame))
  if (length(missing)) stop(sprintf("the sample has no column %s", paste(missing, collapse = ", ")))
  values <- as.matrix(frame[columns])
  if (anyNA(values)) stop(sprintf("plan %s has a missing shifted value at node %d", plan, node))
  if (as_label) values <- matrix(as.character(values), nrow = nrow(values))
  values
}

sdr_ratios <- function(frame, plan, nodes) {
  # The n x T per-node ratios cleverly divides by.
  columns <- sprintf("ratio__%s__%d", plan, seq_len(nodes))
  missing <- setdiff(columns, names(frame))
  if (length(missing)) stop(sprintf("the sample has no column %s", paste(missing, collapse = ", ")))
  ratios <- as.matrix(frame[columns])
  if (anyNA(ratios) || !all(is.finite(ratios)) || any(ratios < 0)) {
    stop(sprintf("plan %s has a missing, negative or non-finite ratio", plan))
  }
  # The first node has no event upstream of it, so a correct first-node ratio averages one.
  first <- ratios[, 1]
  spread <- stats::sd(first) / sqrt(length(first))
  distance <- abs(mean(first) - 1)
  if (!is.finite(spread) || distance > 5 * spread) {
    stop(sprintf(
      "plan %s: the first-node ratio averages %.4f, %.1f standard errors from one",
      plan, mean(first), if (spread > 0) distance / spread else Inf
    ))
  }
  ratios
}

lmtp_sdr_with_folds <- function(
  frame,
  plan,
  trt,
  outcome,
  baseline,
  time_vary,
  cens = NULL,
  outcome_type = "binomial",
  copies = 1L,
  as_label = FALSE,
  learners_outcome = "SL.glm",
  control = lmtp::lmtp_control(.trim = 1, .learners_outcome_folds = 2)
) {
  # frame: one row per unit, the natural columns plus the study's shift__ and ratio__ columns
  # and the Python fold assignment `fold`.  Returns the estimate, the unit-level influence
  # curve and the plug-in at the first node.
  n <- nrow(frame)
  variables <- unique(c(baseline, unlist(time_vary), trt, cens, outcome))
  natural <- frame[, variables, drop = FALSE]
  if (as_label) {
    for (column in trt) natural[[column]] <- as.character(natural[[column]])
  }
  draws <- lapply(seq_along(trt), function(node) sdr_draws(frame, plan, node, copies, as_label))
  ratios <- sdr_ratios(frame, plan, length(trt))

  unit <- rep(seq_len(n), each = copies)
  natural <- natural[unit, , drop = FALSE]
  natural$..unit <- unit
  shifted <- natural
  for (node in seq_along(trt)) {
    shifted[[trt[[node]]]] <- as.vector(t(draws[[node]]))
  }
  replicated <- ratios[unit, , drop = FALSE]
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
    outcome_type = outcome_type,
    bounds = NULL,
    folds = length(unique(frame$fold)),
    weights = NULL
  )
  task$folds <- fold_list(rep(as.integer(frame$fold), each = copies))
  progress <- function(...) invisible(NULL)
  regressions <- lmtp_internal("cf_sdr")(task, replicated, learners_outcome, control, progress)
  if (!identical(dim(regressions$shifted), c(nrow(natural), length(trt) + 1L))) {
    stop("cf_sdr returned shifted regressions of an unexpected shape")
  }

  result <- lmtp_internal("theta_dr")(
    task = task,
    sequential_regressions = list(natural = regressions$natural, shifted = regressions$shifted),
    density_ratios = replicated,
    fits_m = regressions$fits,
    fits_r = NULL,
    shift = "supplied shifted values and ratios with exact folds",
    is_sdr = TRUE
  )
  if (!identical(result$estimator, "SDR")) stop("theta_dr did not take its SDR branch")
  unit_ic <- as.numeric(tapply(result$estimate@eif, unit, mean))
  initial <- as.numeric(tapply(regressions$shifted[, 1], unit, mean))
  if (anyNA(unit_ic) || anyNA(initial)) stop("the influence curve or the plug-in has a gap")
  list(estimate = result$estimate@x, ic = unit_ic, initial = mean(initial))
}

# Saturated cell means, for gate G2 only.  Each prediction is the mean of the training rows in
# the same cell of every design column, so a pooled fit with the arm in its design and a fit
# stratified on the arm coincide.  It replaces lmtp's run_ensemble only when a fit names it.
lmtp_cellmeans_learner <- function(data, y, learners, outcome_type, id, folds) {
  features <- setdiff(names(data), c(id, y))
  key <- do.call(paste, c(lapply(data[features], as.character), sep = "\r"))
  means <- tapply(data[[y]], key, mean)
  structure(list(means = means, features = features), class = "lmtp_cellmeans")
}

predict.lmtp_cellmeans <- function(object, newdata, tol = NULL, ...) {
  key <- do.call(paste, c(lapply(newdata[object$features], as.character), sep = "\r"))
  prediction <- unname(object$means[key])
  # A row whose history no training row shares has no cell mean; lmtp predicts every row and
  # masks the ones it does not read, so the gap is carried as NA rather than invented.
  if (is.null(tol)) return(prediction)
  pmin(1 - tol, pmax(tol, prediction))
}

local({
  namespace <- asNamespace("lmtp")
  original <- get("run_ensemble", envir = namespace)
  dispatch <- function(data, y, learners, outcome_type, id, folds) {
    if (identical(learners, "SL.cellmeans")) {
      return(lmtp_cellmeans_learner(data, y, learners, outcome_type, id, folds))
    }
    original(data, y, learners, outcome_type, id, folds)
  }
  unlockBinding("run_ensemble", namespace)
  assign("run_ensemble", dispatch, envir = namespace)
  lockBinding("run_ensemble", namespace)
})
registerS3method("predict", "lmtp_cellmeans", predict.lmtp_cellmeans)
