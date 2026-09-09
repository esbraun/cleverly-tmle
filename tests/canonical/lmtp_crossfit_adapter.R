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
    training <- if (length(labels) == 1L) validation else which(assignment != label)
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
  id = NULL,
  weights = NULL,
  outcome_type = "binomial",
  fold_assignment,
  learners_outcome = "SL.glm",
  learners_trt = "SL.glm",
  # Optional exact per-node density ratios, supplied so the paired comparison isolates the
  # recursion rather than two unrelated mechanism-fitting pipelines -- the principle the
  # ordinary `ltmle` studies apply by passing a numeric `gform`.  `lmtp` has no `gform`
  # equivalent, so the substitution happens here, the same way `task$folds` is replaced.
  #
  # Column t must be 1{A_t = d_t, C_t = 1} / (g_t * c_t), evaluated on the intervened
  # history.  That formula is not documented; it was established by fitting each candidate
  # against `cf_density_ratios` output on this study's own law, where the censoring-inclusive
  # form matched its zero pattern exactly and the treatment-only form did not.  The checks
  # below re-assert both halves of it on every run rather than trusting the derivation.
  density_ratios = NULL,
  # ``.return_full_fits`` is not optional here.  The block below reads
  # ``regressions$fits`` to rebuild the unfluctuated plug-in, and the check after the call
  # refuses a control without it -- so a default that omitted the flag was a default that
  # could never run.  Every caller still passes its own control; this one exists so that a
  # future caller which does not gets a working adapter rather than an error.
  control = lmtp::lmtp_control(
    .trim = 1,
    .learners_outcome_folds = 5,
    .learners_trt_folds = 5,
    .return_full_fits = TRUE
  )
) {
  if (nrow(data) != length(fold_assignment)) stop("fold assignment has the wrong length")
  if (!is.null(weights)) {
    if (length(weights) != nrow(data)) stop("weights have the wrong length")
    if (!is.numeric(weights) || any(!is.finite(weights)) || any(weights <= 0)) {
      stop("weights must be finite positive numbers")
    }
    weights <- as.numeric(weights)
  }
  if (!is.null(id) && (!is.character(id) || length(id) != 1L || !id %in% names(data))) {
    stop("id must be NULL or the name of one cluster identifier column")
  }
  variables <- unique(c(unlist(trt), outcome, unlist(time_vary), baseline, cens, id))
  natural <- data[, variables, drop = FALSE]
  intervention <- shifted[, variables, drop = FALSE]

  if (!is.null(id)) {
    if (!identical(natural[[id]], intervention[[id]])) {
      stop("natural and shifted data must preserve the cluster identifier")
    }
    cluster_folds <- split(as.integer(fold_assignment), natural[[id]])
    split_clusters <- names(cluster_folds)[vapply(
      cluster_folds, function(labels) length(unique(labels)) != 1L, logical(1)
    )]
    if (length(split_clusters)) {
      stop(sprintf(
        "%d cluster(s) are split across folds; the first is %s",
        length(split_clusters), split_clusters[[1]]
      ))
    }
  }

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
    weights = weights
  )
  task$folds <- fold_list(fold_assignment)
  progress <- function(...) invisible(NULL)

  density <- lmtp_internal("cf_density_ratios")(
    task, learners_trt, FALSE, control, progress
  )
  if (!is.null(density_ratios)) {
    estimated <- density$density_ratios
    if (!identical(dim(density_ratios), dim(estimated))) {
      stop(sprintf(
        "supplied density ratios are %s; lmtp's are %s",
        paste(dim(density_ratios), collapse = " x "),
        paste(dim(estimated), collapse = " x ")
      ))
    }
    # A per-node ratio is zero exactly where the unit left the followed path at that node.
    # lmtp's own matrix has that property, so requiring the supplied one to agree with it
    # cell for cell is what catches a ratio built from the wrong arm, the wrong node, or a
    # missing censoring factor -- none of which changes the shape.
    # ``any(xor(...))`` rather than ``identical(...)``: the latter compares dimnames too, so
    # a supplied matrix built with ``cbind(first, second)`` failed on its column labels while
    # every value agreed.
    disagreeing <- sum(xor(density_ratios == 0, estimated == 0))
    if (disagreeing > 0) {
      stop(sprintf(
        "the supplied density ratios follow a different path from lmtp's: %d of %d cells
         disagree on whether the unit is still on the regimen at that node",
        disagreeing, length(estimated)
      ))
    }
    if (!all(is.finite(density_ratios))) {
      stop("the supplied density ratios contain non-finite values")
    }
    # Agreement in *shape* with lmtp's own estimate.  The zero-pattern check above is the
    # sharp structural one; this catches a formula that is right about which units follow and
    # wrong about the weight -- a dropped censoring factor, an inverted arm probability.
    supplied <- apply(density_ratios, 1, prod)
    fitted <- apply(estimated, 1, prod)
    agreement <- suppressWarnings(cor(supplied, fitted))
    if (!is.finite(agreement) || agreement < 0.8) {
      stop(sprintf(
        "the supplied cumulative density ratio correlates %.4f with lmtp's own estimate of
         the same quantity; a correct formula tracks it closely even where the two differ
         on the value", agreement
      ))
    }
    # A gross-error screen only, deliberately loose.  The exact weights are heavy tailed --
    # under the never-treat plan 1/(1 - g_2) grows without bound as L2 does -- so the sample
    # mean of a correct cumulative ratio wanders well away from its expectation of one at
    # these sample sizes.  Tightening this rejects correct ratios; the two checks above are
    # the ones that carry the weight.
    if (abs(mean(supplied) - 1) > 0.5) {
      stop(sprintf(
        "the supplied cumulative density ratio averages %.4f; a correct one averages one",
        mean(supplied)
      ))
    }
    density$density_ratios <- density_ratios
  }
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
    # The index is "one past the first treatment node", which is the conditioning set of the
    # *earliest* sequential regression.  It is not the number of nodes, so it stays 2 for a
    # one-node task as well as a two-node one -- which is the case the survival runner's
    # horizon-one fit exercises, and which nothing had ever run when this was written.  If
    # lmtp's convention is the other one, the columns are absent rather than wrong, so the
    # guard below turns a cryptic subscript error into the diagnosis.
    first_regression_history <- 2
    history <- task$vars$history("L", first_regression_history)
    first_treatment <- lmtp_internal("current_trt")(task$vars$A, 1)
    missing_columns <- setdiff(c(history, first_treatment), names(natural_fold$valid))
    if (length(missing_columns)) {
      stop(sprintf(
        paste(
          "the earliest sequential regression's history (%s) is not in the fold data for a",
          "%d-node task; lmtp's history index is not the convention this adapter assumes"
        ),
        paste(missing_columns, collapse = ", "), length(task$vars$A)
      ))
    }
    under_shift <- natural_fold$valid[valid, c("..i..lmtp_id", history)]
    under_shift[, first_treatment] <- shifted_fold$valid[valid, first_treatment]
    # ``fits[[fold]]`` is assumed to be indexed from the *earliest* node, so that ``[[1]]``
    # is the regression whose conditioning set is the first node's history.  That is an
    # assumption about lmtp's internals rather than a documented interface.  It is checked
    # by what follows rather than trusted: the count must match the number of nodes, and
    # ``under_shift`` carries only the first node's history, so a regression fitted at a
    # later node would be asked to predict without the columns it was fitted on.  The
    # bounds check is the one that catches a silent success -- a fitted value outside
    # (0, 1) is not a plug-in for a binary node whatever produced it.
    if (length(regressions$fits[[fold]]) != length(task$vars$A)) {
      stop(sprintf(
        "fold %d has %d sequential regressions for %d treatment nodes",
        fold, length(regressions$fits[[fold]]), length(task$vars$A)
      ))
    }
    predicted <- predict(regressions$fits[[fold]][[1]], under_shift, 1e-05)
    if (anyNA(predicted) || any(predicted < 0) || any(predicted > 1)) {
      stop(sprintf("fold %d produced an initial plug-in outside [0, 1]", fold))
    }
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
  result$initial <- if (outcome_type == "continuous") task$rescale(initial) else initial
  result$fold_assignment <- as.integer(fold_assignment)
  result
}
