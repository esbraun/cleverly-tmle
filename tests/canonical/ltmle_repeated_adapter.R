# The repeated-outcome law's plans and mechanisms, for the R `ltmle` side of
# `longitudinal-repeated-outcome`.
#
# The law is `cleverly.datasets.make_longitudinal_repeated`: W on {0, 1, 2}, then at each of
# three nodes L_t (from t = 2), A_t, C_t and a binary Y_t that is not absorbing.  A_t and C_t
# read Y_{t-1} and Y_{t-2}.  The coefficients below restate `_REPEATED_A1`, `_REPEATED_A`,
# `_REPEATED_C1` and `_REPEATED_C` in `src/cleverly/datasets/longitudinal.py`.  Two checks
# guard the transcription before any run.  `KnownRepeatedMechanism`, the Python copy, is
# compared with the generator at every node.  This file is parsed by
# `test_the_r_adapter_restates_the_generator_coefficients`, which compares every literal of
# the four `plogis` lines with the generator's tables.
#
# `ltmle_regimen_adapter.R` serves the two-time-point laws with W1 and W2.  Its plans, its
# mechanism and its initial-estimate design are all specific to that law, and three published
# studies hash it, so this law has its own file rather than a branch there.

repeated_rule_label <- "treat then treat after an outcome"

repeated_arms <- function(frame, label, horizon) {
  # The `abar` matrix: one column per treatment node through `horizon`.  The rule treats at
  # node one and then exactly when the previous outcome was one.  A unit censored before node
  # t has no Y_{t-1}; its arm is filled with 0, because no fit reads a unit past its
  # censoring node.
  n <- nrow(frame)
  arms <- matrix(0, nrow = n, ncol = horizon)
  for (t in seq_len(horizon)) {
    arms[, t] <- if (label == "never") {
      rep(0, n)
    } else if (label == "always" || t == 1) {
      rep(1, n)
    } else if (label == repeated_rule_label) {
      previous <- frame[[sprintf("Y%d", t - 1)]]
      as.numeric(!is.na(previous) & previous > 0)
    } else {
      stop(sprintf("unknown plan %s", label))
    }
  }
  arms
}

repeated_mechanism <- function(frame, abar, horizon) {
  # The generating probabilities as a numeric gform, columns A1, C1, A2, C2, ... in node
  # order.  A column holds P(A_t = 1 | history) or P(C_t = 1 | history); ltmle selects p or
  # 1 - p from abar itself.  Each earlier arm and the current arm of a censoring column is the
  # plan's arm from `abar`: the clever covariate needs g on the intervened history, and on the
  # followed path it equals the observed one.  A missing L_t or Y_{t-1} is filled with 0,
  # inside the range, for the reason `ltmle_regimen_adapter.R` gives.
  fill <- function(name) {
    values <- frame[[name]]
    ifelse(is.na(values), 0, values)
  }
  w <- frame$W
  columns <- list()
  for (t in seq_len(horizon)) {
    if (t == 1) {
      p_a <- plogis(0.2 + 0.2 * w)
      p_c <- plogis(2.4 + 0.2 * w - 0.3 * abar[, 1])
    } else {
      l <- fill(sprintf("L%d", t))
      y <- fill(sprintf("Y%d", t - 1))
      # Y_{t-2} is Y_0 = 0 at t = 2, so the term vanishes there by construction.
      y2 <- if (t >= 3) fill(sprintf("Y%d", t - 2)) else 0
      p_a <- plogis(-1.2 + 0.2 * w + 0.5 * l - 0.2 * y + 0.7 * abar[, t - 1] + 1.6 * y2)
      p_c <- plogis(3.6 + 0.3 * l - 0.2 * y + 0.3 * abar[, t] - 1.4 * y2)
    }
    columns[[length(columns) + 1]] <- p_a
    columns[[length(columns) + 1]] <- p_c
  }
  do.call(cbind, columns)
}

repeated_ltmle_call <- function(arguments) {
  # One `ltmle` call under the study's warning policy and stage.
  #
  # A rank-deficient stratified regression warns "prediction from rank-deficient fit" and stays
  # fatal: its predictions off the fitted rows are not unique, so the paired comparison would
  # read an arbitrary R value.  At n = 400 one smoke replicate hits it (the rule's third node,
  # about 18 followers).  At the declared n = 2,000 none of the 1,600 primary draws does
  # (X32a Study queue item 11).
  withCallingHandlers(
    study_stage("ltmle", do.call(ltmle, arguments)),
    warning = function(condition) {
      if (grepl("Cnodes|censoring", conditionMessage(condition), ignore.case = TRUE)) {
        invokeRestart("muffleWarning")
      }
      study_fatal_error(sprintf("unexpected ltmle warning: %s", conditionMessage(condition)))
    }
  )
}

repeated_ltmle_fit <- function(arguments, label) {
  # The targeted fit, its influence curve and the cumulative-bound check of
  # `regimen_ltmle_fit`, plus the fully untargeted plug-in.
  #
  # The initial estimate is a second call with `gcomp = TRUE`, the sequential regression with
  # no fluctuation.  The first block's regression of the targeted chain, which
  # `ltmle_regimen_adapter.R` publishes, cannot serve here.  W is binary and the first block's
  # design is `~ W`, so that regression is saturated in W and its fluctuation is exactly zero.
  # It would equal the targeted estimate in every replicate, and the check that the reference
  # moved off its plug-in would read nothing.
  targeted <- repeated_ltmle_call(arguments)
  if (any(abs(targeted$cum.g - targeted$cum.g.unbounded) > 1e-12, na.rm = TRUE)) {
    stop(sprintf("%s activated a cumulative g bound", label))
  }
  plug_in <- repeated_ltmle_call(c(arguments, list(gcomp = TRUE)))
  list(
    estimate = unname(targeted$estimates[["tmle"]]),
    initial = unname(plug_in$estimates[["gcomp"]]),
    ic = as.numeric(targeted$IC$tmle)
  )
}
