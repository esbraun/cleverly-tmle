# The repeated-outcome law's plans and mechanisms, for the R `ltmle` side of
# `longitudinal-repeated-outcome`.
#
# The law is `cleverly.datasets.make_longitudinal_repeated`: W, then at each of three nodes
# L_t (from t = 2), A_t, C_t and a binary Y_t that is not absorbing.  The coefficients below
# restate `_REPEATED_*` in `src/cleverly/datasets/longitudinal.py`.  The Python side checks its
# own copy, `KnownRepeatedMechanism`, against the generator at every node, and a paired run
# whose two copies disagreed would publish a cleverly-ltmle displacement far outside the
# margin.
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
      p_a <- plogis(-0.2 + 0.6 * w)
      p_c <- plogis(2.0 + 0.3 * w - 0.3 * abar[, 1])
    } else {
      l <- fill(sprintf("L%d", t))
      y <- fill(sprintf("Y%d", t - 1))
      p_a <- plogis(-0.2 + 0.4 * w + 0.5 * l - 1.4 * y + 0.6 * abar[, t - 1])
      p_c <- plogis(2.2 + 0.3 * l - 1.0 * y + 0.2 * abar[, t])
    }
    columns[[length(columns) + 1]] <- p_a
    columns[[length(columns) + 1]] <- p_c
  }
  do.call(cbind, columns)
}

repeated_ltmle_fit <- function(arguments, frame, label) {
  # One `ltmle` call and what the study reads back, with the warning policy and the
  # cumulative-bound check of `regimen_ltmle_fit`.  The initial estimate is the first block's
  # regression, whose design here is `~ W`.
  targeted <- withCallingHandlers(
    do.call(ltmle, arguments),
    warning = function(condition) {
      if (grepl("Cnodes|censoring", conditionMessage(condition), ignore.case = TRUE)) {
        invokeRestart("muffleWarning")
      }
      stop(sprintf("unexpected ltmle warning: %s", conditionMessage(condition)))
    }
  )
  if (any(abs(targeted$cum.g - targeted$cum.g.unbounded) > 1e-12, na.rm = TRUE)) {
    stop(sprintf("%s activated a cumulative g bound", label))
  }
  first_q <- targeted$fit$Q[[1]]
  initial <- if (inherits(first_q, "no.Y.variation")) {
    unname(first_q$Y.value)
  } else {
    coefficients <- first_q[, "Estimate"]
    design <- model.matrix(~W, data = frame)
    unname(mean(plogis(design[, names(coefficients), drop = FALSE] %*% coefficients)))
  }
  list(
    estimate = unname(targeted$estimates[["tmle"]]),
    initial = initial,
    ic = as.numeric(targeted$IC$tmle)
  )
}
