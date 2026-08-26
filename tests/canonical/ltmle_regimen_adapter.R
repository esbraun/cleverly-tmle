# The generating law's plans and mechanisms, for every study that fits it with R `ltmle`.
#
# Three registered studies draw from the same censored two-time-point law -- the end-of-study
# regimen means, the survival curve, and the MSM projection over four plans -- and each one has
# to tell `ltmle` which arms a plan assigns and what the true treatment and censoring
# probabilities are.  Those two answers are properties of the *law*, not of the study, so they
# are stated here once.
#
# They were stated three times before this file existed, and the copies had already lost
# what matters most about them.  The comments below are not decoration: they record why an
# absent L2 is filled rather than left NA, and why a numeric gform column carries something
# different from what a reader first expects.  The MSM runner was cloned from the end-of-study
# runner with the code byte-identical and all of that removed.

regimen_arms <- function(frame, label, horizon = 2, rule = NULL) {
  # The `abar` matrix `ltmle` intervenes with: one column per treatment node, one row per unit.
  #
  # `rule` is the dynamic plan's second-node decision, because the two laws spell it
  # differently -- the end-of-study and survival panels read `L2 > 0`, the competing-risk panel
  # reads `L2 == 1`.  Everything else is shared, including the "early" plan, which only the MSM
  # study declares: a plan no study names costs the studies that do not name it nothing, and a
  # branch missing from one copy is what this file exists to prevent.
  if (is.null(rule)) rule <- function(l2) as.numeric(!is.na(l2) & l2 > 0)
  a1 <- if (label == "never") 0 else 1
  if (horizon == 1) return(matrix(rep(a1, nrow(frame)), ncol = 1))
  a2 <- if (label == "never" || label == "early") {
    rep(0, nrow(frame))
  } else if (label == "always") {
    rep(1, nrow(frame))
  } else {
    rule(frame$L2)
  }
  cbind(rep(a1, nrow(frame)), a2)
}

regimen_mechanism <- function(frame, abar, horizon = 2) {
  # The generating treatment and censoring probabilities, supplied to `ltmle` as a numeric
  # gform so both implementations are compared on their sequential regressions and targeting
  # rather than on two unrelated mechanism-fitting pipelines.
  p_a1 <- plogis(0.3 * frame$W1 - 0.4 * frame$W2)
  p_c1 <- plogis(2.2 + 0.3 * frame$W1 - 0.3 * abar[, 1])
  if (horizon == 1) return(cbind(p_a1, p_c1))

  # L2 is absent for a unit censored at C1.  Those units left before the second node, so
  # nothing downstream of it reads their probabilities -- but a numeric gform matrix is an
  # input, and an input with NA in it depends on how ltmle happens to carry one through a
  # cumulative product.  Filled with a value in range so the matrix is well defined; each
  # study's cumulative-bound check still runs with na.rm in case a future version produces one
  # anyway.
  #
  # In the survival panel L2 is absent for a second, structurally different reason: a unit that
  # had the event at Y1.  The filler is irrelevant for it for a stronger reason still.  With
  # survivalOutcome = TRUE that row is deterministic after the event, and ltmle's CalcG sets
  # `g[deterministic.newdata] <- 1` regardless of what this matrix says ("a=abar
  # deterministically after death", ltmle/R/ltmle.R).  So the supplied probability there is not
  # merely unused downstream, it is overwritten.
  l2 <- ifelse(is.na(frame$L2), 0, frame$L2)

  # Numeric gform columns carry P(A_t = 1 | history), not the probability of the assigned arm.
  # ltmle selects p or 1-p from abar internally, as cleverly does.
  #
  # The conditioning differs from cleverly's and both are right.  Here the two history terms
  # are abar[, 1], the arm the *regimen* assigns; cleverly's KnownLongitudinalMechanism reads
  # the observed A1.  The clever covariate needs g on the intervened history, and on the
  # followed path the two coincide, so the difference lives entirely among units the follower
  # mask has already zeroed.  That the two implementations then agree to between 1e-10 and
  # 2e-7 is incidental evidence that neither lets an off-path probability reach the estimate.
  p_a2 <- plogis(0.5 * l2 + 0.6 * abar[, 1] - 0.2 * frame$W2)
  p_c2 <- plogis(2.4 + 0.2 * l2)
  cbind(p_a1, p_c1, p_a2, p_c2)
}

regimen_ltmle_fit <- function(arguments, frame, label) {
  # One `ltmle` call, its warning policy, its positivity check and what the study reads back.
  # The three studies differ in the nodes and formulas they build; none of them differs in any
  # of this, and each one carried its own copy of it.
  #
  # Not suppressWarnings(): ltmle warns about a binary censoring column being coerced to a
  # factor on every fit, and blanket suppression made that notice indistinguishable from a
  # positivity warning about the very quantity these studies report.  Only the known message
  # is muffled; anything else stops the run.
  targeted <- withCallingHandlers(
    do.call(ltmle, arguments),
    warning = function(condition) {
      if (grepl("Cnodes|censoring", conditionMessage(condition), ignore.case = TRUE)) {
        invokeRestart("muffleWarning")
      }
      stop(sprintf("unexpected ltmle warning: %s", conditionMessage(condition)))
    }
  )
  # The studies supply the exact mechanism, so a cumulative bound that binds is a defect in
  # the comparison rather than a regularisation: it would mean the two implementations were
  # no longer being handed the same g.
  if (any(abs(targeted$cum.g - targeted$cum.g.unbounded) > 1e-12, na.rm = TRUE)) {
    stop(sprintf("%s activated a cumulative g bound", label))
  }
  # `fit$Q[[1]]` is the earliest node's regression of the *already targeted* later node, which
  # is what the published `initial_estimate` means on both sides of the comparison.
  first_q <- targeted$fit$Q[[1]]
  initial <- if (inherits(first_q, "no.Y.variation")) {
    unname(first_q$Y.value)
  } else {
    coefficients <- first_q[, "Estimate"]
    design <- model.matrix(~ W1 + W2, data = frame)
    unname(mean(plogis(design[, names(coefficients), drop = FALSE] %*% coefficients)))
  }
  list(
    estimate = unname(targeted$estimates[["tmle"]]),
    initial = initial,
    ic = as.numeric(targeted$IC$tmle)
  )
}
