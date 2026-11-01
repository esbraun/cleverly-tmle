# The K-node R ltmle 1.3-0 fits of the long-horizon ordinary LTMLE study (X59, S1).
#
# Every law and plan quantity is read from the sample (tests/studies/long_horizon_reference.py
# writes it): the observed nodes, the numeric gform columns gA<t> = P(A_t = 1 | H_t) and
# gC<t> = P(C_t = 1 | H_t, A_t), the arm each deterministic plan assigns abar__<plan>__<t>, the
# duration of each MSM plan, and the scenario's K and declared mutation (design__*).  Nothing
# here restates a rule of the law or of a plan.
#
# The Q-form mapping (X59-plan.md section 2.1, M3).  ltmle names each Q regression by the L or
# Y node it regresses before: L_{t+1} for cleverly's node t < K, and Y for node K.  cleverly's
# node-t design (tests/studies/long_horizon_common.py, _design) is W and L2..Lt as numeric
# columns, without arm terms for a deterministic plan, fitted on the units that follow the plan
# through t.  stratify = TRUE fits on exactly those units, so each formula regresses on W and
# L2..Lt and no A term:
#
#   Qform[L2] = Q.kplus1 ~ W,  Qform[L3] = Q.kplus1 ~ W + L2,  ...,  Qform[Y] = Q.kplus1 ~ W + L2 + ... + LK
#
# horizon_qform(K) writes that table; the declared mutation 2 drops W from the Y formula.
#
# The warning policy and the cumulative-g-bound stop are those of ltmle_regimen_adapter.R
# (regimen_ltmle_fit), restated in horizon_ltmle_call because that function reads the two-node
# law's design back; the shipped file is not edited.  The Q-bound stop is new: no untargeted
# prediction of a unit at risk may leave cleverly's clip [5e-4, 0.9995], because then the pair
# compares two clips.

horizon_q_bounds <- c(5e-4, 0.9995)

horizon_data_columns <- function(K) {
  # ltmle's node order: W, A1, C1, L2, A2, C2, ..., LK, AK, CK, Y.
  columns <- "W"
  for (t in seq_len(K)) {
    if (t > 1) columns <- c(columns, paste0("L", t))
    columns <- c(columns, paste0("A", t), paste0("C", t))
  }
  c(columns, "Y")
}

horizon_gform_columns <- function(K) {
  as.vector(rbind(paste0("gA", seq_len(K)), paste0("gC", seq_len(K))))
}

horizon_history <- function(t) {
  # The covariates of cleverly's node-t regression.
  c("W", if (t > 1) paste0("L", 2:t))
}

horizon_qform <- function(K, mutation = 0L) {
  names <- c(if (K > 1) paste0("L", 2:K), "Y")
  formulas <- vapply(seq_len(K), function(t) {
    terms <- horizon_history(t)
    if (mutation == 2L && t == K) terms <- setdiff(terms, "W")
    paste("Q.kplus1 ~", if (length(terms)) paste(terms, collapse = " + ") else "1")
  }, character(1))
  setNames(formulas, names)
}

horizon_abar <- function(frame, plan, K) {
  columns <- sprintf("abar__%s__%d", plan, seq_len(K))
  missing <- setdiff(columns, names(frame))
  if (length(missing)) stop(sprintf("the sample has no column %s", paste(missing, collapse = ", ")))
  abar <- as.matrix(frame[columns])
  if (anyNA(abar)) stop(sprintf("plan %s has a missing assigned arm", plan))
  storage.mode(abar) <- "numeric"
  unname(abar)
}

horizon_gform <- function(frame, K) {
  gform <- as.matrix(frame[horizon_gform_columns(K)])
  if (anyNA(gform) || any(gform <= 0 | gform > 1)) {
    stop("the supplied gform has a missing value or a probability outside (0, 1]")
  }
  unname(gform)
}

horizon_q_predictions <- function(fit, frame, K, qform) {
  # Each node's untargeted prediction on the units at risk there (A_t observed), from the
  # coefficients ltmle reports.  The design carries no arm term, so a unit's prediction is the
  # plan's whether or not the unit followed.
  if (length(fit$fit$Q) != K) {
    stop(sprintf("ltmle reports %d Q regressions for %d nodes", length(fit$fit$Q), K))
  }
  lapply(seq_len(K), function(t) {
    table <- fit$fit$Q[[t]]
    rows <- !is.na(frame[[paste0("A", t)]])
    if (inherits(table, "no.Y.variation")) return(rep(unname(table$Y.value), sum(rows)))
    coefficients <- table[, "Estimate"]
    if (anyNA(coefficients)) stop(sprintf("node %d's Q regression has an undefined coefficient", t))
    design <- model.matrix(as.formula(sub("^Q.kplus1", "", qform[[t]])), data = frame[rows, , drop = FALSE])
    as.numeric(plogis(design[, names(coefficients), drop = FALSE] %*% coefficients))
  })
}

horizon_ltmle_arguments <- function(frame, plan, K, mutation, gcomp = FALSE) {
  qform <- horizon_qform(K, mutation)
  list(
    data = frame[horizon_data_columns(K)],
    Anodes = paste0("A", seq_len(K)),
    Cnodes = paste0("C", seq_len(K)),
    Lnodes = if (K > 1) paste0("L", 2:K) else NULL,
    Ynodes = "Y",
    survivalOutcome = FALSE,
    Qform = qform,
    gform = horizon_gform(frame, K),
    abar = horizon_abar(frame, plan, K),
    gbounds = c(1e-8, 1),
    SL.library = "glm",
    stratify = TRUE,
    gcomp = gcomp,
    variance.method = "ic",
    estimate.time = FALSE
  )
}

horizon_ltmle_call <- function(arguments, label) {
  # The warning policy and the cumulative-g-bound stop of ltmle_regimen_adapter.R's
  # regimen_ltmle_fit, stated again because that function reads the two-node law's W1 + W2
  # design back.  Only the binary-censoring notice is muffled; any other warning stops the
  # run (RM39a2: study_fatal_error is never recorded as a group failure).
  fit <- withCallingHandlers(
    study_stage("ltmle", do.call(ltmle, arguments)),
    warning = function(condition) {
      if (grepl("Cnodes|censoring", conditionMessage(condition), ignore.case = TRUE)) {
        invokeRestart("muffleWarning")
      }
      study_fatal_error(sprintf("unexpected ltmle warning: %s", conditionMessage(condition)))
    }
  )
  # The exact mechanism is supplied, so a binding cumulative bound is a defect of the pairing.
  if (any(abs(fit$cum.g - fit$cum.g.unbounded) > 1e-12, na.rm = TRUE)) {
    stop(sprintf("%s activated a cumulative g bound", label))
  }
  fit
}

horizon_initial <- function(fit, frame, qform) {
  # `initial_estimate`: the mean over every unit of the earliest node's regression of the
  # already targeted later node, the convention of long_horizon_primary._initial_mean.
  table <- fit$fit$Q[[1]]
  if (inherits(table, "no.Y.variation")) return(unname(table$Y.value))
  coefficients <- table[, "Estimate"]
  design <- model.matrix(as.formula(sub("^Q.kplus1", "", qform[[1]])), data = frame)
  mean(plogis(design[, names(coefficients), drop = FALSE] %*% coefficients))
}

horizon_ltmle_plan <- function(frame, plan) {
  K <- as.integer(unique(frame$design__K))
  mutation <- as.integer(unique(frame$design__mutation))
  if (length(K) != 1L || length(mutation) != 1L) stop("the sample's design constants vary")
  arguments <- horizon_ltmle_arguments(frame, plan, K, mutation)
  targeted <- horizon_ltmle_call(arguments, plan)
  # The untargeted run: the same regressions without the fluctuation, for G-C.
  untargeted <- horizon_ltmle_call(horizon_ltmle_arguments(frame, plan, K, mutation, TRUE), plan)
  predictions <- unlist(horizon_q_predictions(untargeted, frame, K, arguments$Qform))
  outside <- sum(predictions < horizon_q_bounds[[1]] | predictions > horizon_q_bounds[[2]])
  if (outside > 0) {
    stop(sprintf(
      "plan %s: %d untargeted Q predictions leave [%g, %g], where cleverly's clip would bind",
      plan, outside, horizon_q_bounds[[1]], horizon_q_bounds[[2]]
    ))
  }
  list(
    estimate = unname(targeted$estimates[["tmle"]]),
    ic = as.numeric(targeted$IC$tmle),
    initial = horizon_initial(targeted, frame, arguments$Qform),
    gcomp = unname(untargeted$estimates[["gcomp"]])
  )
}

horizon_msm <- function(fits, durations) {
  # The per-plan TMLEs projected on the working model E[Y_d] = b0 + b1 d with unit weights:
  # the least-squares coefficients and their linear influence curves.  cleverly fits the pooled
  # MSM fluctuation instead, a different estimator of the same target (X59-plan.md, 4.4).
  design <- cbind(1, durations)
  projection <- solve(crossprod(design), t(design))
  estimates <- vapply(fits, function(fit) fit$estimate, numeric(1))
  initials <- vapply(fits, function(fit) fit$initial, numeric(1))
  gcomps <- vapply(fits, function(fit) fit$gcomp, numeric(1))
  curves <- do.call(cbind, lapply(fits, function(fit) fit$ic))
  terms <- c("msm_regimen[(intercept)]", "msm_regimen[duration]")
  setNames(lapply(1:2, function(j) {
    list(
      estimate = sum(projection[j, ] * estimates),
      initial = sum(projection[j, ] * initials),
      gcomp = sum(projection[j, ] * gcomps),
      ic = as.numeric(curves %*% projection[j, ])
    )
  }), terms)
}
