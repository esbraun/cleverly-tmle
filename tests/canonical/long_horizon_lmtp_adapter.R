# The K-node lmtp fits of the long-horizon studies (X59): S2 long-horizon-categorical-ltmle
# through lmtp_tmle and S3 long-horizon-sdr through lmtp_sdr.
#
# Every law and plan quantity is read from the sample (tests/studies/long_horizon_reference.py
# writes it): the per-node ratios ratio__<plan>__<t>, the arm of each copy shift__<plan>__<t>__<c>,
# the plan's policy nodes policy__<plan>, the realised outer fold, and the scenario's K, m and
# declared mutation (design__*).  The plans and the estimands are read from the truth table.
# Nothing here restates a rule of the law or of a plan.
#
# A deterministic plan is fitted by lmtp_9996b04_tmle_with_folds (S2) or lmtp_sdr_with_folds
# (S3) with one copy per unit; a plan with a known stochastic node by lmtp_9996b04_policy_tmle
# (S2) or lmtp_sdr_with_folds with four copies (S3), by proportional replication.  The outcome
# learner is the plan-stratified regression of lmtp_horizon_learners.R; CLEVERLY_X59_LEARNER=
# pooled selects the declared fallback (the pooled quasibinomial fit for S2, SL.glm for S3).
#
# Callers source study_harness.R, lmtp_sdr_adapter.R, the two 9996b04 adapters and
# lmtp_horizon_learners.R before this file, and call study_stream themselves.

#: cleverly's outcome-regression clip, which no prediction may reach (X59-plan.md, M3).
horizon_q_bounds <- c(5e-4, 0.9995)
horizon_z <- qnorm(0.975)

horizon_typed <- function(frame) {
  # One archive holds every scenario, so a column another scenario wrote is empty here.
  frame[] <- lapply(frame, function(column) {
    if (is.character(column)) type.convert(column, as.is = TRUE, na.strings = c("", "NA")) else column
  })
  frame[, colSums(!is.na(frame)) > 0, drop = FALSE]
}

horizon_constant <- function(frame, name) {
  value <- unique(frame[[name]])
  if (length(value) != 1L || is.na(value)) stop(sprintf("the sample's %s is not one constant", name))
  as.integer(value)
}

horizon_parse <- function(name) {
  # ey_regimen[P] or ate_regimen[P vs Q].
  inner <- sub("^[a-z_]+\\[(.*)\\]$", "\\1", name)
  if (startsWith(name, "ate_regimen[")) {
    sides <- strsplit(inner, " vs ", fixed = TRUE)[[1]]
    if (length(sides) != 2L) stop(sprintf("estimand %s is not a contrast of two plans", name))
    return(list(left = sides[[1]], right = sides[[2]]))
  }
  if (!startsWith(name, "ey_regimen[")) stop(sprintf("estimand %s is not a plan mean or contrast", name))
  list(left = inner, right = NULL)
}

horizon_columns <- function(frame, prefix, plan, nodes, suffix = "") {
  columns <- sprintf("%s__%s__%d%s", prefix, plan, nodes, suffix)
  missing <- setdiff(columns, names(frame))
  if (length(missing)) stop(sprintf("the sample has no column %s", paste(missing, collapse = ", ")))
  columns
}

horizon_learner <- function(comparator) {
  choice <- study_env("CLEVERLY_X59_LEARNER", "stratified")
  if (identical(choice, "stratified")) {
    return(if (comparator == "lmtp_sdr") "SL.horizon.sdr" else "SL.horizon.quasibinomial")
  }
  if (identical(choice, "pooled")) {
    return(if (comparator == "lmtp_sdr") "SL.glm" else "SL.horizon.pooled")
  }
  stop(sprintf("CLEVERLY_X59_LEARNER=%s is not a declared learner", choice))
}

horizon_check_q <- function(predictions, at_risk, plan) {
  # predictions: rows x 2K (natural then shifted); at_risk: rows x K.
  risk <- cbind(at_risk, at_risk)
  values <- predictions[risk]
  values <- values[!is.na(values)]
  outside <- sum(values < horizon_q_bounds[[1]] | values > horizon_q_bounds[[2]])
  if (outside > 0) {
    stop(sprintf(
      "plan %s: %d outcome predictions leave [%g, %g], where cleverly's clip would bind",
      plan, outside, horizon_q_bounds[[1]], horizon_q_bounds[[2]]
    ))
  }
  invisible(TRUE)
}

horizon_lmtp_plan <- function(frame, plan, comparator) {
  K <- horizon_constant(frame, "design__K")
  m <- horizon_constant(frame, "design__m")
  mutation <- horizon_constant(frame, "design__mutation")
  nodes <- seq_len(K)
  trt <- paste0("A", nodes)
  cens <- paste0("C", nodes)
  time_vary <- c(list(NULL), as.list(paste0("L", nodes[-1])))
  ratios <- as.matrix(frame[horizon_columns(frame, "ratio", plan, nodes)])
  if (anyNA(ratios)) stop(sprintf("plan %s has a missing ratio", plan))
  mask <- unique(frame[[sprintf("policy__%s", plan)]])
  if (length(mask) != 1L) stop(sprintf("plan %s has no single policy mask", plan))
  policy <- horizon_policy_nodes(mask, K)
  copies <- if (length(policy)) 4L else 1L
  draws <- lapply(nodes, function(node) {
    as.matrix(frame[horizon_columns(frame, "shift", plan, node, sprintf("__%d", seq_len(copies)))])
  })
  horizon_learner_state(ratios, policy, m, K, mutation)
  learner <- horizon_learner(comparator)
  frame$..unit <- seq_len(nrow(frame))
  at_risk <- !is.na(as.matrix(frame[trt]))

  if (comparator == "lmtp_sdr") {
    fit <- lmtp_sdr_with_folds(
      frame,
      plan,
      trt = trt,
      outcome = "Y",
      baseline = "W",
      time_vary = time_vary,
      cens = cens,
      outcome_type = "binomial",
      copies = copies,
      as_label = FALSE,
      learners_outcome = learner
    )
    return(list(estimate = fit$estimate, ic = fit$ic, initial = fit$initial))
  }
  if (comparator != "lmtp_tmle") stop(sprintf("unknown lmtp comparator %s", comparator))
  variables <- c("W", unlist(time_vary), trt, cens, "Y")
  if (!length(policy)) {
    natural <- frame[c(variables, "..unit")]
    shifted <- natural
    for (node in nodes) shifted[[trt[[node]]]] <- as.vector(draws[[node]][, 1])
    fit <- lmtp_9996b04_tmle_with_folds(
      natural,
      shifted,
      trt = trt,
      outcome = "Y",
      baseline = "W",
      time_vary = time_vary,
      cens = cens,
      id = "..unit",
      fold_assignment = frame$fold,
      learners_outcome = learner,
      density_ratios = ratios
    )
    if (!identical(fit$fold_assignment, as.integer(frame$fold))) {
      stop("lmtp did not retain the supplied fold assignment")
    }
    horizon_check_q(fit$predictions, at_risk, plan)
  } else {
    fit <- lmtp_9996b04_policy_tmle(
      frame[variables],
      draws,
      ratios,
      trt = trt,
      outcome = "Y",
      baseline = "W",
      time_vary = time_vary,
      cens = cens,
      fold_assignment = frame$fold,
      copies = copies,
      learners_outcome = learner
    )
    horizon_check_q(fit$predictions, at_risk[rep(seq_len(nrow(frame)), each = copies), , drop = FALSE], plan)
  }
  list(estimate = fit$estimate, ic = fit$ic, initial = fit$initial)
}

horizon_row <- function(scenario, replicate, name, truth, estimate, initial, ic, n) {
  standard_error <- sd(ic) / sqrt(n)
  low <- estimate - horizon_z * standard_error
  high <- estimate + horizon_z * standard_error
  data.frame(
    implementation = "lmtp",
    scenario = scenario,
    replicate = replicate,
    n = n,
    estimand = name,
    truth = truth,
    estimate = estimate,
    inference_estimate = estimate,
    std_error = standard_error,
    ci_lower = low,
    ci_upper = high,
    inference_scale = "identity",
    covered = as.integer(low <= truth && truth <= high),
    initial_estimate = initial,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

horizon_lmtp_scenario <- function(frame, scenario, truths, comparator) {
  replicate <- frame$replicate[[1]]
  frame <- horizon_typed(frame[order(frame$row), ])
  # "design": the sample names its comparator (design__sdr), as the parity record's scenarios do.
  if (identical(comparator, "design")) {
    comparator <- if (horizon_constant(frame, "design__sdr") == 1L) "lmtp_sdr" else "lmtp_tmle"
  }
  wanted <- truths[truths$scenario == scenario & truths$replicate == replicate, ]
  if (!nrow(wanted)) stop(sprintf("no truth rows for %s replication %s", scenario, replicate))
  fits <- list()
  get_fit <- function(plan) {
    if (is.null(fits[[plan]])) fits[[plan]] <<- horizon_lmtp_plan(frame, plan, comparator)
    fits[[plan]]
  }
  rows <- lapply(seq_len(nrow(wanted)), function(index) {
    name <- wanted$estimand[[index]]
    parsed <- horizon_parse(name)
    left <- get_fit(parsed$left)
    if (is.null(parsed$right)) {
      return(horizon_row(
        scenario, replicate, name, wanted$truth[[index]],
        left$estimate, left$initial, left$ic, nrow(frame)
      ))
    }
    right <- get_fit(parsed$right)
    horizon_row(
      scenario, replicate, name, wanted$truth[[index]],
      left$estimate - right$estimate, left$initial - right$initial, left$ic - right$ic,
      nrow(frame)
    )
  })
  do.call(rbind, rows)
}

horizon_lmtp_replicate <- function(frame, truths, comparator) {
  # One replicate holds every paired scenario's sample, contiguous in the archive.
  pieces <- split(frame, frame$scenario)
  do.call(rbind, lapply(names(pieces), function(scenario) {
    horizon_lmtp_scenario(pieces[[scenario]], scenario, truths, comparator)
  }))
}
