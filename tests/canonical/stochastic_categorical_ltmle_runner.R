suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_policy_adapter.R")
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments(
  "usage: stochastic_categorical_ltmle_runner.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv"
)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)

scenario <- "categorical_policy_end_of_study"
plans <- c("low", "mix", "taper")
arms <- c("standard", "high", "low")
copies <- 4L
z <- qnorm(0.975)

# The law's mechanism, transcribed as in categorical_ltmle_runner.R.
g1 <- data.frame(
  W = rep(c(0, 1), each = 3),
  arm = rep(c("standard", "high", "low"), 2),
  probability = c(0.25, 0.25, 0.50, 0.50, 0.25, 0.25),
  stringsAsFactors = FALSE
)

g2 <- data.frame(
  W = rep(c(0, 1), each = 6),
  A1 = rep(rep(c("standard", "high", "low"), each = 2), 2),
  L2 = rep(c(0, 1), 6),
  standard = c(0.25, 0.50, 0.25, 0.25, 0.50, 0.25, 0.25, 0.50, 0.50, 0.25, 0.25, 0.25),
  high = c(0.25, 0.25, 0.50, 0.25, 0.25, 0.50, 0.50, 0.25, 0.25, 0.25, 0.25, 0.50),
  low = c(0.50, 0.25, 0.25, 0.50, 0.25, 0.25, 0.25, 0.25, 0.25, 0.50, 0.50, 0.25),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

# The declared policies, transcribed from tests/discrete_law_longitudinal_policy.py
# (POLICY1 and POLICY2) by label.  A gate test reads these vectors back.
q1 <- data.frame(
  W = c(0, 1),
  q_standard = c(0.25, 0.25),
  q_high = c(0.50, 0.50),
  q_low = c(0.25, 0.25),
  stringsAsFactors = FALSE
)

q2 <- data.frame(
  W = rep(c(0, 1), each = 6),
  A1 = rep(rep(c("standard", "high", "low"), each = 2), 2),
  L2 = rep(c(0, 1), 6),
  q_standard = c(0.50, 0.25, 0.25, 0.50, 0.25, 0.25, 0.25, 0.25, 0.50, 0.25, 0.25, 0.50),
  q_high = c(0.25, 0.25, 0.50, 0.25, 0.25, 0.50, 0.25, 0.50, 0.25, 0.25, 0.50, 0.25),
  q_low = c(0.25, 0.50, 0.25, 0.25, 0.50, 0.25, 0.50, 0.25, 0.25, 0.50, 0.25, 0.25),
  stringsAsFactors = FALSE
)

lookup_g1 <- function(w, arm) {
  key <- paste(g1$W, g1$arm, sep = "|")
  value <- g1$probability[match(paste(w, arm, sep = "|"), key)]
  if (anyNA(value)) stop("the first-node mechanism lookup failed")
  value
}

lookup_g2 <- function(w, a1, l2, a2) {
  key <- paste(g2$W, g2$A1, g2$L2, sep = "|")
  row <- match(paste(w, a1, l2, sep = "|"), key)
  if (anyNA(row)) stop("the second-node mechanism lookup failed")
  value <- vapply(seq_along(row), function(i) g2[row[[i]], a2[[i]]], numeric(1))
  if (anyNA(value)) stop("the second-arm probability is unavailable")
  value
}

policy_rows <- function(table, frame, node) {
  # The policy row of each unit, an n x 3 matrix in `arms` order.
  if (node == 1) {
    row <- match(frame$W, table$W)
  } else {
    key <- paste(table$W, table$A1, table$L2, sep = "|")
    row <- match(paste(frame$W, frame$A1, frame$L2, sep = "|"), key)
  }
  if (anyNA(row)) stop(sprintf("the node %d policy lookup failed", node))
  as.matrix(table[row, paste0("q_", arms)])
}

one_hot <- function(frame, label) {
  matrix(rep(as.numeric(arms == label), each = nrow(frame)), nrow = nrow(frame))
}

node_density <- function(frame, label, node) {
  # The plan's intervention density at node `node`, an n x 3 matrix in `arms` order.
  if (label == "mix") return(policy_rows(if (node == 1) q1 else q2, frame, node))
  if (label == "taper") {
    return(if (node == 1) one_hot(frame, "high") else policy_rows(q2, frame, 2))
  }
  stop(sprintf("plan %s has no policy node", label))
}

at_observed <- function(density, observed) {
  density[cbind(seq_len(nrow(density)), match(observed, arms))]
}

fit_policy <- function(frame, label) {
  natural <- frame[c("W", "A1", "L2", "A2", "Y")]
  first <- node_density(frame, label, 1)
  second <- node_density(frame, label, 2)
  draws <- lapply(list(first, second), function(density) {
    t(apply(density, 1, allocate_copies, arms = arms, copies = copies))
  })
  density_at_observed <- cbind(at_observed(first, frame$A1), at_observed(second, frame$A2))
  ratios <- cbind(
    density_at_observed[, 1] / lookup_g1(frame$W, frame$A1),
    density_at_observed[, 2] / lookup_g2(frame$W, frame$A1, frame$L2, frame$A2)
  )
  lmtp_policy_tmle(
    natural,
    draws,
    ratios,
    density_at_observed,
    trt = c("A1", "A2"),
    outcome = "Y",
    baseline = "W",
    time_vary = list(NULL, "L2"),
    fold_assignment = frame$fold,
    copies = copies,
    learners_outcome = "SL.glm",
    control = lmtp_control(.trim = 1, .learners_outcome_folds = 2, .return_full_fits = TRUE)
  )
}

fit_static <- function(frame, label) {
  # The deterministic reference plan, fitted exactly as categorical_ltmle_runner.R fits it.
  natural <- frame[c("W", "A1", "L2", "A2", "Y")]
  shifted <- natural
  shifted$A1 <- label
  shifted$A2 <- label
  assigned <- rep(label, nrow(frame))
  ratios <- cbind(
    ifelse(frame$A1 == label, 1 / lookup_g1(frame$W, assigned), 0),
    ifelse(frame$A2 == label, 1 / lookup_g2(frame$W, assigned, frame$L2, assigned), 0)
  )
  fit <- lmtp_tmle_with_folds(
    natural,
    shifted,
    trt = c("A1", "A2"),
    outcome = "Y",
    baseline = "W",
    time_vary = list(NULL, "L2"),
    outcome_type = "binomial",
    fold_assignment = frame$fold,
    learners_outcome = "SL.glm",
    learners_trt = "SL.glm",
    density_ratios = ratios,
    control = lmtp_control(
      .trim = 1,
      .learners_outcome_folds = 2,
      .learners_trt_folds = 2,
      .return_full_fits = TRUE
    )
  )
  list(estimate = fit$estimate@x, initial = mean(fit$initial), ic = fit$estimate@eif)
}

fit_plan <- function(frame, label) {
  if (label == "low") fit_static(frame, label) else fit_policy(frame, label)
}

truth_for <- function(replicate, estimand) {
  selected <- truths$replicate == replicate & truths$scenario == scenario & truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows", sum(selected)))
  truths$truth[selected]
}

row_for <- function(replicate, name, estimate, initial, ic, n) {
  truth <- truth_for(replicate, name)
  standard_error <- sd(ic) / sqrt(n)
  low <- estimate - z * standard_error
  high <- estimate + z * standard_error
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

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  frame <- frame[order(frame$row), ]
  fits <- setNames(lapply(plans, function(label) fit_plan(frame, label)), plans)
  rows <- lapply(plans, function(label) {
    fit <- fits[[label]]
    row_for(replicate, sprintf("ey_regimen[%s]", label), fit$estimate, fit$initial, fit$ic, nrow(frame))
  })
  for (label in setdiff(plans, "low")) {
    left <- fits[[label]]
    right <- fits[["low"]]
    rows[[length(rows) + 1]] <- row_for(
      replicate,
      sprintf("ate_regimen[%s vs low]", label),
      left$estimate - right$estimate,
      left$initial - right$initial,
      left$ic - right$ic,
      nrow(frame)
    )
  }
  do.call(rbind, rows)
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(results, expected, paths$output, versions = study_version("lmtp"))
