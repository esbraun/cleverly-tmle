suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_crossfit_adapter.R")
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments("usage: categorical_ltmle_runner.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)

scenario <- "categorical_end_of_study"
plans <- c("low", "standard", "high", "step_down", "respond")
z <- qnorm(0.975)

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

plan_arms <- function(frame, label) {
  if (label == "low") return(cbind(rep("low", nrow(frame)), rep("low", nrow(frame))))
  if (label == "standard") {
    return(cbind(rep("standard", nrow(frame)), rep("standard", nrow(frame))))
  }
  if (label == "high") return(cbind(rep("high", nrow(frame)), rep("high", nrow(frame))))
  if (label == "step_down") {
    return(cbind(rep("high", nrow(frame)), rep("standard", nrow(frame))))
  }
  if (label == "respond") {
    return(cbind(rep("standard", nrow(frame)), ifelse(frame$L2 == 1, "high", "low")))
  }
  stop(sprintf("unknown categorical plan %s", label))
}

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
  if (anyNA(value)) stop("the assigned second-arm probability is unavailable")
  value
}

exact_ratios <- function(frame, arms) {
  first_probability <- lookup_g1(frame$W, arms[, 1])
  second_probability <- lookup_g2(frame$W, arms[, 1], frame$L2, arms[, 2])
  cbind(
    ifelse(frame$A1 == arms[, 1], 1 / first_probability, 0),
    ifelse(frame$A2 == arms[, 2], 1 / second_probability, 0)
  )
}

fit_plan <- function(frame, label) {
  natural <- frame[c("W", "A1", "L2", "A2", "Y")]
  shifted <- natural
  arms <- plan_arms(frame, label)
  shifted$A1 <- arms[, 1]
  shifted$A2 <- arms[, 2]
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
    density_ratios = exact_ratios(frame, arms),
    control = lmtp_control(
      .trim = 1,
      .learners_outcome_folds = 2,
      .learners_trt_folds = 2,
      .return_full_fits = TRUE
    )
  )
  if (!identical(fit$fold_assignment, as.integer(frame$fold))) {
    stop("lmtp did not retain the supplied fold assignment")
  }
  list(estimate = fit$estimate@x, initial = mean(fit$initial), ic = fit$estimate@eif)
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
  fits <- setNames(lapply(plans, function(label) fit_plan(frame, label)), plans)
  rows <- lapply(plans, function(label) {
    fit <- fits[[label]]
    row_for(
      replicate,
      sprintf("ey_regimen[%s]", label),
      fit$estimate,
      fit$initial,
      fit$ic,
      nrow(frame)
    )
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
