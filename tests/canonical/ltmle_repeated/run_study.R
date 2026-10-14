suppressPackageStartupMessages(library(ltmle))
source("/fixture/study_harness.R")
source("/fixture/ltmle_repeated_adapter.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(paths$samples), stringsAsFactors = FALSE, check.names = FALSE)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)

scenario <- "censored_repeated_outcome"
rule <- repeated_rule_label
z <- qnorm(0.975)

# One main-terms regression per L/Y block, on every node before the block's first node except
# the arms and censoring indicators: a stratified fit holds those constant.  The blocks are
# {Y1, L2}, {Y2, L3} and {Y3}, and each formula reads the history cleverly regresses on at that
# node, [W, Y1, L2, ..., Y_{t-1}, L_t].
q_forms <- c(
  Y1 = "Q.kplus1 ~ W",
  Y2 = "Q.kplus1 ~ W + Y1 + L2",
  Y3 = "Q.kplus1 ~ W + Y1 + L2 + Y2 + L3"
)

fit_plan <- function(frame, label, horizon) {
  # The prefix through Y_h.  survivalOutcome = FALSE makes Y_h the outcome and every earlier
  # Y_s a covariate, which is the repeated outcome's reduction to the truncated panel.
  nodes <- c("W")
  for (t in seq_len(horizon)) {
    if (t > 1) nodes <- c(nodes, sprintf("L%d", t))
    nodes <- c(nodes, sprintf("A%d", t), sprintf("C%d", t), sprintf("Y%d", t))
  }
  abar <- repeated_arms(frame, label, horizon)
  arguments <- list(
    data = frame[nodes],
    Anodes = sprintf("A%d", seq_len(horizon)),
    Cnodes = sprintf("C%d", seq_len(horizon)),
    Ynodes = sprintf("Y%d", seq_len(horizon)),
    survivalOutcome = FALSE,
    Qform = q_forms[seq_len(horizon)],
    gform = repeated_mechanism(frame, abar, horizon),
    abar = if (horizon == 1) matrix(abar, ncol = 1) else abar,
    gbounds = c(1e-8, 1),
    SL.library = "glm",
    stratify = TRUE,
    variance.method = "ic"
  )
  if (horizon > 1) arguments$Lnodes <- sprintf("L%d", 2:horizon)
  repeated_ltmle_fit(arguments, sprintf("%s at horizon %d", label, horizon))
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
    implementation = "ltmle",
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

# The rule equals `always` at the first node, so its t = 1 mean and contrast are not primary
# cells and are not fitted.
plans <- list(
  list(label = "never", horizons = 1:3),
  list(label = "always", horizons = 1:3),
  list(label = rule, horizons = 2:3)
)

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  n <- nrow(frame)
  fits <- list()
  for (plan in plans) {
    for (horizon in plan$horizons) {
      fits[[sprintf("%s @ t=%d", plan$label, horizon)]] <- fit_plan(frame, plan$label, horizon)
    }
  }
  rows <- list()
  for (plan in plans) {
    for (horizon in plan$horizons) {
      key <- sprintf("%s @ t=%d", plan$label, horizon)
      fit <- fits[[key]]
      rows[[length(rows) + 1]] <- row_for(
        replicate, sprintf("ey_regimen[%s]", key), fit$estimate, fit$initial, fit$ic, n
      )
    }
  }
  for (plan in plans[2:3]) {
    for (horizon in plan$horizons) {
      left <- fits[[sprintf("%s @ t=%d", plan$label, horizon)]]
      right <- fits[[sprintf("never @ t=%d", horizon)]]
      rows[[length(rows) + 1]] <- row_for(
        replicate,
        sprintf("ate_regimen[%s vs never @ t=%d]", plan$label, horizon),
        left$estimate - right$estimate,
        left$initial - right$initial,
        left$ic - right$ic,
        n
      )
    }
  }
  do.call(rbind, rows)
}

groups <- split(samples, samples$replicate)
expected <- length(groups)
rm(samples)
invisible(gc())

fit_group <- study_fitter(groups, fit_one)
cores <- study_cores(groups)
results <- parallel::mclapply(seq_along(groups), fit_group, mc.cores = cores, mc.preschedule = FALSE)
study_collect(results, expected, paths$output, versions = study_version("ltmle"))
