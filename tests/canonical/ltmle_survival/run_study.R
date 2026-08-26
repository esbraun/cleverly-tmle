suppressPackageStartupMessages(library(ltmle))
source("/fixture/study_harness.R")
source("/fixture/ltmle_regimen_adapter.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(paths$samples), stringsAsFactors = FALSE, check.names = FALSE)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)

scenario <- "censored_survival_curve"
dynamic_label <- "treat then continue if l2 positive"
z <- qnorm(0.975)

fit_regimen <- function(frame, label, horizon) {
  abar <- regimen_arms(frame, label, horizon)
  arguments <- if (horizon == 1) {
    list(
      data = frame[c("W1", "W2", "A1", "C1", "Y1")],
      Anodes = "A1",
      Cnodes = "C1",
      Ynodes = "Y1",
      survivalOutcome = TRUE,
      Qform = c(Y1 = "Q.kplus1 ~ W1 + W2")
    )
  } else {
    list(
      data = frame[c("W1", "W2", "A1", "C1", "Y1", "L2", "A2", "C2", "Y2")],
      Anodes = c("A1", "A2"),
      Cnodes = c("C1", "C2"),
      Lnodes = "L2",
      Ynodes = c("Y1", "Y2"),
      survivalOutcome = TRUE,
      Qform = c(Y1 = "Q.kplus1 ~ W1 + W2", Y2 = "Q.kplus1 ~ W1 + W2 + L2")
    )
  }
  arguments <- c(
    arguments,
    list(
      gform = regimen_mechanism(frame, abar, horizon),
      abar = abar,
      gbounds = c(1e-8, 1),
      SL.library = "glm",
      stratify = TRUE,
      variance.method = "ic"
    )
  )
  regimen_ltmle_fit(arguments, frame, sprintf("%s at horizon %d", label, horizon))
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

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  fits <- list(
    "never @ t=1" = fit_regimen(frame, "never", 1),
    "always @ t=1" = fit_regimen(frame, "always", 1),
    "never @ t=2" = fit_regimen(frame, "never", 2),
    "always @ t=2" = fit_regimen(frame, "always", 2),
    "dynamic @ t=2" = fit_regimen(frame, dynamic_label, 2)
  )
  labels <- list(
    c("never", "1", "never @ t=1"),
    c("never", "2", "never @ t=2"),
    c("always", "1", "always @ t=1"),
    c("always", "2", "always @ t=2"),
    c(dynamic_label, "2", "dynamic @ t=2")
  )
  rows <- lapply(labels, function(spec) {
    fit <- fits[[spec[[3]]]]
    name <- sprintf("risk_regimen[%s @ t=%s]", spec[[1]], spec[[2]])
    row_for(replicate, name, fit$estimate, fit$initial, fit$ic, nrow(frame))
  })
  comparisons <- list(
    c("always", "1", "always @ t=1", "never @ t=1"),
    c("always", "2", "always @ t=2", "never @ t=2"),
    c(dynamic_label, "2", "dynamic @ t=2", "never @ t=2")
  )
  for (spec in comparisons) {
    left <- fits[[spec[[3]]]]
    right <- fits[[spec[[4]]]]
    name <- sprintf("ate_regimen[%s vs never @ t=%s]", spec[[1]], spec[[2]])
    rows[[length(rows) + 1]] <- row_for(
      replicate,
      name,
      left$estimate - right$estimate,
      left$initial - right$initial,
      left$ic - right$ic,
      nrow(frame)
    )
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
