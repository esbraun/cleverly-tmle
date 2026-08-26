suppressPackageStartupMessages(library(ltmle))
source("/fixture/study_harness.R")
source("/fixture/ltmle_regimen_adapter.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(paths$samples), stringsAsFactors = FALSE, check.names = FALSE)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)

scenario <- "censored_end_of_study"
dynamic_label <- "treat then continue if l2 positive"
regimens <- c("never", "always", dynamic_label)
qform <- c(L2 = "Q.kplus1 ~ W1 + W2", Y = "Q.kplus1 ~ W1 + W2 + L2")
z <- qnorm(0.975)

fit_regimen <- function(frame, label) {
  abar <- regimen_arms(frame, label)
  arguments <- list(
    data = frame[c("W1", "W2", "A1", "C1", "L2", "A2", "C2", "Y")],
    Anodes = c("A1", "A2"),
    Cnodes = c("C1", "C2"),
    Lnodes = "L2",
    Ynodes = "Y",
    survivalOutcome = FALSE,
    Qform = qform,
    gform = regimen_mechanism(frame, abar),
    abar = abar,
    gbounds = c(1e-8, 1),
    SL.library = "glm",
    stratify = TRUE,
    variance.method = "ic"
  )
  regimen_ltmle_fit(arguments, frame, label)
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
  fits <- setNames(lapply(regimens, function(label) fit_regimen(frame, label)), regimens)
  n <- nrow(frame)
  rows <- lapply(regimens, function(label) {
    name <- sprintf("ey_regimen[%s]", label)
    fit <- fits[[label]]
    row_for(replicate, name, fit$estimate, fit$initial, fit$ic, n)
  })
  for (label in c("always", dynamic_label)) {
    name <- sprintf("ate_regimen[%s vs never]", label)
    fit <- fits[[label]]
    reference <- fits[["never"]]
    rows[[length(rows) + 1]] <- row_for(
      replicate,
      name,
      fit$estimate - reference$estimate,
      fit$initial - reference$initial,
      fit$ic - reference$ic,
      n
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
