suppressPackageStartupMessages(library(ltmle))
source("/fixture/study_harness.R")
source("/fixture/ltmle_regimen_adapter.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(paths$samples), stringsAsFactors = FALSE, check.names = FALSE)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)

scenario <- "selected_censored_end_of_study"
dynamic_label <- "treat then continue if l2 positive"
regimens <- c("never", "always", dynamic_label)
qform <- c(L2 = "Q.kplus1 ~ W1 + W2", Y = "Q.kplus1 ~ W1 + W2 + L2")
z <- qnorm(0.975)

fit_regimen <- function(frame, label) {
  abar <- regimen_arms(frame, label)
  arguments <- list(
    data = frame[c("W1", "W2", "A1", "C1", "L2", "A2", "C2", "Y")],
    Anodes = c("A1", "A2"), Cnodes = c("C1", "C2"), Lnodes = "L2", Ynodes = "Y",
    survivalOutcome = FALSE, Qform = qform, gform = regimen_mechanism(frame, abar),
    abar = abar, gbounds = c(1e-8, 1), SL.library = "glm", stratify = TRUE,
    variance.method = "ic", observation.weights = frame$obs_weight
  )
  fit <- regimen_ltmle_fit(arguments, frame, label)
  computed_se <- sd(fit$ic) / sqrt(nrow(frame))
  if (!isTRUE(all.equal(fit$native_standard_error, computed_se, tolerance = 1e-12))) {
    stop(sprintf(
      "ltmle native standard error %.17g does not equal influence-curve formula %.17g",
      fit$native_standard_error, computed_se
    ))
  }
  fit
}

truth_for <- function(replicate, estimand) {
  selected <- truths$replicate == replicate & truths$scenario == scenario & truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows", sum(selected)))
  truths$truth[selected]
}

row_for <- function(replicate, name, estimate, initial, standard_error, n) {
  truth <- truth_for(replicate, name)
  low <- estimate - z * standard_error
  high <- estimate + z * standard_error
  data.frame(
    implementation = "ltmle-weighted", scenario = scenario, replicate = replicate, n = n,
    estimand = name, truth = truth, estimate = estimate, inference_estimate = estimate,
    std_error = standard_error, ci_lower = low, ci_upper = high,
    inference_scale = "identity", covered = as.integer(low <= truth && truth <= high),
    initial_estimate = initial, stringsAsFactors = FALSE, check.names = FALSE
  )
}

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  fits <- setNames(lapply(regimens, function(label) fit_regimen(frame, label)), regimens)
  rows <- lapply(regimens, function(label) {
    fit <- fits[[label]]
    row_for(
      replicate, sprintf("ey_regimen[%s]", label), fit$estimate, fit$initial,
      fit$native_standard_error, nrow(frame)
    )
  })
  for (label in c("always", dynamic_label)) {
    left <- fits[[label]]
    right <- fits[["never"]]
    rows[[length(rows) + 1]] <- row_for(
      replicate, sprintf("ate_regimen[%s vs never]", label), left$estimate - right$estimate,
      left$initial - right$initial, sd(left$ic - right$ic) / sqrt(nrow(frame)), nrow(frame)
    )
  }
  do.call(rbind, rows)
}

groups <- split(samples, samples$replicate)
expected <- length(groups)
rm(samples)
invisible(gc())
fit_group <- study_fitter(groups, fit_one)
results <- parallel::mclapply(
  seq_along(groups), fit_group, mc.cores = study_cores(groups), mc.preschedule = FALSE
)
study_collect(results, expected, paths$output, versions = study_version("ltmle"))
