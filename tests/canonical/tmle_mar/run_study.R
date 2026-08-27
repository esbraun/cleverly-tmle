suppressPackageStartupMessages(library(tmle))
source("/fixture/study_harness.R")
options(digits = 17)

args <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(args$samples), stringsAsFactors = FALSE)
truths <- read.csv(args$truths, stringsAsFactors = FALSE)
truth_key <- paste(truths$scenario, truths$replicate, truths$estimand, sep = "|")
critical <- qnorm(0.975)

fit_one <- function(frame) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  qn <- cbind(frame$qn0, frame$qn1)
  pin <- cbind(frame$pin0, frame$pin1)
  fit <- tmle::tmle(
    Y = frame$Y,
    A = frame$A,
    W = frame["W"],
    Delta = frame$Delta,
    Q = qn,
    g1W = frame$gn1,
    pDelta1 = pin,
    family = "binomial",
    fluctuation = "logistic",
    Qbounds = c(0.001, 0.999),
    gbound = c(0.01, 0.99),
    cvQinit = FALSE,
    verbose = FALSE
  )
  values <- list(
    ey0 = fit$estimates$EY0,
    ey1 = fit$estimates$EY1,
    ate = fit$estimates$ATE
  )
  rows <- lapply(names(values), function(estimand) {
    value <- values[[estimand]]
    truth_row <- match(paste(scenario, replicate, estimand, sep = "|"), truth_key)
    if (is.na(truth_row)) stop(sprintf("missing truth for %s/%s/%s", scenario, replicate, estimand))
    truth <- truths$truth[[truth_row]]
    estimate <- as.numeric(value$psi)
    standard_error <- sqrt(as.numeric(value$var.psi))
    low <- estimate - critical * standard_error
    high <- estimate + critical * standard_error
    initial <- switch(
      estimand,
      ey0 = mean(frame$qn0),
      ey1 = mean(frame$qn1),
      ate = mean(frame$qn1 - frame$qn0)
    )
    data.frame(
      implementation = "tmle-r",
      scenario = scenario,
      replicate = replicate,
      n = nrow(frame),
      estimand = estimand,
      truth = truth,
      estimate = estimate,
      inference_estimate = estimate,
      std_error = standard_error,
      ci_lower = low,
      ci_upper = high,
      inference_scale = "identity",
      covered = as.integer(low <= truth && truth <= high),
      initial_estimate = initial,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

groups <- split(samples, interaction(samples$scenario, samples$replicate, drop = TRUE))
cores <- study_cores(groups)
results <- parallel::mclapply(
  seq_along(groups),
  study_fitter(groups, fit_one),
  mc.cores = cores,
  mc.preschedule = TRUE
)
study_collect(
  results,
  expected = length(groups),
  output = args$output,
  versions = c(
    study_version("tmle"),
    paste("source sha256", Sys.getenv("TMLE_SHA256"))
  )
)
