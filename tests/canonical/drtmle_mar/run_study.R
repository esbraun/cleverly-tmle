suppressPackageStartupMessages(library(drtmle))
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
  fit <- drtmle::drtmle(
    Y = frame$Y,
    A = frame$A,
    W = frame["W"],
    DeltaY = frame$Delta,
    a_0 = c(0, 1),
    family = binomial(),
    Qn = list(frame$qn0, frame$qn1),
    gn = list(frame$gn0, frame$gn1),
    glm_Qr = "gn",
    glm_gr = "Qn",
    guard = c("Q", "g"),
    reduction = "univariate",
    maxIter = 100,
    tolIC = 1e-8,
    tolg = 0.01,
    Qsteps = 2,
    cvFolds = 1,
    se_cv = "none",
    returnModels = FALSE,
    returnNuisance = TRUE,
    use_future = FALSE
  )
  estimates <- as.numeric(fit$drtmle$est)
  covariance <- fit$drtmle$cov
  psi <- c(estimates[[1]], estimates[[2]], estimates[[2]] - estimates[[1]])
  se <- c(
    sqrt(covariance[1, 1]),
    sqrt(covariance[2, 2]),
    sqrt(covariance[1, 1] + covariance[2, 2] - 2 * covariance[1, 2])
  )
  names(psi) <- names(se) <- c("ey0", "ey1", "ate")
  initial <- c(mean(frame$qn0), mean(frame$qn1), mean(frame$qn1 - frame$qn0))
  names(initial) <- names(psi)
  rows <- lapply(names(psi), function(estimand) {
    truth_row <- match(paste(scenario, replicate, estimand, sep = "|"), truth_key)
    if (is.na(truth_row)) stop(sprintf("missing truth for %s/%s/%s", scenario, replicate, estimand))
    truth <- truths$truth[[truth_row]]
    low <- psi[[estimand]] - critical * se[[estimand]]
    high <- psi[[estimand]] + critical * se[[estimand]]
    data.frame(
      implementation = "drtmle-r-mar",
      scenario = scenario,
      replicate = replicate,
      n = nrow(frame),
      estimand = estimand,
      truth = truth,
      estimate = psi[[estimand]],
      inference_estimate = psi[[estimand]],
      std_error = se[[estimand]],
      ci_lower = low,
      ci_upper = high,
      inference_scale = "identity",
      covered = as.integer(low <= truth && truth <= high),
      initial_estimate = initial[[estimand]],
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
    study_version("drtmle"),
    paste("commit", Sys.getenv("DRTMLE_COMMIT"))
  )
)
