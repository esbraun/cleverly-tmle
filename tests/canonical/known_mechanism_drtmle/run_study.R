suppressPackageStartupMessages(library(drtmle))
source("/fixture/study_harness.R")
options(digits = 17)

# R drtmle 1.1.2 with the design's known mechanism as gn, on the samples and the initial
# outcome regression the Python phase wrote.  tolg = 0.1 lies below every declared value, and
# the runner refuses a sample where the floor would bind.  The three-arm scenario passes the
# three levels in a_0, with one Qn and one gn per level, at guard = NULL.

args <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(args$samples), stringsAsFactors = FALSE)
truths <- read.csv(args$truths, stringsAsFactors = FALSE)
truth_key <- paste(truths$scenario, truths$replicate, truths$estimand, sep = "|")
critical <- qnorm(0.975)
tolg <- 0.1
guards <- list(
  guard_none = NULL, guard_q = "Q", guard_g = "g", guard_qg = c("Q", "g"),
  three_arm_guard_none = NULL
)

fit_one <- function(frame) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  if (!scenario %in% names(guards)) stop(sprintf("unknown scenario %s", scenario))
  arms <- if (scenario == "three_arm_guard_none") 3 else 2
  levels <- seq_len(arms) - 1
  qn <- lapply(levels, function(code) frame[[sprintf("qn%d", code)]])
  gn <- lapply(levels, function(code) frame[[sprintf("gn%d", code)]])
  if (any(unlist(gn) <= tolg)) {
    stop(sprintf("the tolg floor would bind for %s/%s", scenario, replicate))
  }
  fit <- drtmle::drtmle(
    Y = frame$Y,
    A = frame$A,
    W = frame[c("W1", "W2", "W3")],
    a_0 = levels,
    family = binomial(),
    Qn = qn,
    gn = gn,
    glm_Qr = "gn",
    glm_gr = "Qn",
    guard = guards[[scenario]],
    reduction = "univariate",
    maxIter = 100,
    tolIC = 1e-8,
    tolg = tolg,
    Qsteps = 2,
    cvFolds = 1,
    se_cv = "none",
    returnModels = FALSE,
    returnNuisance = FALSE,
    use_future = FALSE
  )
  estimates <- as.numeric(fit$drtmle$est)
  covariance <- fit$drtmle$cov
  means <- sapply(qn, mean)
  contrast_se <- function(a) {
    sqrt(covariance[1, 1] + covariance[a, a] - 2 * covariance[1, a])
  }
  if (arms == 2) {
    psi <- c(estimates[[1]], estimates[[2]], estimates[[2]] - estimates[[1]])
    se <- c(sqrt(covariance[1, 1]), sqrt(covariance[2, 2]), contrast_se(2))
    initial <- c(means[[1]], means[[2]], means[[2]] - means[[1]])
    names(psi) <- names(se) <- names(initial) <- c("ey0", "ey1", "ate")
  } else {
    psi <- c(estimates, estimates[[2]] - estimates[[1]], estimates[[3]] - estimates[[1]])
    se <- c(sqrt(diag(covariance)), contrast_se(2), contrast_se(3))
    initial <- c(means, means[[2]] - means[[1]], means[[3]] - means[[1]])
    names(psi) <- names(se) <- names(initial) <- c(
      "ey[0.0]", "ey[1.0]", "ey[2.0]", "ate[1.0 vs 0.0]", "ate[2.0 vs 0.0]"
    )
  }
  rows <- lapply(names(psi), function(estimand) {
    truth_row <- match(paste(scenario, replicate, estimand, sep = "|"), truth_key)
    if (is.na(truth_row)) stop(sprintf("missing truth for %s/%s/%s", scenario, replicate, estimand))
    truth <- truths$truth[[truth_row]]
    low <- psi[[estimand]] - critical * se[[estimand]]
    high <- psi[[estimand]] + critical * se[[estimand]]
    data.frame(
      implementation = "drtmle-r-known-g",
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
  ),
  key = c("scenario", "replicate")
)
