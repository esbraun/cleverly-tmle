suppressPackageStartupMessages(library(ltmle))
source("/fixture/study_harness.R")
source("/fixture/ltmle_regimen_adapter.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(paths$samples), stringsAsFactors = FALSE, check.names = FALSE)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)

scenario <- "censored_regimen_projection"
dynamic_label <- "treat then continue if l2 positive"
regimens <- c("never", "always", "early", dynamic_label)
duration <- c(never = 0, always = 2, early = 1)
duration[[dynamic_label]] <- 1
projection_weight <- c(never = 0.1, always = 10, early = 0.1)
projection_weight[[dynamic_label]] <- 10
labels <- c("msm_regimen[(intercept)]", "msm_regimen[duration]")
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

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  fits <- setNames(lapply(regimens, function(label) fit_regimen(frame, label)), regimens)
  design <- cbind(1, unname(duration[regimens]))
  weights <- unname(projection_weight[regimens])
  operator <- solve(t(design) %*% (design * weights), t(design * weights))
  estimates <- vapply(fits, `[[`, numeric(1), "estimate")
  initials <- vapply(fits, `[[`, numeric(1), "initial")
  # The projection acts on the joint regimen influence curves, not on four marginal standard
  # errors: every plan is fitted on the same sample, so the marginals would discard exactly the
  # covariance the coefficient's variance is made of.
  influence <- do.call(cbind, lapply(fits, `[[`, "ic")) %*% t(operator)
  beta <- as.numeric(operator %*% estimates)
  initial_beta <- as.numeric(operator %*% initials)
  standard_error <- apply(influence, 2, sd) / sqrt(nrow(frame))
  low <- beta - z * standard_error
  high <- beta + z * standard_error

  selected <- truths$replicate == replicate & truths$scenario == scenario
  truth <- truths$truth[selected][match(labels, truths$estimand[selected])]
  if (any(is.na(truth))) stop(sprintf("truth join failed for replicate %s", replicate))
  data.frame(
    implementation = "ltmle projected regimen fits",
    scenario = scenario,
    replicate = replicate,
    n = nrow(frame),
    estimand = labels,
    truth = truth,
    estimate = beta,
    inference_estimate = beta,
    std_error = standard_error,
    ci_lower = low,
    ci_upper = high,
    inference_scale = "identity",
    covered = as.integer(low <= truth & truth <= high),
    initial_estimate = initial_beta,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

groups <- split(samples, samples$replicate)
expected <- length(groups)
rm(samples)
invisible(gc())

fit_group <- study_fitter(groups, fit_one)
cores <- study_cores(groups)
results <- parallel::mclapply(seq_along(groups), fit_group, mc.cores = cores, mc.preschedule = FALSE)
study_collect(results, expected, paths$output, versions = study_version("ltmle"))
