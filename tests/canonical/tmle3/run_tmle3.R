suppressPackageStartupMessages({
  library(sl3)
  library(tmle3)
})
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments("usage: run_tmle3.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")

# The truth arrives once per replication rather than repeated on every row.  Carrying ten
# constant columns through 3.2 million rows was most of this process's memory, and memory is
# what decides how many workers fit.
samples <- read.csv(gzfile(paths$samples), stringsAsFactors = FALSE)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE)
truth_key <- paste(truths$scenario, truths$replicate, sep = "|")
truth_columns <- grep("^truth_", names(truths), value = TRUE)
learners <- list(A = sl3::Lrnr_glm$new(), Y = sl3::Lrnr_glm$new())

# The pinned package's ATT and ATC tests use this explicit constrained updater.
# The public ATT convenience updater reaches a non-finite convergence state for
# a small fraction of bounded-continuous samples, so both conditional estimands
# follow their package-authored test path.
fit_conditional <- function(spec, data, nodes, learners) {
  task <- spec$make_tmle_task(data, nodes)
  initial <- spec$make_initial_likelihood(task, learners)
  updater <- tmle3_Update$new(
    cvtmle = FALSE,
    convergence_type = "sample_size",
    constrain_step = TRUE,
    one_dimensional = TRUE,
    delta_epsilon = 0.001,
    optim_delta_epsilon = FALSE,
    maxit = 200
  )
  targeted <- Targeted_Likelihood$new(initial, updater)
  parameters <- spec$make_params(task, targeted)
  updater$tmle_params <- parameters
  fit_tmle3(task, targeted, parameters, updater)
}

extract <- function(fit, labels, scenario, replicate, truth, n) {
  tab <- as.data.frame(fit$summary)
  stopifnot(nrow(tab) == length(labels))
  transform_scale <- labels %in% c("rr", "or", "paf")
  data.frame(
    implementation = "tmle3",
    scenario = scenario,
    replicate = replicate,
    n = n,
    estimand = labels,
    truth = unname(truth[labels]),
    estimate = tab$psi_transformed,
    inference_estimate = tab$tmle_est,
    std_error = tab$se,
    ci_lower = tab$lower_transformed,
    ci_upper = tab$upper_transformed,
    inference_scale = ifelse(
      labels == "paf",
      "negative_log_complement",
      ifelse(transform_scale, "log", "identity")
    ),
    covered = as.integer(tab$lower_transformed <= unname(truth[labels]) &
      unname(truth[labels]) <= tab$upper_transformed),
    initial_estimate = tab$init_est,
    stringsAsFactors = FALSE
  )
}

fit_one <- function(frame, scenario, replicate) {
  row <- match(paste(scenario, replicate, sep = "|"), truth_key)
  if (is.na(row)) stop(sprintf("no truth for %s replicate %s", scenario, replicate))
  truth <- as.numeric(truths[row, truth_columns])
  names(truth) <- sub("^truth_", "", truth_columns)
  covariates <- grep("^W", names(frame), value = TRUE)
  covariates <- covariates[!vapply(frame[covariates], function(x) all(is.na(x)), logical(1))]
  data <- frame[c(covariates, "A", "Y")]
  nodes <- list(W = covariates, A = "A", Y = "Y")
  n <- nrow(data)
  rows <- list()
  stage <- function(label, expression) {
    tryCatch(
      force(expression),
      error = function(condition) {
        stop(sprintf("%s: %s", label, conditionMessage(condition)), call. = FALSE)
      }
    )
  }

  tsm <- stage("TSM", tmle3(tmle_TSM_all(), data, nodes, learners))
  rows[[length(rows) + 1]] <- extract(
    tsm, c("ey0", "ey1"), scenario, replicate, truth, n
  )
  rows[[length(rows) + 1]] <- extract(
    stage("ATE", tmle3(tmle_ATE(1, 0), data, nodes, learners)),
    "ate", scenario, replicate, truth, n
  )
  rows[[length(rows) + 1]] <- extract(
    stage("ATT", fit_conditional(tmle_ATT(1, 0), data, nodes, learners)),
    "att", scenario, replicate, truth, n
  )
  rows[[length(rows) + 1]] <- extract(
    stage("ATC", fit_conditional(tmle_ATC(1, 0), data, nodes, learners)),
    "atc", scenario, replicate, truth, n
  )
  par <- stage("PAR", tmle3(tmle_PAR(0), data, nodes, learners))
  # tmle_PAR always reports these four; the continuous law has no PAF, so its row is
  # dropped below rather than requested differently.
  par_rows <- extract(
    par, c("ey0", "ey_obs", "par", "paf"), scenario, replicate, truth, n
  )
  if (scenario != "binary") par_rows <- par_rows[par_rows$estimand != "paf", ]
  rows[[length(rows) + 1]] <- par_rows[par_rows$estimand != "ey0", ]

  if (scenario == "binary") {
    rr <- stage("RR", tmle3(tmle_RR(0, 1), data, nodes, learners))
    rows[[length(rows) + 1]] <- extract(
      rr, c("ey0", "ey1", "rr"), scenario, replicate, truth, n
    )[3, ]
    or <- stage("OR", tmle3(tmle_OR(0, 1), data, nodes, learners))
    rows[[length(rows) + 1]] <- extract(
      or, c("ey0", "ey1", "or"), scenario, replicate, truth, n
    )[3, ]
  }
  do.call(rbind, rows)
}

groups <- split(samples, interaction(samples$scenario, samples$replicate, drop = TRUE))
expected <- length(groups)
rm(samples)
invisible(gc())

fit_group <- study_fitter(
  groups,
  function(frame) fit_one(frame, frame$scenario[[1]], frame$replicate[[1]])
)
cores <- study_cores(groups)
results <- parallel::mclapply(seq_along(groups), fit_group, mc.cores = cores, mc.preschedule = FALSE)
study_collect(
  results,
  expected,
  paths$output,
  versions = c(study_version("tmle3"), study_version("sl3")),
  key = c("scenario", "replicate"),
  na = "NA"
)
