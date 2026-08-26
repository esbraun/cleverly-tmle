suppressPackageStartupMessages({
  library(origami)
  library(sl3)
  library(tmle3)
})
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments("usage: run_tmle3_cvtmle.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(paths$samples), stringsAsFactors = FALSE)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE)
truth_key <- paste(truths$scenario, truths$replicate, sep = "|")
truth_columns <- grep("^truth_", names(truths), value = TRUE)

exact_folds <- function(assignment) {
  labels <- sort(unique(assignment))
  stopifnot(identical(labels, seq.int(0, length(labels) - 1)))
  lapply(seq_along(labels), function(v) {
    validation <- which(assignment == labels[[v]])
    training <- which(assignment != labels[[v]])
    origami::make_fold(v = v, training_set = training, validation_set = validation)
  })
}

fit_cv <- function(spec, data, nodes, folds, conditional = FALSE) {
  task <- point_tx_task(data, nodes, folds = folds)
  stopifnot(all(vapply(seq_along(folds), function(v) {
    identical(task$folds[[v]]$validation_set, folds[[v]]$validation_set)
  }, logical(1))))
  learners <- list(
    A = Lrnr_cv$new(Lrnr_glm$new(), folds = folds, full_fit = TRUE),
    Y = Lrnr_cv$new(Lrnr_glm$new(), folds = folds, full_fit = TRUE)
  )
  initial <- spec$make_initial_likelihood(task, learners)
  updater <- tmle3_Update$new(
    cvtmle = TRUE,
    convergence_type = "sample_size",
    constrain_step = conditional,
    one_dimensional = conditional,
    delta_epsilon = 0.001,
    optim_delta_epsilon = !conditional,
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
    implementation = "tmle3-cvtmle",
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
  folds <- exact_folds(frame$fold)
  covariates <- grep("^W", names(frame), value = TRUE)
  covariates <- covariates[!vapply(frame[covariates], function(x) all(is.na(x)), logical(1))]
  data <- frame[c(covariates, "A", "Y")]
  nodes <- list(W = covariates, A = "A", Y = "Y")
  n <- nrow(data)
  rows <- list()
  stage <- function(label, expression) {
    tryCatch(force(expression), error = function(condition) {
      stop(sprintf("%s: %s", label, conditionMessage(condition)), call. = FALSE)
    })
  }

  tsm <- stage("TSM", fit_cv(tmle_TSM_all(), data, nodes, folds))
  rows[[length(rows) + 1]] <- extract(tsm, c("ey0", "ey1"), scenario, replicate, truth, n)
  rows[[length(rows) + 1]] <- extract(
    stage("ATE", fit_cv(tmle_ATE(1, 0), data, nodes, folds)),
    "ate", scenario, replicate, truth, n
  )
  rows[[length(rows) + 1]] <- extract(
    stage("ATT", fit_cv(tmle_ATT(1, 0), data, nodes, folds, conditional = TRUE)),
    "att", scenario, replicate, truth, n
  )
  rows[[length(rows) + 1]] <- extract(
    stage("ATC", fit_cv(tmle_ATC(1, 0), data, nodes, folds, conditional = TRUE)),
    "atc", scenario, replicate, truth, n
  )
  par <- stage("PAR", fit_cv(tmle_PAR(0), data, nodes, folds))
  par_rows <- extract(par, c("ey0", "ey_obs", "par", "paf"), scenario, replicate, truth, n)
  if (scenario != "binary") par_rows <- par_rows[par_rows$estimand != "paf", ]
  rows[[length(rows) + 1]] <- par_rows[par_rows$estimand != "ey0", ]

  if (scenario == "binary") {
    rows[[length(rows) + 1]] <- extract(
      stage("RR", fit_cv(tmle_RR(0, 1), data, nodes, folds)),
      c("ey0", "ey1", "rr"), scenario, replicate, truth, n
    )[3, ]
    rows[[length(rows) + 1]] <- extract(
      stage("OR", fit_cv(tmle_OR(0, 1), data, nodes, folds)),
      c("ey0", "ey1", "or"), scenario, replicate, truth, n
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
