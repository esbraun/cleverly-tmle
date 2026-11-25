# The hte3 side of the comparison of the EP curve (tests/studies/ep_hte3.py; P1 plan, PR 4b,
# section 4.2).  One runner serves both scales.
#
# Usage: hte3_ep_runner.R INPUTS.csv.gz POINTS.csv.gz OUTPUT.csv
#
# INPUTS holds one block per (scenario, replicate): scenario is the comparison cell, k the
# cosines per modifier, scale "difference" or "log_ratio", the modifiers W1..Wd, A, Y, and the
# Python fit's nuisances mu0, mu1 (the arm regressions) and pi0, pi1 (the propensity after its
# g_bounds).  POINTS holds each cell's ISE points.  For each block the runner builds hte3's task
# with the supplied nuisances (cross_fit_and_cv = FALSE, so they become LF_known factors),
# checks that the task returns them unchanged and in the column order (A = 0, A = 1), fits
# Lrnr_cate_EP(targeting_style = "dr") or Lrnr_crr_EP with sieve_num_basis = k d and
# sieve_interaction_order = 1, and writes the curve at the points: columns scenario, replicate,
# point, theta and seconds.  The comparison and the readings are computed in Python.
#
# The final stage is Lrnr_sieve_ls below, a runner-local learner, so hte3 is not patched (the
# tmle3_survival l2 updater subclass is the precedent).  It fits the pseudo-outcome on hte3's
# own sieve: least squares for the CATE, and for the log CRR the quasibinomial fit with hte3's
# pseudo-weights, predicting on the probability scale, because Lrnr_hte applies
# bounded_qlogis to the base learner's predictions once.  That is EPL Algorithm 3 with the
# sieve final stage.  It keeps Sieve's training normalization and index matrix and applies
# them to the prediction points.
#
# Beside OUTPUT the runner writes <stem>.session.txt (the package versions and sessionInfo()),
# <stem>.index.csv (Sieve's index matrix per cell, at basisN = k d + 1) and one file per fitted
# replicate in <stem>.fits/: the smallest and largest propensity per arm after
# hte3:::truncate_propensity (each is hte3's cutoff wherever that cutoff binds; the driver
# compares them with the emulation's at 1e-10), the warnings the EP step raised, and the glmnet
# fits it made.
#
# Failures.  Each hte3 call, and the check that its predictions are complete and finite, runs
# inside study_stage, so an error there costs one replicate: it is recorded in
# <stem>.failures.csv, counted and published by the driver, and never refilled.  A cell that
# loses more than 5% of its replicates is reported, not gated (tests/studies/ep_hte3.py).  The
# integrity checks (the nuisance round trip, an unknown scale, a cell with no ISE points, a
# glmnet fit with an empty coefficient vector) raise study_fatal_error and stop the run.
#
# Warnings.  Every warning the EP step raises is counted, logged with its replicate and
# muffled; one that names glmnet or convergence is counted apart.  glmnet's fits are observed
# through trace() on glmnet::glmnet, which records each fit's coefficients and changes nothing
# hte3 computes.

suppressPackageStartupMessages({
  library(sl3)
  library(hte3)
})
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments("usage: hte3_ep_runner.R INPUTS.csv.gz POINTS.csv.gz OUTPUT.csv")

inputs <- utils::read.csv(gzfile(paths$samples), stringsAsFactors = FALSE)
points <- utils::read.csv(gzfile(paths$truths), stringsAsFactors = FALSE)
stem <- file.path(dirname(paths$output), sub("(.)\\.[^.]+$", "\\1", basename(paths$output)))

fits_dir <- paste0(stem, ".fits")
dir.create(fits_dir, showWarnings = FALSE)

# Each glmnet fit hte3 makes, recorded by the trace below: its coefficient count, how many are
# nonzero, and its passes.  Reset per replicate.
glmnet_fits <- new.env(parent = emptyenv())
glmnet_fits$records <- list()
note_glmnet_fit <- function(fit) {
  beta <- if (is.null(fit$beta)) numeric(0) else as.vector(as.matrix(fit$beta))
  glmnet_fits$records[[length(glmnet_fits$records) + 1L]] <- c(
    columns = length(beta),
    nonzero = sum(beta != 0),
    npasses = if (is.null(fit$npasses)) NA_real_ else as.numeric(fit$npasses)
  )
  invisible(NULL)
}
invisible(suppressMessages(trace(
  "glmnet",
  exit = bquote(.(note_glmnet_fit)(returnValue())),
  where = asNamespace("glmnet"),
  print = FALSE
)))

# The warnings of the current replicate's EP step.
warned <- new.env(parent = emptyenv())

Lrnr_sieve_ls <- R6::R6Class(
  classname = "Lrnr_sieve_ls",
  inherit = sl3::Lrnr_base,
  portable = TRUE,
  class = TRUE,
  public = list(
    initialize = function(sieve_num_basis, family = stats::gaussian(), ...) {
      super$initialize(params = list(sieve_num_basis = sieve_num_basis, family = family, ...))
    }
  ),
  private = list(
    .properties = c("continuous", "binomial", "quasibinomial", "weights"),
    .train = function(task) {
      basis <- Sieve::sieve_preprocess(
        as.matrix(task$X),
        basisN = as.integer(self$params$sieve_num_basis) + 1L,
        interaction_order = 1,
        type = "cosine"
      )
      family <- self$params$family
      if (is.character(family)) family <- get(family, mode = "function")()
      fit <- stats::glm.fit(
        basis$Phi, task$Y,
        weights = task$weights,
        family = family,
        intercept = FALSE,
        control = stats::glm.control(epsilon = 1e-14, maxit = 100)
      )
      list(
        coefficients = fit$coefficients,
        index_matrix = basis$index_matrix,
        norm_para = basis$norm_para,
        basisN = basis$basisN,
        family = family
      )
    },
    .predict = function(task) {
      fit <- self$fit_object
      basis <- Sieve::sieve_preprocess(
        as.matrix(task$X),
        basisN = fit$basisN,
        interaction_order = 1,
        type = "cosine",
        index_matrix = fit$index_matrix,
        norm_para = fit$norm_para
      )
      as.vector(fit$family$linkinv(as.vector(basis$Phi %*% fit$coefficients)))
    }
  )
)

check_round_trip <- function(task, mu, pi, label) {
  # Plan 4.2: the task hands the EP step exactly the supplied nuisances, column j at level
  # sort(unique(A))[j].
  for (node in c("mu", "pi")) {
    held <- as.matrix(task$get_nuisance_estimates(node))
    supplied <- if (node == "mu") mu else pi
    if (!identical(colnames(held), c("0", "1"))) {
      study_fatal_error(sprintf("%s: the %s columns are %s, not 0 and 1", label, node, paste(colnames(held), collapse = ",")))
    }
    if (!identical(dim(held), dim(supplied)) || max(abs(unname(held) - unname(supplied))) != 0) {
      study_fatal_error(sprintf("%s: the task's %s estimates differ from the supplied ones", label, node))
    }
  }
}

fit_one <- function(frame) {
  set.seed(0)
  label <- sprintf("%s replicate %s", frame$scenario[[1]], frame$replicate[[1]])
  covariates <- grep("^W[0-9]+$", names(frame), value = TRUE)
  basis_n <- frame$k[[1]] * length(covariates)
  data <- frame[c(covariates, "A", "Y")]
  mu <- cbind(frame$mu0, frame$mu1)
  pi <- cbind(frame$pi0, frame$pi1)
  started <- proc.time()[["elapsed"]]
  task <- study_stage(
    "hte3 task",
    make_hte3_Task_tx(
      data,
      modifiers = covariates, confounders = covariates, treatment = "A", outcome = "Y",
      pi.hat = pi, mu.hat = mu, cross_fit_and_cv = FALSE
    )
  )
  check_round_trip(task, mu, pi, label)
  # hte3's own truncation of the supplied propensity, as resolve_contrast_nuisances applies it.
  truncated_1 <- study_stage(
    "hte3 truncation",
    hte3:::truncate_propensity(frame$pi1, frame$A, treatment_level = 1, truncation_method = "adaptive")
  )
  truncated_0 <- study_stage(
    "hte3 truncation",
    hte3:::truncate_propensity(frame$pi0, frame$A, treatment_level = 0, truncation_method = "adaptive")
  )
  if (identical(frame$scale[[1]], "difference")) {
    learner <- Lrnr_cate_EP$new(
      base_learner = Lrnr_sieve_ls$new(sieve_num_basis = basis_n),
      sieve_num_basis = basis_n,
      sieve_interaction_order = 1,
      screen_basis_with_lasso = FALSE,
      targeting_style = "dr",
      treatment_level = 1,
      control_level = 0
    )
  } else if (identical(frame$scale[[1]], "log_ratio")) {
    learner <- Lrnr_crr_EP$new(
      base_learner = Lrnr_sieve_ls$new(sieve_num_basis = basis_n),
      sieve_num_basis = basis_n,
      sieve_interaction_order = 1,
      treatment_level = 1,
      control_level = 0
    )
  } else {
    study_fatal_error(sprintf("%s: unknown scale %s", label, frame$scale[[1]]))
  }
  grid <- points[points$scenario == frame$scenario[[1]], ]
  if (!nrow(grid)) study_fatal_error(sprintf("%s: no ISE points", label))
  glmnet_fits$records <- list()
  warned$all <- 0L
  warned$glmnet <- 0L
  fitted <- withCallingHandlers(
    study_stage("hte3 EP", learner$train(task)),
    warning = function(condition) {
      text <- conditionMessage(condition)
      warned$all <- warned$all + 1L
      if (grepl("glmnet|converge", text, ignore.case = TRUE)) warned$glmnet <- warned$glmnet + 1L
      cat(sprintf("%s: warning in the hte3 EP step: %s\n", label, text))
      invokeRestart("muffleWarning")
    }
  )
  records <- glmnet_fits$records
  if (!length(records)) {
    study_fatal_error(sprintf("%s: the trace saw no glmnet fit in hte3's EP step", label))
  }
  empty <- vapply(records, function(record) record[["columns"]] == 0 || record[["nonzero"]] == 0, logical(1))
  if (any(empty)) {
    study_fatal_error(sprintf("%s: glmnet returned an empty coefficient vector in hte3's EP step", label))
  }
  # A prediction task reads the modifiers only; A and Y are placeholders of both levels.
  newdata <- data.frame(
    grid[covariates],
    A = rep_len(c(0, 1), nrow(grid)),
    Y = rep_len(c(0, 1), nrow(grid))
  )
  prediction_task <- study_stage(
    "hte3 prediction task",
    make_hte3_Task_tx(
      newdata,
      modifiers = covariates, confounders = covariates, treatment = "A", outcome = "Y",
      for_prediction = TRUE
    )
  )
  theta <- study_stage("hte3 predict", {
    predicted <- as.numeric(unlist(fitted$predict(prediction_task)))
    if (length(predicted) != nrow(grid) || any(!is.finite(predicted))) {
      stop(sprintf("%d predictions for %d points, or a non-finite one", length(predicted), nrow(grid)))
    }
    predicted
  })
  utils::write.csv(
    data.frame(
      scenario = frame$scenario[[1]],
      replicate = frame$replicate[[1]],
      pi1_low = min(truncated_1),
      pi1_high = max(truncated_1),
      pi0_low = min(truncated_0),
      pi0_high = max(truncated_0),
      warnings = warned$all,
      glmnet_warnings = warned$glmnet,
      glmnet_fits = length(records),
      glmnet_npasses = max(vapply(records, function(record) record[["npasses"]], numeric(1)))
    ),
    file.path(fits_dir, sprintf("%s-%s.csv", frame$scenario[[1]], frame$replicate[[1]])),
    row.names = FALSE
  )
  data.frame(
    scenario = frame$scenario[[1]],
    replicate = frame$replicate[[1]],
    point = grid$point,
    theta = theta,
    seconds = proc.time()[["elapsed"]] - started,
    stringsAsFactors = FALSE
  )
}

# The record the smoke reads: versions, the session, and Sieve's index matrix per cell.
versions <- c(
  study_version("hte3"), study_version("sl3"), study_version("tmle3"),
  study_version("Sieve"), study_version("glmnet"), study_version("data.table")
)
writeLines(
  c(versions, R.version.string, paste0("HTE3_COMMIT ", Sys.getenv("HTE3_COMMIT")),
    utils::capture.output(utils::sessionInfo())),
  paste0(stem, ".session.txt")
)
indices <- list()
for (cell in unique(inputs$scenario)) {
  first <- inputs[inputs$scenario == cell, ]
  first <- first[first$replicate == first$replicate[[1]], ]
  covariates <- grep("^W[0-9]+$", names(first), value = TRUE)
  index <- Sieve::sieve_preprocess(
    as.matrix(first[covariates]),
    basisN = first$k[[1]] * length(covariates) + 1L,
    interaction_order = 1,
    type = "cosine"
  )$index_matrix
  # Long form, one row per basis column and modifier, so cells of different dimension stack.
  indices[[cell]] <- data.frame(
    scenario = cell,
    column = rep(seq_len(nrow(index)), times = ncol(index)),
    modifier = rep(seq_len(ncol(index)), each = nrow(index)),
    index = as.vector(index)
  )
}
utils::write.csv(do.call(rbind, unname(indices)), paste0(stem, ".index.csv"), row.names = FALSE)

groups <- split(inputs, interaction(inputs$scenario, inputs$replicate, drop = TRUE, lex.order = TRUE))
expected <- length(groups)
rm(inputs)
invisible(gc())

fit_group <- study_fitter(groups, fit_one)
cores <- study_cores(groups)
results <- parallel::mclapply(seq_along(groups), fit_group, mc.cores = cores, mc.preschedule = FALSE)
study_collect(
  results,
  expected,
  paths$output,
  versions = versions,
  key = c("scenario", "replicate"),
  na = "NA"
)
