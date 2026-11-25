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
# Beside OUTPUT the runner writes <stem>.session.txt (the package versions and sessionInfo())
# and <stem>.index.csv (Sieve's index matrix per cell, at basisN = k d + 1).  Each hte3 call
# runs inside study_stage, so an error hte3 raises costs one replicate.  The runner's own
# checks run outside every stage and stop the run.

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
      stop(sprintf("%s: the %s columns are %s, not 0 and 1", label, node, paste(colnames(held), collapse = ",")))
    }
    if (!identical(dim(held), dim(supplied)) || max(abs(unname(held) - unname(supplied))) != 0) {
      stop(sprintf("%s: the task's %s estimates differ from the supplied ones", label, node))
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
    stop(sprintf("%s: unknown scale %s", label, frame$scale[[1]]))
  }
  fitted <- study_stage("hte3 EP", learner$train(task))
  grid <- points[points$scenario == frame$scenario[[1]], ]
  if (!nrow(grid)) stop(sprintf("%s: no ISE points", label))
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
  theta <- study_stage("hte3 predict", fitted$predict(prediction_task))
  theta <- as.numeric(unlist(theta))
  if (length(theta) != nrow(grid) || any(!is.finite(theta))) {
    stop(sprintf("%s: %d predictions for %d points, or a non-finite one", label, length(theta), nrow(grid)))
  }
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
