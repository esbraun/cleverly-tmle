suppressPackageStartupMessages({
  library(sl3)
  library(tmle3)
})
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments(
  "usage: run_tmle3_stratified.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv"
)
samples <- read.csv(gzfile(paths$samples), stringsAsFactors = FALSE, check.names = FALSE)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
truth_key <- paste(truths$scenario, truths$replicate, sep = "|")
truth_columns <- grep("^truth_", names(truths), value = TRUE)
learners <- list(A = sl3::Lrnr_glm$new(), Y = sl3::Lrnr_glm$new())
nodes <- list(W = "W", V = "V", A = "A", Y = "Y")

# The stratum of each stratified parameter, read off its name.  ``Param_stratified$name`` is
# "<base name> | V=<label>", where the label is the stratum's value of V.  The strata are
# numbered in order of first appearance in the data, so the code never assumes an order.
stratum_of <- function(param_names) {
  labels <- sub("^.* \\| V=", "", param_names)
  stopifnot(all(labels != param_names))
  as.integer(labels)
}

# ``get_strata_weights`` merges V with the strata table under ``sort = FALSE`` and indexes the
# result by ``.I``.  The row order of that merge is not documented, so check it: every row's one
# nonzero weight column must be its own stratum, and the weight must be n / n_s.
check_alignment <- function(fit, data) {
  for (param in fit$tmle_params) {
    weights <- param$get_strata_weights(fit$tmle_task)
    strata <- param$strata
    codes <- as.integer(as.character(strata$strata_i))
    labels <- as.integer(strata[["V"]])
    column_label <- labels[match(seq_len(ncol(weights)), codes)]
    nonzero <- weights != 0
    stopifnot(all(rowSums(nonzero) == 1L))
    own <- column_label[max.col(nonzero, ties.method = "first")]
    if (!identical(as.integer(own), as.integer(data$V))) {
      stop("tmle3 stratum weights are not aligned with the rows' own strata")
    }
    counts <- table(data$V)
    expected <- nrow(data) / as.numeric(counts[as.character(data$V)])
    observed <- weights[cbind(seq_len(nrow(weights)), max.col(nonzero, ties.method = "first"))]
    stopifnot(isTRUE(all.equal(unname(observed), unname(expected), tolerance = 1e-12)))
  }
}

# ``fit$summary`` fails on a stratified spec with two base parameters at ed72f8a.  The fit
# stores ``tmle_param_names`` and ``initial_psi`` as stratum-by-parameter matrices, and
# ``summary_from_estimates`` then builds a 12-column table that it names with 10 names.  This
# computes the same summary from ``fit$estimates`` with the same arithmetic: the covariance of
# the stacked influence curves with an n - 1 denominator, ``se = sqrt(diag / n)``, and the
# Wald interval at ``qnorm(0.975)``.  Every transform here is the identity.
stratified_summary <- function(fit) {
  estimates <- fit$estimates
  psi <- unlist(lapply(estimates, `[[`, "psi"), use.names = FALSE)
  curves <- do.call(cbind, lapply(estimates, `[[`, "IC"))
  se <- sqrt(diag(stats::cov(curves)) / nrow(curves))
  q <- abs(stats::qnorm(0.025))
  data.frame(
    param = as.vector(fit$tmle_param_names),
    init_est = as.vector(fit$initial_psi),
    tmle_est = psi,
    se = se,
    lower = psi - q * se,
    upper = psi + q * se,
    stringsAsFactors = FALSE
  )
}

rows_from <- function(tab, data, estimand, truth, scenario, replicate) {
  reference <- unname(truth[estimand])
  stopifnot(!anyNA(reference))
  data.frame(
    implementation = "tmle3-stratified",
    scenario = scenario,
    replicate = replicate,
    n = nrow(data),
    estimand = estimand,
    truth = reference,
    estimate = tab$tmle_est,
    inference_estimate = tab$tmle_est,
    std_error = tab$se,
    ci_lower = tab$lower,
    ci_upper = tab$upper,
    inference_scale = "identity",
    covered = as.integer(tab$lower <= reference & reference <= tab$upper),
    initial_estimate = tab$init_est,
    stringsAsFactors = FALSE
  )
}

fit_one <- function(frame) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  row <- match(paste(scenario, replicate, sep = "|"), truth_key)
  if (is.na(row)) stop(sprintf("no truth for %s replicate %s", scenario, replicate))
  truth <- as.numeric(truths[row, truth_columns])
  names(truth) <- sub("^truth_", "", truth_columns)
  data <- frame[c("V", "W", "A", "Y")]
  stage <- function(label, expression) {
    tryCatch(
      force(expression),
      error = function(condition) {
        stop(sprintf("%s: %s", label, conditionMessage(condition)), call. = FALSE)
      }
    )
  }
  # Two stages, as tests/canonical/tmle3/run_tmle3.R fits the marginal parameters.
  # ``base_estimate = FALSE``: the marginal clever covariate is the sum of the stratum
  # columns, so fluctuating it as well would make the design collinear.
  tsm <- stage(
    "TSM",
    tmle3(tmle_stratified(tmle_TSM_all(), base_estimate = FALSE), data, nodes, learners)
  )
  check_alignment(tsm, data)
  tsm_table <- stratified_summary(tsm)
  treated <- grepl("A=1", tsm_table$param, fixed = TRUE)
  stopifnot(all(treated | grepl("A=0", tsm_table$param, fixed = TRUE)))
  tsm_rows <- rows_from(
    tsm_table, data,
    sprintf("ey[%s][V=%d]", ifelse(treated, "1", "0"), stratum_of(tsm_table$param)),
    truth, scenario, replicate
  )
  ate <- stage(
    "ATE",
    tmle3(tmle_stratified(tmle_ATE(1, 0), base_estimate = FALSE), data, nodes, learners)
  )
  check_alignment(ate, data)
  ate_table <- stratified_summary(ate)
  ate_rows <- rows_from(
    ate_table, data, sprintf("ate[V=%d]", stratum_of(ate_table$param)),
    truth, scenario, replicate
  )
  out <- rbind(tsm_rows, ate_rows)
  stopifnot(nrow(out) == 9L, !anyDuplicated(out$estimand))
  out
}

groups <- split(samples, interaction(samples$scenario, samples$replicate, drop = TRUE))
expected <- length(groups)
rm(samples)
invisible(gc())
results <- parallel::mclapply(
  seq_along(groups), study_fitter(groups, fit_one),
  mc.cores = study_cores(groups), mc.preschedule = FALSE
)
study_collect(
  results, expected, paths$output,
  versions = c(study_version("tmle3"), study_version("sl3")),
  key = c("scenario", "replicate"), na = "NA"
)
