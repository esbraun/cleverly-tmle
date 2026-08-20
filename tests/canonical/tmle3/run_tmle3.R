suppressPackageStartupMessages({
  library(sl3)
  library(tmle3)
})
options(digits = 17)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: run_tmle3.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
input <- args[[1]]
truth_input <- args[[2]]
output <- args[[3]]

# The truth arrives once per replication rather than repeated on every row.  Carrying ten
# constant columns through 3.2 million rows was most of this process's memory, and memory is
# what decides how many workers fit.
samples <- read.csv(gzfile(input), stringsAsFactors = FALSE)
truths <- read.csv(truth_input, stringsAsFactors = FALSE)
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
expected_groups <- length(groups)
rm(samples)
invisible(gc())
fit_group <- function(i) {
  frame <- groups[[i]]
  tryCatch(
    {
      result <- fit_one(frame, frame$scenario[[1]], frame$replicate[[1]])
      if (i %% 10 == 0) cat(sprintf("completed %d/%d samples\n", i, length(groups)))
      result
    },
    error = function(condition) {
      structure(
        list(
          index = i,
          scenario = frame$scenario[[1]],
          replicate = frame$replicate[[1]],
          message = conditionMessage(condition)
        ),
        class = "study_error"
      )
    }
  )
}
# The whole core budget the caller measured, not a hard-coded four.  The Python side has
# already finished by the time this runs, so nothing else is competing for these cores.
requested <- suppressWarnings(as.integer(Sys.getenv("CLEVERLY_R_CORES", "")))
cores <- if (is.na(requested) || requested < 1L) {
  max(1L, parallel::detectCores(logical = TRUE))
} else {
  requested
}
# Capped by memory as well as by cores.  ``mclapply`` forks, and a forked worker that the
# kernel kills does not raise: it returns a non-data-frame that ``rbind`` drops without a
# word, which is how a run once lost 86 of 3,200 replications while reporting success.  The
# check after the loop refuses that outcome; this cap is what stops it happening.
memory_kb <- as.numeric(
  sub("[^0-9]*([0-9]+).*", "\\1", grep("MemTotal", readLines("/proc/meminfo"), value = TRUE))
)
footprint_kb <- as.numeric(utils::object.size(groups)) / 1024
affordable <- max(1L, as.integer(floor(memory_kb * 0.5 / max(footprint_kb * 0.5, 1))))
if (affordable < cores) {
  cat(sprintf("capping %d cores to %d for memory\n", cores, affordable))
  cores <- affordable
}
cat(sprintf("fitting %d samples on %d cores\n", length(groups), cores))
results <- parallel::mclapply(
  seq_along(groups), fit_group, mc.cores = cores, mc.preschedule = FALSE
)
malformed <- which(!vapply(results, is.data.frame, logical(1)) &
  !vapply(results, inherits, logical(1), what = "study_error"))
if (length(malformed)) {
  stop(sprintf(
    "%d of %d workers returned no result at all (groups %s); they were killed rather than erroring",
    length(malformed), expected_groups, paste(utils::head(malformed, 10), collapse = ", ")
  ))
}
failed <- vapply(results, inherits, logical(1), what = "study_error")
if (any(failed)) {
  messages <- vapply(
    results[failed],
    function(error) sprintf(
      "%s replicate %s (group %s): %s",
      error$scenario, error$replicate, error$index, error$message
    ),
    character(1)
  )
  stop(paste(messages, collapse = "\n"))
}
out <- do.call(rbind, results)
if (length(unique(paste(out$scenario, out$replicate))) != expected_groups) {
  stop(sprintf(
    "wrote %d of %d replications; a silently dropped replication is not a shorter study",
    length(unique(paste(out$scenario, out$replicate))), expected_groups
  ))
}
write.csv(out, output, row.names = FALSE)
cat("tmle3 ", as.character(packageVersion("tmle3")), "\n", sep = "")
cat("sl3 ", as.character(packageVersion("sl3")), "\n", sep = "")
cat("R ", R.version.string, "\n", sep = "")
