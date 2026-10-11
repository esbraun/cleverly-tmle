# A reference runner that fits nothing, for the RM39 harness smoke.
#
# usage: Rscript fixture_runner.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv
#
# tests/canonical/harness_fixture/smoke.py runs it in the tmle3 image with tests/canonical
# mounted at /fixture.  Each fit returns the mean of its rows, so the table is a function of the
# samples alone, and the controls below break the run in each way the harness must survive.
# The file is not named run_*.R, so the runner parity tests do not read it as a study runner.
# With no control set it uses only what the harness had before RM39, so the smoke can run it
# under the old harness too and compare the two tables.
#
# CLEVERLY_FIXTURE_MODE     fitter (default), preschedule, onecore or stream
# CLEVERLY_FIXTURE_SHAPE    replicate (a group spans both scenarios, default) or scenario
# CLEVERLY_FIXTURE_KILL     group ids whose fit kills its own fork, comma-separated
# CLEVERLY_FIXTURE_KILL_ATTEMPTS  how many attempts of those groups are killed (default 1)
# CLEVERLY_FIXTURE_ERROR    a group id whose fit raises an error
# CLEVERLY_FIXTURE_VANISH   a group id whose fit removes its start marker and exits without a
#                           checkpoint: a pass that returns without fitting
# CLEVERLY_HARNESS_STOP_AFTER  after k checkpoints, a fit kills the parent and itself
# CLEVERLY_FIXTURE_PARENT_KILL 1: kill the parent once, after 4 checkpoints; 2: on every run
# CLEVERLY_FIXTURE_HEAP_MB  the parent holds about this many MB in small objects
# CLEVERLY_FIXTURE_TRANSIENT_MB  <scenario>:<m>: a fit of that scenario allocates m MB and
#                           frees it before it returns
# CLEVERLY_FIXTURE_LOG      a file each fit appends "id,pid,parent,time" to
# CLEVERLY_FIXTURE_SLEEP    seconds each fit sleeps, so a kill can land while others run

source("/fixture/study_harness.R")
paths <- study_arguments("usage: fixture_runner.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(paths$samples), stringsAsFactors = FALSE)

fixture_env <- function(name, default = "") {
  value <- Sys.getenv(name, "")
  if (nzchar(value)) value else default
}
fixture_number <- function(name, default) {
  value <- suppressWarnings(as.numeric(fixture_env(name)))
  if (is.na(value)) default else value
}
fixture_list <- function(name) {
  value <- strsplit(fixture_env(name), ",", fixed = TRUE)[[1]]
  value[nzchar(value)]
}

mode <- fixture_env("CLEVERLY_FIXTURE_MODE", "fitter")
shape <- fixture_env("CLEVERLY_FIXTURE_SHAPE", "replicate")
random <- exists("fixture_random", inherits = FALSE) && isTRUE(fixture_random)
heap_mb <- fixture_number("CLEVERLY_FIXTURE_HEAP_MB", 0)
# Many small objects, not one vector: a full collection writes each object's header, so the
# copy a forked worker makes of this heap grows with the object count.
ballast <- if (heap_mb > 0) lapply(seq_len(as.integer(heap_mb * 1024)), function(i) numeric(120)) else NULL
killed <- fixture_list("CLEVERLY_FIXTURE_KILL")
kill_attempts <- fixture_number("CLEVERLY_FIXTURE_KILL_ATTEMPTS", 1)
transient <- strsplit(fixture_env("CLEVERLY_FIXTURE_TRANSIENT_MB"), ":", fixed = TRUE)[[1]]
stop_after <- fixture_number("CLEVERLY_HARNESS_STOP_AFTER", NA_real_)
parent_kill <- fixture_number("CLEVERLY_FIXTURE_PARENT_KILL", 0)
if (parent_kill > 0 && is.na(stop_after)) stop_after <- 4

if (shape == "scenario") {
  groups <- split(samples, list(samples$scenario, samples$replicate), drop = TRUE)
  groups <- groups[order(
    vapply(groups, function(g) g$replicate[[1]], numeric(1)),
    vapply(groups, function(g) g$scenario[[1]], character(1))
  )]
} else {
  groups <- split(samples, samples$replicate)
}
names(groups) <- NULL

group_id <- function(frame) {
  if (mode == "stream") return(sprintf("r%09d", as.integer(frame$replicate[[1]])))
  for (index in seq_along(groups)) {
    candidate <- groups[[index]]
    if (identical(candidate$replicate[[1]], frame$replicate[[1]]) &&
        (shape != "scenario" || identical(candidate$scenario[[1]], frame$scenario[[1]]))) {
      return(sprintf("g%09d", index))
    }
  }
  stop("the fixture could not name a group")
}

fit_one <- function(frame) {
  id <- group_id(frame)
  log <- fixture_env("CLEVERLY_FIXTURE_LOG")
  if (nzchar(log)) {
    cat(sprintf("%s,%d,%d,%.3f\n", id, Sys.getpid(), get0("study_parent_pid", ifnotfound = -1L),
        as.numeric(Sys.time())), file = log, append = TRUE)
  }
  pause <- fixture_number("CLEVERLY_FIXTURE_SLEEP", 0)
  if (pause > 0) Sys.sleep(pause)
  if (!is.na(stop_after)) {
    dir <- study_checkpoint_dir()
    once <- file.path(dir, "_parent_killed")
    checkpoints <- length(list.files(dir, pattern = "\\.rds$"))
    if (checkpoints >= stop_after && (parent_kill != 1 || !file.exists(once))) {
      if (parent_kill == 1) writeLines("1", once)
      tools::pskill(study_parent_pid, tools::SIGKILL)
      tools::pskill(Sys.getpid(), tools::SIGKILL)
    }
  }
  if (id %in% killed && !study_calibrating()) {
    kills_file <- study_checkpoint_paths(study_checkpoint_dir(), id)$kills
    kills <- if (file.exists(kills_file)) as.integer(readLines(kills_file)[[1]]) else 0L
    if (kills < kill_attempts) tools::pskill(Sys.getpid(), tools::SIGKILL)
  }
  if (identical(id, fixture_env("CLEVERLY_FIXTURE_ERROR"))) stop(sprintf("fixture error in %s", id))
  if (identical(id, fixture_env("CLEVERLY_FIXTURE_VANISH"))) {
    unlink(study_checkpoint_paths(study_checkpoint_dir(), id)$started)
    tools::pskill(Sys.getpid(), tools::SIGKILL)
  }
  if (length(transient) == 2L && transient[[1]] %in% frame$scenario) {
    block <- seq_len(as.integer(as.numeric(transient[[2]]) * 262144)) + 0L
    rm(block)
  }
  rows <- lapply(sort(unique(frame$scenario)), function(scenario) {
    part <- frame[frame$scenario == scenario, ]
    estimate <- mean(part$y) + if (random) stats::rnorm(1) else 0
    data.frame(
      implementation = "fixture",
      scenario = scenario,
      replicate = part$replicate[[1]],
      n = nrow(part),
      estimand = "mean",
      truth = 0,
      estimate = estimate,
      inference_estimate = estimate,
      std_error = 1,
      ci_lower = estimate - 1,
      ci_upper = estimate + 1,
      inference_scale = "identity",
      covered = 1L,
      initial_estimate = estimate,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

expected <- length(unique(paste(samples$scenario, samples$replicate)))
key <- c("scenario", "replicate")
if (mode == "stream") {
  results <- study_stream(paths$samples, fit_one, length(unique(samples$replicate)), chunk_lines = 40L)
} else {
  fit_group <- study_fitter(groups, fit_one, progress_every = 4L)
  cores <- if (mode == "onecore") 1L else study_cores(groups)
  results <- parallel::mclapply(
    seq_along(groups), fit_group,
    mc.cores = cores, mc.preschedule = identical(mode, "preschedule")
  )
}
study_collect(results, expected, paths$output, versions = "fixture 1", key = key)
