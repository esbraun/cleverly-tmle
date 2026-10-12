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
# CLEVERLY_FIXTURE_MODE     fitter (default), preschedule, onecore, stream or inplace (the tail
#                           of the three runners that stay off the harness: study_retry and
#                           study_finish)
# CLEVERLY_FIXTURE_SHAPE    replicate (a group spans both scenarios, default), scenario, or
#                           noscenario (replicate groups whose frame carries no scenario column)
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
# CLEVERLY_FIXTURE_LOG      a path prefix; each fit appends "id,pid,parent,time" to <prefix>.<pid>.
#                           One file per process: concurrent appends to one file on a Windows bind
#                           mount can lose a line
# CLEVERLY_FIXTURE_SLEEP    seconds each fit sleeps, so a kill can land while others run
#
# RM39a2 controls.  None is read unless set, so the fixture still runs under the old harnesses.
# CLEVERLY_FIXTURE_STAGE_ERROR  group ids whose fit fails inside `study_stage`, comma-separated;
#                           `all`; or `scenario:<name>` for every group of that scenario
# CLEVERLY_FIXTURE_STAGE_KIND  how it fails: stop (default), future (a `future::value`
#                           re-signal), argument (a package argument error), fatal
#                           (`study_fatal_error`), memory (a memory-exhaustion message on the
#                           first attempt only), cstack (a C stack message), outside (a runner
#                           `tryCatch` re-signal around the stage), ltmle (a warning raised in an
#                           inner `try()` and promoted by a calling handler outside the stage)
# CLEVERLY_FIXTURE_INVALID  <rule>:<id> pairs, comma-separated: the group's first row breaks
#                           one rule of `study_invalid` (nonfinite, nonpositive, inverted,
#                           na_endpoint, coverage)

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

if (shape == "noscenario") {
  # The scenario travels as `arm`, so the harness sees a frame without a scenario column.
  names(samples)[names(samples) == "scenario"] <- "arm"
}

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
  if (shape == "noscenario") frame$scenario <- frame$arm
  for (index in seq_along(groups)) {
    candidate <- groups[[index]]
    if (identical(candidate$replicate[[1]], frame$replicate[[1]]) &&
        (shape != "scenario" || identical(candidate$scenario[[1]], frame$scenario[[1]]))) {
      return(sprintf("g%09d", index))
    }
  }
  stop("the fixture could not name a group")
}

fixture_kills <- function(id) {
  kills_file <- study_checkpoint_paths(study_checkpoint_dir(), id)$kills
  if (file.exists(kills_file)) as.integer(readLines(kills_file)[[1]]) else 0L
}

fixture_selected <- function(id, frame) {
  chosen <- fixture_list("CLEVERLY_FIXTURE_STAGE_ERROR")
  if (!length(chosen)) return(FALSE)
  if (identical(chosen, "all")) return(TRUE)
  scenario <- sub("^scenario:", "", chosen[startsWith(chosen, "scenario:")])
  id %in% chosen || (length(scenario) && all(frame$scenario %in% scenario))
}

fixture_stage_fault <- function(id, frame) {
  if (!fixture_selected(id, frame)) return(invisible(NULL))
  kind <- fixture_env("CLEVERLY_FIXTURE_STAGE_KIND", "stop")
  text <- sprintf("fixture stage error in %s", id)
  if (kind == "stop") {
    study_stage("fixture fit", stop(text))
  } else if (kind == "future") {
    study_stage("fixture fit", future::value(future::future(stop(text))))
  } else if (kind == "argument") {
    study_stage("fixture fit", stats::glm(y ~ 1, data = frame, family = "nonsense"))
  } else if (kind == "fatal") {
    study_stage("fixture fit", study_fatal_error(text))
  } else if (kind == "memory") {
    if (fixture_kills(id) < 1L) study_stage("fixture fit", stop("cannot allocate vector of size 8.0 Gb"))
  } else if (kind == "cstack") {
    study_stage("fixture fit", stop("C stack usage  7971092 is too close to the limit"))
  } else if (kind == "outside") {
    tryCatch(
      study_stage("fixture fit", stop(text)),
      error = function(condition) {
        stop(sprintf("re-signalled outside the stage: %s", conditionMessage(condition)), call. = FALSE)
      }
    )
  } else if (kind == "ltmle") {
    withCallingHandlers(
      study_stage("fixture fit", try(warning(text), silent = TRUE)),
      warning = function(condition) {
        study_fatal_error(sprintf("promoted warning: %s", conditionMessage(condition)))
      }
    )
  } else {
    stop(sprintf("unknown fixture stage kind %s", kind))
  }
  invisible(NULL)
}

fixture_invalid <- function(id, rows) {
  pairs <- strsplit(fixture_list("CLEVERLY_FIXTURE_INVALID"), ":", fixed = TRUE)
  for (pair in pairs) {
    if (length(pair) != 2L || pair[[2]] != id) next
    rule <- pair[[1]]
    if (rule == "nonfinite") rows$inference_estimate[[1]] <- Inf
    else if (rule == "nonpositive") rows$std_error[[1]] <- 0
    else if (rule == "inverted") rows$ci_lower[[1]] <- rows$ci_upper[[1]] + 1
    else if (rule == "na_endpoint") rows$ci_upper[[1]] <- NA_real_
    else if (rule == "coverage") rows$covered[[1]] <- 1L - rows$covered[[1]]
    else stop(sprintf("unknown fixture rule %s", rule))
  }
  rows
}

fit_one <- function(frame) {
  id <- group_id(frame)
  if (shape == "noscenario") frame$scenario <- frame$arm
  log <- fixture_env("CLEVERLY_FIXTURE_LOG")
  if (nzchar(log)) {
    cat(sprintf("%s,%d,%d,%.3f\n", id, Sys.getpid(), get0("study_parent_pid", ifnotfound = -1L),
        as.numeric(Sys.time())), file = sprintf("%s.%d", log, Sys.getpid()), append = TRUE)
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
  if (nzchar(fixture_env("CLEVERLY_FIXTURE_STAGE_ERROR"))) fixture_stage_fault(id, frame)
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
      # The coverage identity of `study_invalid` (RM39a2): truth 0 against estimate +/- 1.
      covered = as.integer(estimate - 1 <= 0 && 0 <= estimate + 1),
      initial_estimate = estimate,
      stringsAsFactors = FALSE
    )
  })
  rows <- do.call(rbind, rows)
  if (nzchar(fixture_env("CLEVERLY_FIXTURE_INVALID"))) rows <- fixture_invalid(id, rows)
  rows
}

scenario_column <- if (shape == "noscenario") samples$arm else samples$scenario
expected <- length(unique(paste(scenario_column, samples$replicate)))
key <- c("scenario", "replicate")
if (mode == "inplace") {
  # The tail of ctmle3_oat, ctmle_selector and drtmle, with this fixture's fits.
  ids <- sprintf("g%09d", seq_along(groups))
  seeds <- study_seeds(study_env("CLEVERLY_R_SEEDS"))
  study_record_state$prefixes <- study_prefixes(groups, ids)
  dir <- study_checkpoint_dir()
  study_startup(dir, study_env("CLEVERLY_R_META"), ids)
  plan <- study_plan(study_granted(1L))
  fit_id <- function(id) study_fit_group(id, groups[[match(id, ids)]], fit_one, dir, seeds)
  study_retry(ids, fit_id, dir, plan$workers, study_chunk(length(ids), plan$workers))
  study_finish(
    dir, ids, paths$output,
    count = function(out) length(unique(paste(out$scenario, out$replicate))),
    expected = expected,
    na = "",
    dropped = "wrote %d of %d replications"
  )
  cat("fixture 1\n")
  quit(save = "no", status = 0)
}
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
