suppressPackageStartupMessages({
  library(data.table)
  library(sl3)
  library(tmle3)
  library(ctmle3)
})
options(digits = 17)

# ---- RM39 shared resilience functions: begin ----
# tests/canonical/study_harness.R and the three runners that stay off it (ctmle3_oat,
# ctmle_selector, drtmle) carry this block byte for byte; tests/unit/test_harness_drift.py
# compares the four copies.  Change it here and copy it to the other three.
#
# What it does: every fit runs in a forked child, writes a start marker, and leaves its result
# in a checkpoint file.  A killed child leaves the marker and no checkpoint, so the retry loop
# can name it, run it again alone, and stop when it is killed twice.  A rerun of the container
# reuses every checkpoint.  The worker count comes from measured memory, not from the input
# size.
#
# RM39 part a2: an error that escapes a runner's `study_stage` becomes a `study_fitting_failure`,
# and the group's checkpoint holds a failure record in place of a result.  A row that breaks the
# finite, positive, ordering or coverage rule (`study_invalid`) is recorded the same way.  When
# the first 20 groups of a scenario all fail, the run stops as systematic.  Every other error,
# and a `study_fatal_error` raised anywhere, is still fatal: the first one stops the run.

study_parent_pid <- Sys.getpid()

#: The id prefixes `study_systematic` reads, set by the parent before any group is fitted.
study_record_state <- new.env(parent = emptyenv())

study_recording <- function() {
  # The RM19 diagnostic wrapper turns recording off, so every comparator error stays raw.
  !identical(study_env("CLEVERLY_R_NO_RECORD"), "1")
}

study_fatal_error <- function(message) {
  # The non-recordable class: a check that a runner or adapter makes inside a stage (a promoted
  # warning, for example) raises this and stays fatal.
  stop(structure(
    class = c("study_fatal_error", "error", "condition"),
    list(message = message, call = sys.call(-1))
  ))
}

study_memory_kill <- function(label, message) {
  # Decision Q2: memory exhaustion is a kill, not a recorded failure.  The worker kills itself
  # as the kernel would, leaving its start marker and no checkpoint, so the retry loop halves
  # the workers and fits the group again.  The parent never kills itself.
  if (Sys.getpid() == study_parent_pid) {
    study_fatal_error(sprintf("%s: %s (memory exhausted in the parent)", label, message))
  }
  cat(sprintf("memory exhausted in %s; this worker kills itself: %s\n", label, message))
  tools::pskill(Sys.getpid(), tools::SIGKILL)
  study_fatal_error(sprintf("%s: %s (the worker survived its own kill)", label, message))
}

study_stage <- function(label, expr) {
  # The declared comparator-fit stage.  `expr` is lazy, so it is evaluated inside the handler
  # and its value comes back unchanged.  The handler is exiting: it sees only an error that
  # escapes `expr`, so a package's own `try()` keeps working, and an error re-signalled by
  # `future::value()` or a package's `mclapply` is recorded like any other.  A fatal class and an
  # inner stage's failure pass through, so the innermost label is the one recorded.
  if (!study_recording()) return(expr)
  tryCatch(
    expr,
    error = function(condition) {
      if (inherits(condition, c("study_fatal_error", "study_fitting_failure"))) stop(condition)
      text <- conditionMessage(condition)
      if (grepl("cannot allocate|vector memory exhausted|memory exhausted", text)) {
        study_memory_kill(label, text)
      }
      if (grepl("C stack usage", text, fixed = TRUE)) {
        study_fatal_error(sprintf("%s: %s", label, text))
      }
      stop(structure(
        class = c("study_fitting_failure", "error", "condition"),
        list(
          message = text, call = conditionCall(condition),
          stage = label, original = paste(class(condition), collapse = "/")
        )
      ))
    }
  )
}

study_invalid <- function(result) {
  # The first rule a data-frame result breaks, or NULL: the finite, positive and ordering rules
  # and the coverage identity that the Python step refuses.  Only columns present are read, and
  # a missing value breaks a rule rather than raising.
  if (!is.data.frame(result) || !nrow(result)) return(NULL)
  columns <- names(result)
  number <- function(column) suppressWarnings(as.numeric(result[[column]]))
  where <- function(rows) {
    row <- rows[[1L]]
    estimand <- if ("estimand" %in% columns) as.character(result$estimand[[row]]) else "row"
    scenario <- if ("scenario" %in% columns) sprintf(" of %s", as.character(result$scenario[[row]])) else ""
    sprintf("%s%s (row %d)", estimand, scenario, row)
  }
  for (column in intersect(c("estimate", "inference_estimate", "std_error"), columns)) {
    rows <- which(!is.finite(number(column)))
    if (length(rows)) return(sprintf("non-finite %s for %s", column, where(rows)))
  }
  if ("std_error" %in% columns) {
    positive <- number("std_error") > 0
    rows <- which(is.na(positive) | !positive)
    if (length(rows)) return(sprintf("non-positive std_error for %s", where(rows)))
  }
  if (all(c("ci_lower", "ci_upper") %in% columns)) {
    ordered <- number("ci_lower") <= number("ci_upper")
    rows <- which(is.na(ordered) | !ordered)
    if (length(rows)) return(sprintf("ci_lower above ci_upper for %s", where(rows)))
  }
  if (all(c("covered", "truth", "ci_lower", "ci_upper") %in% columns)) {
    truth <- number("truth")
    inside <- number("ci_lower") <= truth & truth <= number("ci_upper")
    inside[is.na(inside)] <- FALSE
    covered <- number("covered")
    rows <- which(is.na(covered) | covered != as.numeric(inside))
    if (length(rows)) return(sprintf("covered disagrees with the interval and truth for %s", where(rows)))
  }
  NULL
}

study_failure_row <- function(id, frame, attempt, class, stage, call, message, seed) {
  # One failure record (M7).  The Python driver adds the implementation column.
  clean <- function(text) gsub("[\r\n]", " ", paste(text, collapse = " "))
  data.frame(
    group_id = id,
    scenarios = if ("scenario" %in% names(frame)) {
      paste(unique(as.character(frame$scenario)), collapse = ";")
    } else {
      "*"
    },
    replicates = paste(sprintf("%d", as.integer(unique(frame$replicate))), collapse = ";"),
    attempt = as.integer(attempt),
    class = class,
    stage = clean(stage),
    call = clean(if (is.null(call)) "none" else deparse(call, width.cutoff = 500L, nlines = 1L)),
    message = clean(message),
    seed = if (is.null(seed)) NA_integer_ else as.integer(seed),
    stringsAsFactors = FALSE
  )
}

study_prefixes <- function(frames, ids) {
  # The systematic stop's id prefixes: the first 20 ids of each scenario when every group holds
  # one scenario, else the first 20 ids of the run.
  scenarios <- lapply(frames, function(frame) {
    if ("scenario" %in% names(frame)) unique(as.character(frame$scenario)) else "*"
  })
  if (length(frames) && all(lengths(scenarios) == 1L)) {
    keys <- unlist(scenarios)
    return(lapply(split(ids, factor(keys, levels = unique(keys))), utils::head, 20L))
  }
  list(utils::head(ids, 20L))
}

study_failure_path <- function(output) {
  # `<output stem>.failures.csv` beside the output, the stem as Python's `Path.stem` reads it.
  file.path(dirname(output), paste0(sub("(.)\\.[^.]+$", "\\1", basename(output)), ".failures.csv"))
}

study_env <- function(name, default = "") {
  value <- Sys.getenv(name, "")
  if (identical(value, "")) default else value
}

study_number <- function(name, default) {
  value <- suppressWarnings(as.numeric(study_env(name, "")))
  if (length(value) != 1L || is.na(value)) default else value
}

study_granted <- function(default) {
  requested <- suppressWarnings(as.integer(study_env("CLEVERLY_R_CORES", "")))
  if (is.na(requested) || requested < 1L) as.integer(default) else requested
}

study_calibrating <- function() {
  identical(study_env("CLEVERLY_HARNESS_CALIBRATE"), "1")
}

study_checkpoint_dir <- function() {
  dir <- study_env("CLEVERLY_CHECKPOINT", file.path(tempdir(), "cleverly-checkpoints"))
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  dir
}

study_declared_groups <- function() {
  declared <- strsplit(study_env("CLEVERLY_R_CALIBRATION_GROUPS"), ",", fixed = TRUE)[[1]]
  declared[nzchar(declared)]
}

study_runner_name <- function() {
  file <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file)) basename(sub("^--file=", "", file[[1]])) else "runner"
}

study_memory <- function() {
  # MB.  `memory.max` reads `max` when the container has no `-m`, which is no limit.
  info <- readLines("/proc/meminfo")
  field <- function(name) {
    line <- grep(sprintf("^%s:", name), info, value = TRUE)
    if (!length(line)) return(NA_real_)
    as.numeric(sub("^[^0-9]*([0-9]+).*$", "\\1", line[[1]])) / 1024
  }
  cgroup <- function(name) {
    path <- file.path("/sys/fs/cgroup", name)
    if (!file.exists(path)) return(NA_real_)
    value <- trimws(readLines(path, n = 1L, warn = FALSE))
    if (!length(value) || identical(value, "max")) return(NA_real_)
    as.numeric(value) / 1048576
  }
  list(
    total = field("MemTotal"),
    available = field("MemAvailable"),
    limit = cgroup("memory.max"),
    current = cgroup("memory.current")
  )
}

study_memory_line <- function() {
  memory <- study_memory()
  sprintf(
    "MemTotal %.0f MB, MemAvailable %.0f MB, cgroup limit %s",
    memory$total, memory$available,
    if (is.na(memory$limit)) "none" else sprintf("%.0f MB (%.0f MB used)", memory$limit, memory$current)
  )
}

study_status_mb <- function(name) {
  line <- grep(sprintf("^%s:", name), readLines("/proc/self/status"), value = TRUE)
  as.numeric(sub("^[^0-9]*([0-9]+).*$", "\\1", line[[1]])) / 1024
}

study_rss_mb <- function() study_status_mb("VmRSS")

study_hwm_mb <- function() study_status_mb("VmHWM")

study_private_mb <- function() {
  # A forked child shares its parent's pages until it writes them.  A full collection writes
  # the mark bits of every object, so after it the child's private memory includes its copy of
  # the parent heap, which is what a worker really costs.
  invisible(gc(full = TRUE))
  lines <- readLines("/proc/self/smaps_rollup")
  private <- grep("^Private_(Clean|Dirty):", lines, value = TRUE)
  sum(as.numeric(sub("^[^0-9]*([0-9]+).*$", "\\1", private))) / 1024
}

study_fork <- function(expr) {
  # One fresh child for `expr`.  A killed child returns NULL, as it does under `mclapply`.  An
  # error comes back as a `try-error` value, so it is raised again here: a fit that errors in
  # a fork must stop the run like any other error.
  job <- parallel::mcparallel(expr)
  collected <- parallel::mccollect(job, wait = TRUE)
  value <- if (is.null(collected) || !length(collected)) NULL else collected[[1]]
  if (inherits(value, "try-error")) {
    condition <- attr(value, "condition")
    if (is.null(condition)) stop(as.character(value), call. = FALSE)
    stop(condition)
  }
  value
}

study_write_state <- function(...) {
  # The driver cannot read the container's output, so the plan and any stop go to a file in
  # the work directory.  After a whole-container kill the driver reads the last worker count.
  path <- study_env("CLEVERLY_R_STATE")
  if (identical(path, "")) return(invisible(NULL))
  fields <- list(...)
  held <- if (file.exists(path)) readLines(path, warn = FALSE) else character(0)
  keys <- sub("=.*$", "", held)
  for (name in names(fields)) {
    line <- sprintf("%s=%s", name, gsub("[\r\n]+", " ", as.character(fields[[name]])))
    if (name %in% keys) held[keys == name] <- line else held <- c(held, line)
    keys <- sub("=.*$", "", held)
  }
  temporary <- sprintf("%s.%d.tmp", path, Sys.getpid())
  writeLines(held, temporary)
  file.rename(temporary, path)
  invisible(NULL)
}

study_exit <- function(status, text) {
  # Exit 3 means "a rerun resumes": a group killed twice, or a pass that made no progress.
  study_write_state(stop = text)
  message(text)
  quit(save = "no", status = status)
}

study_worker_plan <- function(granted, per_worker_mb, factor = 1.25, share = 0.85, cap = Inf,
                              basis = "measured") {
  memory <- study_memory()
  available <- memory$available
  if (!is.na(memory$limit)) available <- min(available, memory$limit - memory$current)
  bound <- min(granted, cap)
  reason <- if (cap < granted) "worker cap" else "cores"
  workers <- bound
  if (!is.na(per_worker_mb) && per_worker_mb > 0) {
    affordable <- floor(share * available / (factor * per_worker_mb))
    if (affordable < bound) {
      reason <- if (identical(basis, "declared")) "declared estimate" else "memory"
      workers <- affordable
    }
  }
  workers <- as.integer(max(1, workers))
  cat(sprintf(
    "worker plan: granted %d, worker cap %s, per-worker %s MB, factor %s, share %s, available %.0f MB, workers %d, binding %s\n",
    as.integer(granted), if (is.finite(cap)) as.character(cap) else "none",
    if (is.na(per_worker_mb)) "unmeasured" else sprintf("%.0f", per_worker_mb),
    format(factor), format(share), available, workers, reason
  ))
  list(workers = workers, reason = reason, available = available)
}

study_plan <- function(granted) {
  # The run log of a phase: what memory there is, what one worker costs, and why the worker
  # count is what it is.  The driver passes the measured cost after a calibration run.
  memory <- study_memory()
  reason <- study_env("CLEVERLY_R_MEMORY_REASON")
  cat(study_memory_line(), "\n", sep = "")
  cat(sprintf("calibration: %s\n", study_env("CLEVERLY_R_MEASURED", "not run")))
  cat(sprintf("declared per-worker MB: %s\n", study_env("CLEVERLY_R_DECLARED_MB", "none")))
  cat(sprintf("parent private MB: %.0f\n", study_private_mb()))
  if (nzchar(reason)) cat(sprintf("memory override: %s\n", reason))
  plan <- study_worker_plan(
    granted,
    study_number("CLEVERLY_R_WORKER_MB", NA_real_),
    factor = study_number("CLEVERLY_R_MEMORY_FACTOR", 1.25),
    share = study_number("CLEVERLY_R_MEMORY_SHARE", 0.85),
    cap = study_number("CLEVERLY_R_WORKER_CAP", Inf),
    basis = study_env("CLEVERLY_R_WORKER_BASIS", "measured")
  )
  study_write_state(
    workers = plan$workers, reason = plan$reason,
    mem_available_mb = sprintf("%.0f", memory$available)
  )
  plan
}

study_chunk <- function(groups, workers) {
  chunk <- study_number("CLEVERLY_R_CHUNK", NA_real_)
  if (is.na(chunk)) chunk <- ceiling(groups / (workers * 8))
  as.integer(max(1, chunk))
}

study_seeds <- function(path) {
  if (identical(path, "") || !file.exists(path)) return(NULL)
  read.csv(path, colClasses = c("character", "integer", "integer"), stringsAsFactors = FALSE)
}

study_seed_for <- function(frame, seeds) {
  # A group of one scenario takes that scenario's seed; a group that spans scenarios, or a
  # frame without a scenario column, takes the replicate's `*` row.
  if (is.null(seeds)) return(NULL)
  replicate <- unique(frame$replicate)
  if (length(replicate) != 1L) {
    stop(sprintf("a seeded group holds one replicate, not %d", length(replicate)))
  }
  scenarios <- if ("scenario" %in% names(frame)) unique(as.character(frame$scenario)) else character(0)
  if (length(scenarios) == 1L) {
    row <- which(seeds$scenario == scenarios & seeds$replicate == replicate)
    if (length(row) == 1L) return(seeds$seed[[row]])
  }
  row <- which(seeds$scenario == "*" & seeds$replicate == replicate)
  if (length(row) != 1L) stop(sprintf("the seeds file has no row for replicate %s", replicate))
  seeds$seed[[row]]
}

study_label <- function(frame) {
  # How a failure names the replication it came from.  A scenario column is not universal:
  # the studies that draw from one law do not carry one.
  if ("scenario" %in% names(frame)) {
    sprintf("%s replicate %s", frame$scenario[[1]], frame$replicate[[1]])
  } else {
    sprintf("replicate %s", frame$replicate[[1]])
  }
}

study_checkpoint_paths <- function(dir, id) {
  list(
    started = file.path(dir, paste0(id, ".started")),
    rds = file.path(dir, paste0(id, ".rds")),
    kills = file.path(dir, paste0(id, ".kills"))
  )
}

study_write_fatal <- function(dir, id, frame, condition) {
  call <- conditionCall(condition)
  lines <- c(
    sprintf("%s (group %s): %s", study_label(frame), id, conditionMessage(condition)),
    sprintf("call: %s", if (is.null(call)) "none" else paste(deparse(call), collapse = " "))
  )
  temporary <- file.path(dir, sprintf("_fatal.%d.tmp", Sys.getpid()))
  writeLines(lines, temporary)
  file.rename(temporary, file.path(dir, "_fatal"))
}

study_check_fatal <- function(dir) {
  fatal <- file.path(dir, "_fatal")
  if (file.exists(fatal)) {
    text <- paste(readLines(fatal, warn = FALSE), collapse = "\n")
    study_write_state(stop = text)
    stop(text, call. = FALSE)
  }
  invisible(NULL)
}

study_fit_group <- function(id, frame, fit_one, dir, seeds) {
  # No fit runs in the parent R process.  Called there -- by a one-core `mclapply`, a one-chunk
  # map or a solo second attempt -- it forks itself, so a kill costs one child, never the run.
  paths <- study_checkpoint_paths(dir, id)
  if (file.exists(paths$rds)) return(invisible(NULL))
  if (Sys.getpid() == study_parent_pid) {
    return(study_fork(study_fit_group(id, frame, fit_one, dir, seeds)))
  }
  if (file.exists(file.path(dir, "_fatal"))) return(invisible(NULL))
  writeLines(as.character(Sys.getpid()), paths$started)
  # The determinism probe's R profile reads which group this child fits.
  Sys.setenv(CLEVERLY_GROUP_ID = id)
  probe <- identical(study_env("CLEVERLY_HARNESS_RNG_PROBE"), "1")
  seed <- NULL
  result <- tryCatch(
    {
      seed <- study_seed_for(frame, seeds)
      if (!is.null(seed)) set.seed(seed) else if (probe) set.seed(0L)
      before <- if (probe) get(".Random.seed", envir = globalenv()) else NULL
      fitted <- fit_one(frame)
      if (probe) {
        after <- get0(".Random.seed", envir = globalenv(), inherits = FALSE)
        cat(sprintf(
          "rng-probe: %s %s %s\n", study_runner_name(), id,
          if (identical(before, after)) "untouched" else "consumed"
        ))
      }
      fitted
    },
    study_fitting_failure = function(condition) condition,
    error = function(condition) {
      study_write_fatal(dir, id, frame, condition)
      stop(condition)
    }
  )
  failure <- NULL
  attempt <- 1L + if (file.exists(paths$kills)) as.integer(readLines(paths$kills, warn = FALSE)[[1]]) else 0L
  if (inherits(result, "study_fitting_failure")) {
    failure <- study_failure_row(
      id, frame, attempt, "fitting_failure", result$stage, conditionCall(result),
      conditionMessage(result), seed
    )
  } else if (study_recording()) {
    invalid <- study_invalid(result)
    if (!is.null(invalid)) {
      failure <- study_failure_row(id, frame, attempt, "invalid_result", "row check", NULL, invalid, seed)
    }
  }
  record <- list(
    id = id,
    pid = Sys.getpid(),
    scenarios = if ("scenario" %in% names(frame)) unique(frame$scenario) else NULL,
    replicates = unique(frame$replicate)
  )
  if (is.null(failure)) record["result"] <- list(result) else record$failure <- failure
  temporary <- sprintf("%s.%d.tmp", paths$rds, Sys.getpid())
  saveRDS(record, temporary)
  file.rename(temporary, paths$rds)
  unlink(paths$started)
  if (!is.null(failure)) {
    cat(sprintf(
      "recorded %s for group %s in %s: %s\n",
      failure$class, id, failure$stage, failure$message
    ))
    study_systematic(dir, study_record_state$prefixes)
    return(invisible(NULL))
  }
  result
}

study_failures <- function(dir, ids) {
  # The failure records among the checkpoints of `ids`, in id order, or NULL.
  rows <- list()
  for (id in sort(ids)) {
    path <- study_checkpoint_paths(dir, id)$rds
    if (!file.exists(path)) next
    record <- readRDS(path)
    if (!is.null(record$failure)) rows[[length(rows) + 1L]] <- record$failure
  }
  if (!length(rows)) return(NULL)
  do.call(rbind, rows)
}

study_systematic <- function(dir, prefixes) {
  # A2's systematic stop: when every id of one 20-id prefix holds a failure record, `_fatal`
  # names the prefix and the lowest id's message, and the next `study_check_fatal` stops the
  # run.  A prefix shorter than 20 never triggers; the declared bound covers that run.
  for (index in seq_along(prefixes)) {
    prefix <- prefixes[[index]]
    if (length(prefix) < 20L) next
    rds <- vapply(prefix, function(id) study_checkpoint_paths(dir, id)$rds, character(1))
    if (!all(file.exists(rds))) next
    failures <- study_failures(dir, prefix)
    if (is.null(failures) || nrow(failures) < length(prefix)) next
    name <- names(prefixes)[index]
    scope <- if (is.null(name) || !nzchar(name) || identical(name, "*")) "the run" else name
    text <- sprintf(
      "systematic: the first %d groups of %s failed; first: %s in %s (group %s): %s",
      length(prefix), scope, failures$class[[1]], failures$stage[[1]],
      failures$group_id[[1]], failures$message[[1]]
    )
    temporary <- file.path(dir, sprintf("_fatal.%d.tmp", Sys.getpid()))
    writeLines(text, temporary)
    file.rename(temporary, file.path(dir, "_fatal"))
    return(invisible(TRUE))
  }
  invisible(FALSE)
}

study_calibrate_group <- function(id, frame, fit_one, seeds) {
  # One fresh fork per group, so the high-water mark starts at the fork and not at an earlier
  # group's peak.  A transient allocation freed before the fit returns still raises VmHWM, and
  # a retained one counts in both terms, which errs large.
  # A fitting failure comes back as its condition, so calibration can take another group.
  measured <- tryCatch(
    study_fork({
      Sys.setenv(CLEVERLY_GROUP_ID = id)
      rss_start <- study_rss_mb()
      seed <- study_seed_for(frame, seeds)
      if (!is.null(seed)) set.seed(seed)
      invisible(fit_one(frame))
      hwm_end <- study_hwm_mb()
      private <- study_private_mb()
      list(
        id = id, rss_start = rss_start, hwm_end = hwm_end, peak_growth = hwm_end - rss_start,
        private_after_gc = private, per_worker = hwm_end - rss_start + private
      )
    }),
    study_fitting_failure = function(condition) condition
  )
  if (inherits(measured, "study_fitting_failure")) return(measured)
  if (is.null(measured)) {
    stop(sprintf("calibration group %s was killed; %s", id, study_memory_line()), call. = FALSE)
  }
  measured
}

study_calibrate <- function(frames, ids, fit_one, seeds, declared = character(0)) {
  # Calibration mode only: fit the first group of each scenario, and each group the runner
  # declares heaviest, one at a time in fresh forks; write calibration.json; quit.
  if (!study_calibrating()) return(invisible(FALSE))
  seen <- character(0)
  selected <- character(0)
  holds <- lapply(frames, function(frame) {
    if ("scenario" %in% names(frame)) unique(as.character(frame$scenario)) else "*"
  })
  keys <- character(0)
  for (index in seq_along(frames)) {
    fresh <- setdiff(holds[[index]], seen)
    if (length(fresh)) {
      selected <- c(selected, ids[[index]])
      keys <- c(keys, fresh[[1]])
      seen <- c(seen, fresh)
    }
  }
  absent <- setdiff(declared, ids)
  if (length(absent)) cat(sprintf("declared calibration groups not present: %s\n", paste(absent, collapse = ", ")))
  heaviest <- setdiff(intersect(declared, ids), selected)
  measures <- list()
  # Decision Q3: a first group whose fit fails is replaced by the next group of its scenario, up
  # to the 20-group prefix; 20 failures, or a scenario with no group left, stop as systematic.
  for (position in seq_along(selected)) {
    start <- match(selected[[position]], ids)
    key <- keys[[position]]
    candidates <- which(vapply(holds, function(held) key %in% held, logical(1)))
    candidates <- utils::head(candidates[candidates >= start], 20L)
    first <- NULL
    measured <- NULL
    for (index in candidates) {
      attempt <- study_calibrate_group(ids[[index]], frames[[index]], fit_one, seeds)
      if (!inherits(attempt, "study_fitting_failure")) {
        measured <- attempt
        break
      }
      if (is.null(first)) first <- sprintf("%s (group %s): %s", attempt$stage, ids[[index]], conditionMessage(attempt))
      cat(sprintf(
        "calibration group %s failed in %s: %s; taking the next group of %s\n",
        ids[[index]], attempt$stage, gsub("[\r\n]", " ", conditionMessage(attempt)), key
      ))
    }
    if (is.null(measured)) {
      stop(sprintf(
        "systematic: the first %d groups of %s failed in calibration; first: %s",
        length(candidates), if (identical(key, "*")) "the run" else key, first
      ), call. = FALSE)
    }
    measures[[length(measures) + 1L]] <- measured
  }
  for (id in heaviest) {
    attempt <- study_calibrate_group(id, frames[[match(id, ids)]], fit_one, seeds)
    if (inherits(attempt, "study_fitting_failure")) {
      cat(sprintf("declared calibration group %s failed in %s and is not measured: %s\n",
                  id, attempt$stage, gsub("[\r\n]", " ", conditionMessage(attempt))))
    } else {
      measures[[length(measures) + 1L]] <- attempt
    }
  }
  parent <- study_private_mb()
  memory <- study_memory()
  number <- function(value) if (is.na(value)) "null" else sprintf("%.3f", value)
  entries <- vapply(measures, function(m) {
    sprintf(
      "{\"id\": \"%s\", \"rss_start\": %s, \"hwm_end\": %s, \"peak_growth\": %s, \"private_after_gc\": %s, \"per_worker\": %s}",
      m$id, number(m$rss_start), number(m$hwm_end), number(m$peak_growth),
      number(m$private_after_gc), number(m$per_worker)
    )
  }, character(1))
  json <- c(
    "{",
    sprintf("  \"groups\": [%s],", paste(entries, collapse = ", ")),
    sprintf("  \"parent_private_mb\": %s,", number(parent)),
    sprintf(
      "  \"memory\": {\"total\": %s, \"available\": %s, \"limit\": %s, \"current\": %s}",
      number(memory$total), number(memory$available), number(memory$limit), number(memory$current)
    ),
    "}"
  )
  path <- study_env("CLEVERLY_R_CALIBRATION", file.path(tempdir(), "calibration.json"))
  writeLines(json, path)
  for (m in measures) {
    cat(sprintf(
      "calibration %s: peak_growth %.0f MB, private_after_gc %.0f MB, per_worker %.0f MB\n",
      m$id, m$peak_growth, m$private_after_gc, m$per_worker
    ))
  }
  cat(sprintf("calibration: parent private %.0f MB; %s\n", parent, study_memory_line()))
  quit(save = "no", status = 0)
}

study_map <- function(ids, fun, workers, chunk) {
  # Dynamic scheduling in chunks: a free worker takes the next chunk, and a kill loses the
  # group in flight and leaves the rest of its chunk unstarted for the next pass.
  chunks <- split(ids, ceiling(seq_along(ids) / chunk))
  invisible(parallel::mclapply(
    chunks,
    function(part) {
      for (id in part) fun(id)
      length(part)
    },
    mc.cores = workers,
    mc.preschedule = FALSE
  ))
}

study_startup <- function(dir, meta, ids = NULL) {
  # Called once in the parent before any fit.  The leaf's meta copy is the resume key: a leaf
  # written under another key (in practice, another image) refuses.
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  copy <- file.path(dir, "_meta.json")
  if (!identical(meta, "") && file.exists(meta)) {
    wanted <- readLines(meta, warn = FALSE)
    if (file.exists(copy)) {
      held <- readLines(copy, warn = FALSE)
      if (!identical(held, wanted)) {
        stop(sprintf(
          "the checkpoints in %s were written under another resume key: %s",
          dir, paste(trimws(setdiff(held, wanted)), collapse = "; ")
        ), call. = FALSE)
      }
    } else {
      writeLines(wanted, copy)
    }
  }
  stale <- list.files(dir, pattern = "\\.started$", full.names = TRUE)
  unlink(stale)
  cat(sprintf("cleared %d stale start markers without counting them as kills\n", length(stale)))
  fatal <- file.path(dir, "_fatal")
  if (file.exists(fatal)) {
    cat("cleared a fatal error left by an earlier run:\n", paste(readLines(fatal, warn = FALSE), collapse = "\n"), "\n", sep = "")
    unlink(fatal)
  }
  unlink(list.files(dir, pattern = "\\.tmp$", full.names = TRUE))
  saved <- list.files(dir, pattern = "\\.rds$", full.names = TRUE)
  readable <- vapply(saved, function(path) {
    !inherits(try(readRDS(path), silent = TRUE), "try-error")
  }, logical(1))
  if (any(!readable)) cat(sprintf("deleted %d unreadable checkpoints\n", sum(!readable)))
  unlink(saved[!readable])
  reused <- sum(readable)
  cat(sprintf(
    "resumed: reused %d groups, fitting %s\n", reused,
    if (is.null(ids)) "the rest" else as.character(length(ids) - reused)
  ))
  study_write_state(reused = reused)
  # A resume over failure checkpoints may already hold a systematic prefix.
  study_systematic(dir, study_record_state$prefixes)
  invisible(reused)
}

study_retry <- function(ids, fun, dir, workers, chunk) {
  # Pass after pass until every group has a checkpoint.  A group with a start marker and no
  # checkpoint was killed in flight: the next pass halves the workers and fits it alone.  A
  # second kill, or a pass that changes nothing, stops with exit 3, and a rerun resumes.
  no_retry <- identical(study_env("CLEVERLY_R_NO_RETRY"), "1")
  cat(sprintf(
    "retry loop: %d groups, workers %d, chunk %d\n",
    length(ids), as.integer(workers), as.integer(chunk)
  ))
  first <- TRUE
  before <- -1L
  repeat {
    study_check_fatal(dir)
    paths <- lapply(ids, study_checkpoint_paths, dir = dir)
    done <- vapply(paths, function(p) file.exists(p$rds), logical(1))
    if (all(done)) break
    pending <- ids[!done]
    pending_paths <- paths[!done]
    started <- vapply(pending_paths, function(p) file.exists(p$started), logical(1))
    killed <- pending[started]
    for (p in pending_paths[started]) {
      kills <- if (file.exists(p$kills)) as.integer(readLines(p$kills, warn = FALSE)[[1]]) else 0L
      writeLines(as.character(kills + 1L), p$kills)
      unlink(p$started)
    }
    kills <- vapply(pending_paths, function(p) {
      if (file.exists(p$kills)) as.integer(readLines(p$kills, warn = FALSE)[[1]]) else 0L
    }, integer(1))
    if (length(killed) && no_retry) {
      stop(sprintf(
        "killed: %s; CLEVERLY_R_NO_RETRY=1 makes a kill fatal; %s",
        paste(killed, collapse = ", "), study_memory_line()
      ), call. = FALSE)
    }
    twice <- pending[kills >= 2L]
    if (length(twice)) {
      study_exit(3L, sprintf(
        "incomplete: %s were killed twice; %s", paste(twice, collapse = ", "), study_memory_line()
      ))
    }
    if (!first && sum(done) == before && !length(killed)) {
      study_exit(3L, sprintf(
        "no progress: %s; %s", paste(utils::head(pending, 20), collapse = ", "), study_memory_line()
      ))
    }
    before <- sum(done)
    if (length(killed)) workers <- max(1L, workers %/% 2L)
    memory <- study_memory()
    cat(sprintf(
      "pass: %d pending, %d killed, MemAvailable %.0f MB, workers %d, chunk %d\n",
      length(pending), length(killed), memory$available, as.integer(workers), as.integer(chunk)
    ))
    study_write_state(workers = workers, mem_available_mb = sprintf("%.0f", memory$available))
    second <- pending[kills == 1L]
    rest <- setdiff(pending, second)
    if (length(rest)) study_map(rest, fun, workers, chunk)
    for (id in second) fun(id)
    first <- FALSE
  }
  as.integer(workers)
}

study_assemble <- function(dir, ids) {
  # Zero-padded ids, so string order is numeric order and the table keeps the group order.  A
  # failure checkpoint holds no result; `study_failures` reads it.
  records <- lapply(sort(ids), function(id) readRDS(study_checkpoint_paths(dir, id)$rds))
  kept <- vapply(records, function(record) is.null(record$failure), logical(1))
  lapply(records[kept], function(record) record$result)
}

study_accounted <- function(out, failures, observed, expected, groups) {
  # The keys a run accounts for: `observed` from the table, plus each failed group's share of
  # `expected` (B3: counted by group, so a record without a scenario column needs no expansion).
  # A failed group whose replication also has rows refuses.
  if (is.null(failures) || !nrow(failures)) return(observed)
  per_group <- expected / groups
  if (per_group != floor(per_group)) {
    stop(sprintf(
      "%s keys over %d groups is not a whole number per group, so a failed replication cannot be counted",
      format(expected), as.integer(groups)
    ), call. = FALSE)
  }
  if (!is.null(out) && "replicate" %in% names(out)) {
    for (row in seq_len(nrow(failures))) {
      replicates <- as.integer(strsplit(failures$replicates[[row]], ";", fixed = TRUE)[[1]])
      scenarios <- strsplit(failures$scenarios[[row]], ";", fixed = TRUE)[[1]]
      hit <- out$replicate %in% replicates
      if ("scenario" %in% names(out) && !identical(scenarios, "*")) hit <- hit & out$scenario %in% scenarios
      if (any(hit)) {
        stop(sprintf(
          "failed group %s covers a replication that also has rows", failures$group_id[[row]]
        ), call. = FALSE)
      }
    }
  }
  observed + per_group * nrow(failures)
}

study_write_failures <- function(output, failures) {
  # The failures file exists only when a record does, so a run without one writes no new file.
  path <- study_failure_path(output)
  unlink(path)
  if (!is.null(failures) && nrow(failures)) write.csv(failures, path, row.names = FALSE)
  invisible(path)
}

study_finish <- function(dir, ids, output, count, expected, na = "NA",
                         dropped = "wrote %d of %d replications") {
  # The tail of a runner that stays off the harness: assemble the successes, refuse a worker
  # that returned no result, count `count(table)` plus the failed groups' keys against
  # `expected`, write the failures file when a record exists, and write the table.  `dropped`
  # is the runner's own stop text; a `%d` pair in it takes the accounted and expected counts.
  unlink(study_failure_path(output))
  study_check_fatal(dir)
  results <- study_assemble(dir, ids)
  failures <- study_failures(dir, ids)
  failed <- !vapply(results, is.data.frame, logical(1))
  if (any(failed)) stop(sprintf("%d R worker(s) returned no result", sum(failed)))
  if (!is.null(failures) && !length(results)) {
    stop(sprintf("every group failed; first: %s", failures$message[[1]]), call. = FALSE)
  }
  out <- do.call(rbind, results)
  accounted <- study_accounted(out, failures, count(out), expected, length(ids))
  if (accounted != expected) {
    stop(if (grepl("%d", dropped, fixed = TRUE)) {
      sprintf(dropped, as.integer(accounted), as.integer(expected))
    } else {
      dropped
    })
  }
  study_write_failures(output, failures)
  write.csv(out, output, row.names = FALSE, na = na)
  invisible(out)
}
# ---- RM39 shared resilience functions: end ----

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: run_ctmle3_oat.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(args[[1]]), stringsAsFactors = FALSE)
truths <- read.csv(args[[2]], stringsAsFactors = FALSE)
output <- args[[3]]
truth_key <- paste(truths$scenario, truths$replicate, sep = "|")

wald <- function(estimate, influence, transform = "identity") {
  n <- length(influence)
  se <- stats::sd(influence) / sqrt(n)
  z <- stats::qnorm(0.975)
  if (transform == "identity") {
    ci <- estimate + c(-1, 1) * z * se
    return(list(estimate = estimate, inference = estimate, se = se, ci = ci))
  }
  log_estimate <- log(estimate)
  ci <- exp(log_estimate + c(-1, 1) * z * se)
  list(estimate = estimate, inference = log_estimate, se = se, ci = ci)
}

fit_one <- function(frame, scenario, replicate) {
  row <- match(paste(scenario, replicate, sep = "|"), truth_key)
  if (is.na(row)) stop(sprintf("no truth for %s replicate %s", scenario, replicate))
  covariates <- c("W1", "W2", "W3")
  data <- frame[c(covariates, "A", "Y")]
  nodes <- list(W = covariates, A = "A", Y = "Y")
  learners <- list(A = Lrnr_glm$new(), Y = Lrnr_glm$new())
  fit <- study_stage("tmle3 oat", tmle3(tmle_oat_TSM_all(), data, nodes, learners))
  estimates <- fit$estimates
  if (length(estimates) != 2) stop("the two-arm OAT comparison did not return two means")
  psi <- vapply(estimates, `[[`, numeric(1), "psi")
  influence <- do.call(cbind, lapply(estimates, `[[`, "IC"))
  initial <- as.numeric(fit$initial_psi)
  truth <- function(name) truths[[paste0("truth_", name)]][[row]]
  make_row <- function(name, value, curve, native, initial_value) {
    result <- wald(value, curve, native)
    reference <- truth(name)
    data.frame(
      implementation = "tlverse-ctmle3-oat",
      scenario = scenario,
      replicate = replicate,
      n = nrow(frame),
      estimand = name,
      truth = reference,
      estimate = result$estimate,
      inference_estimate = result$inference,
      std_error = result$se,
      ci_lower = result$ci[[1]],
      ci_upper = result$ci[[2]],
      inference_scale = native,
      covered = as.integer(result$ci[[1]] <= reference && reference <= result$ci[[2]]),
      initial_estimate = initial_value,
      stringsAsFactors = FALSE
    )
  }
  rows <- list(
    make_row("ey0", psi[[1]], influence[, 1], "identity", initial[[1]]),
    make_row("ey1", psi[[2]], influence[, 2], "identity", initial[[2]]),
    make_row("ate", psi[[2]] - psi[[1]], influence[, 2] - influence[, 1], "identity", initial[[2]] - initial[[1]])
  )
  if (scenario == "binary") {
    rr <- psi[[2]] / psi[[1]]
    rr_curve <- influence[, 2] / psi[[2]] - influence[, 1] / psi[[1]]
    odds <- function(x) x / (1 - x)
    or <- odds(psi[[2]]) / odds(psi[[1]])
    or_curve <- influence[, 2] / (psi[[2]] * (1 - psi[[2]])) -
      influence[, 1] / (psi[[1]] * (1 - psi[[1]]))
    rows[[length(rows) + 1]] <- make_row("rr", rr, rr_curve, "log", initial[[2]] / initial[[1]])
    rows[[length(rows) + 1]] <- make_row("or", or, or_curve, "log", odds(initial[[2]]) / odds(initial[[1]]))
  }
  do.call(rbind, rows)
}

groups <- split(samples, interaction(samples$scenario, samples$replicate, drop = TRUE))
fit_group <- function(frame) fit_one(frame, frame$scenario[[1]], frame$replicate[[1]])
# RM39: the shared functions above fit each group in a forked child, checkpoint it, retry a
# killed group once and resume a rerun from the checkpoints.  An error inside `study_stage`
# is recorded as a failure of its group (RM39a2); every other error is still fatal and names
# its group in `_fatal`.
ids <- sprintf("g%09d", seq_along(groups))
seeds <- study_seeds(study_env("CLEVERLY_R_SEEDS"))
study_record_state$prefixes <- study_prefixes(groups, ids)
study_calibrate(groups, ids, fit_group, seeds, study_declared_groups())
dir <- study_checkpoint_dir()
study_startup(dir, study_env("CLEVERLY_R_META"), ids)
plan <- study_plan(study_granted(1L))
fit_id <- function(id) study_fit_group(id, groups[[match(id, ids)]], fit_group, dir, seeds)
study_retry(ids, fit_id, dir, plan$workers, study_chunk(length(ids), plan$workers))
study_finish(
  dir, ids, output,
  count = nrow,
  expected = 5L * length(groups),
  dropped = "an OAT reference replication was silently dropped"
)
cat("ctmle3 ", as.character(packageVersion("ctmle3")), "\n", sep = "")
cat("tmle3 ", as.character(packageVersion("tmle3")), "\n", sep = "")
cat("sl3 ", as.character(packageVersion("sl3")), "\n", sep = "")
cat("R ", R.version.string, "\n", sep = "")
