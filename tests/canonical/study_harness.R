# The collection half of every canonical reference runner.
#
# A runner's own half is its fits.  The other half -- take three paths off the command line,
# split the rows by replication, fit them across the cores the driver granted, refuse the run
# if a worker failed or a replication went missing, write the table and print what was pinned
# -- is the same for all of them, and it was written out twelve times before this file
# existed.
#
# The cost of that was not the duplication.  It was that a fix landed in the copy the author
# happened to be in.  The tmle3 runner grew the memory cap and the malformed-worker check
# below after a run lost 86 of 3,200 replications while reporting success; the runners cloned
# from other copies did not.  Two of them later grew the cap without the reasoning, and one
# still has neither.
#
# Every divergence between the old copies is an argument here rather than a branch, so a study
# keeps the semantics it published under.  The table does not depend on how many workers ran,
# which is what lets the core count be decided per machine.  Each fit runs after `set.seed`
# with the seed the driver declared for its replicate, so a resumed or retried group draws what
# it drew the first time.  A runner that seeds inside its own fit overrides that seed -- see
# tests/canonical/ctmle_selector/run_ctmle.R, which does, and says why.
#
# RM39 made the run survive a killed worker and a killed container.  Each group's result is
# checkpointed as it finishes, a killed group is retried once, and a rerun resumes from the
# checkpoints.  Every error is still fatal.

study_arguments <- function(usage) {
  # The driver passes exactly samples, truths and output, in that order.  A runner that reads
  # them positionally without checking writes its results over one of its inputs the day the
  # contract changes.
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 3) stop(usage)
  list(samples = args[[1]], truths = args[[2]], output = args[[3]])
}

#: What `study_fitter` or `study_stream` registered for `study_collect`.
study_state <- new.env(parent = emptyenv())

# ---- RM39 shared resilience functions: begin ----
# tests/canonical/study_harness.R and the three runners that stay off it (ctmle3_oat,
# ctmle_selector, drtmle) carry this block byte for byte; tests/unit/test_harness_drift.py
# compares the four copies.  Change it here and copy it to the other three.
#
# What it does: every fit runs in a forked child, writes a start marker, and leaves its result
# in a checkpoint file.  A killed child leaves the marker and no checkpoint, so the retry loop
# can name it, run it again alone, and stop when it is killed twice.  A rerun of the container
# reuses every checkpoint.  The worker count comes from measured memory, not from the input
# size.  Every error is still fatal: the first one stops the run with its message.

study_parent_pid <- Sys.getpid()

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
  probe <- identical(study_env("CLEVERLY_HARNESS_RNG_PROBE"), "1")
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
    error = function(condition) {
      study_write_fatal(dir, id, frame, condition)
      stop(condition)
    }
  )
  record <- list(
    id = id,
    pid = Sys.getpid(),
    scenarios = if ("scenario" %in% names(frame)) unique(frame$scenario) else NULL,
    replicates = unique(frame$replicate),
    result = result
  )
  temporary <- sprintf("%s.%d.tmp", paths$rds, Sys.getpid())
  saveRDS(record, temporary)
  file.rename(temporary, paths$rds)
  unlink(paths$started)
  result
}

study_calibrate_group <- function(id, frame, fit_one, seeds) {
  # One fresh fork per group, so the high-water mark starts at the fork and not at an earlier
  # group's peak.  A transient allocation freed before the fit returns still raises VmHWM, and
  # a retained one counts in both terms, which errs large.
  measured <- study_fork({
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
  })
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
  for (index in seq_along(frames)) {
    frame <- frames[[index]]
    scenarios <- if ("scenario" %in% names(frame)) unique(as.character(frame$scenario)) else "*"
    fresh <- setdiff(scenarios, seen)
    if (length(fresh)) {
      selected <- c(selected, ids[[index]])
      seen <- c(seen, fresh)
    }
  }
  absent <- setdiff(declared, ids)
  if (length(absent)) cat(sprintf("declared calibration groups not present: %s\n", paste(absent, collapse = ", ")))
  selected <- unique(c(selected, intersect(declared, ids)))
  measures <- lapply(selected, function(id) {
    study_calibrate_group(id, frames[[match(id, ids)]], fit_one, seeds)
  })
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
  invisible(reused)
}

study_retry <- function(ids, fun, dir, workers, chunk) {
  # Pass after pass until every group has a checkpoint.  A group with a start marker and no
  # checkpoint was killed in flight: the next pass halves the workers and fits it alone.  A
  # second kill, or a pass that changes nothing, stops with exit 3, and a rerun resumes.
  no_retry <- identical(study_env("CLEVERLY_R_NO_RETRY"), "1")
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
  # Zero-padded ids, so string order is numeric order and the table keeps the group order.
  lapply(sort(ids), function(id) readRDS(study_checkpoint_paths(dir, id)$rds)$result)
}
# ---- RM39 shared resilience functions: end ----

study_cores <- function(groups) {
  # Capped by memory as well as by cores.  `mclapply` forks, and a forked worker that the
  # kernel kills does not raise: it returns a non-data-frame that `rbind` drops without a
  # word, which is how a run once lost 86 of 3,200 replications while reporting success.
  # `study_collect` refuses that outcome, and the retry loop refits the group; this cap and
  # the measured worker plan are what stop it happening.  In calibration mode the plan is not
  # read: `mclapply` evaluates its core count before its function, so this runs first.
  if (study_calibrating()) return(1L)
  cores <- study_granted(max(1L, parallel::detectCores()))
  memory_kb <- as.numeric(
    sub("[^0-9]*([0-9]+).*", "\\1", grep("MemTotal", readLines("/proc/meminfo"), value = TRUE))
  )
  footprint_kb <- as.numeric(utils::object.size(groups)) / 1024
  affordable <- max(1L, as.integer(floor(memory_kb * 0.5 / max(footprint_kb * 0.5, 1))))
  if (affordable < cores) {
    cat(sprintf("capping %d cores to %d for memory\n", cores, affordable))
    cores <- affordable
  }
  cores <- study_plan(cores)$workers
  study_state$workers <- cores
  cat(sprintf("fitting %d samples on %d cores\n", length(groups), cores))
  cores
}

study_fitter <- function(groups, fit_one, progress_every = 10L) {
  # Returns the function `mclapply` is handed.  Each call fits one group in a forked child and
  # checkpoints its result.  `study_collect` then retries what a kill lost and assembles the
  # table from the checkpoints, so what `mclapply` hands back is only a progress signal.
  ids <- sprintf("g%09d", seq_along(groups))
  seeds <- study_seeds(study_env("CLEVERLY_R_SEEDS"))
  study_calibrate(groups, ids, fit_one, seeds, study_declared_groups())
  dir <- study_checkpoint_dir()
  study_startup(dir, study_env("CLEVERLY_R_META"), ids)
  expected <- length(groups)
  study_state$ids <- ids
  study_state$dir <- dir
  study_state$stream <- FALSE
  study_state$fit <- function(id) {
    study_fit_group(id, groups[[match(id, ids)]], fit_one, dir, seeds)
  }
  function(index) {
    result <- study_fit_group(ids[[index]], groups[[index]], fit_one, dir, seeds)
    if (index %% progress_every == 0) {
      cat(sprintf("completed %d/%d samples\n", index, expected))
    }
    result
  }
}

study_refuse_failures <- function(results) {
  # Applied by `study_collect` to the assembled table.  An error stops the run before it
  # leaves a checkpoint, so this refusal is a second line, kept for a value that carries the
  # class without having raised.
  failed <- vapply(results, inherits, logical(1), what = "study_error")
  if (!any(failed)) {
    return(invisible(NULL))
  }
  messages <- vapply(
    results[failed],
    function(error) {
      if (is.null(error$index) || is.na(error$index)) {
        sprintf("%s: %s", error$label, error$message)
      } else {
        sprintf("%s (group %s): %s", error$label, error$index, error$message)
      }
    },
    character(1)
  )
  stop(paste(messages, collapse = "\n"))
}

study_stream <- function(sample_path, fit_one, expected, chunk_lines = 64000L) {
  # For a study whose sample archive does not fit in memory beside its fits.  Reads the archive
  # in replicate-aligned chunks and fits each chunk across the planned workers, so only one
  # chunk is ever resident.  The last replication in a chunk is carried forward rather than
  # fitted, because a panel split across two reads is not a panel.
  #
  # A kill is retried inside its chunk, while the chunk is still in memory, and the halved
  # worker count carries into the next chunk.  Results stay in checkpoints, not in memory, and
  # `study_collect` assembles them.  Calibration fits the first chunk only and quits.
  calibrating <- study_calibrating()
  seeds <- study_seeds(study_env("CLEVERLY_R_SEEDS"))
  dir <- study_checkpoint_dir()
  workers <- 1L
  if (!calibrating) {
    study_startup(dir, study_env("CLEVERLY_R_META"))
    workers <- study_plan(study_granted(max(1L, parallel::detectCores())))$workers
    cat(sprintf("fitting %d samples on %d cores\n", expected, workers))
  }
  connection <- gzfile(sample_path, open = "rt")
  on.exit(close(connection), add = TRUE)
  header <- readLines(connection, n = 1L)
  carry <- NULL
  every <- character(0)
  highest <- -Inf
  completed <- 0L
  repeat {
    lines <- readLines(connection, n = chunk_lines)
    at_end <- length(lines) == 0L
    current <- if (at_end) {
      NULL
    } else {
      as.data.frame(data.table::fread(text = paste(c(header, lines), collapse = "\n")))
    }
    frame <- if (is.null(carry)) current else if (is.null(current)) carry else rbind(carry, current)
    carry <- NULL
    if (!is.null(frame) && nrow(frame) && !at_end) {
      last_replicate <- frame$replicate[[nrow(frame)]]
      carry <- frame[frame$replicate == last_replicate, ]
      frame <- frame[frame$replicate != last_replicate, ]
    }
    if (!is.null(frame) && nrow(frame)) {
      groups <- split(frame, frame$replicate)
      replicates <- as.numeric(names(groups))
      # A group id is its replicate, so a replicate that reappears in a later chunk would
      # overwrite a checkpoint.  The archives are written in replicate order; refuse otherwise.
      if (min(replicates) <= highest) {
        stop(sprintf(
          "study_stream needs the samples in replicate order: replicate %s follows %s",
          min(replicates), highest
        ))
      }
      highest <- max(replicates)
      ids <- sprintf("r%09d", as.integer(replicates))
      study_calibrate(groups, ids, fit_one, seeds, study_declared_groups())
      fit <- function(id) study_fit_group(id, groups[[match(id, ids)]], fit_one, dir, seeds)
      workers <- study_retry(ids, fit, dir, workers, study_chunk(length(ids), workers))
      every <- c(every, ids)
      completed <- completed + length(ids)
      cat(sprintf("completed %d/%d samples\n", completed, expected))
    }
    if (at_end) break
  }
  study_state$ids <- every
  study_state$dir <- dir
  study_state$stream <- TRUE
  invisible(list())
}

study_collect <- function(results, expected, output, versions, key = "replicate", na = "") {
  # The registered groups are retried until each has a checkpoint, and the table is assembled
  # from the checkpoints.  The `results` argument can hold NULLs from killed slices, so the
  # three refusals below apply to the assembled list.  They are not the same refusal.  A worker
  # that returned no result at all was killed rather than having errored, so it carries no
  # message and has to be reported by index.  A worker that errored carries one.  A run where
  # neither happened can still be short, because `rbind` of a list with a NULL in it is simply
  # narrower, so the replication count is checked against what was drawn rather than against
  # what came back.
  if (!is.null(study_state$ids)) {
    if (!isTRUE(study_state$stream)) {
      workers <- study_state$workers
      if (is.null(workers)) workers <- study_plan(study_granted(max(1L, parallel::detectCores())))$workers
      study_state$workers <- study_retry(
        study_state$ids, study_state$fit, study_state$dir, workers,
        study_chunk(length(study_state$ids), workers)
      )
    }
    study_check_fatal(study_state$dir)
    results <- study_assemble(study_state$dir, study_state$ids)
  }
  malformed <- which(
    !vapply(results, is.data.frame, logical(1)) &
      !vapply(results, inherits, logical(1), what = "study_error")
  )
  if (length(malformed)) {
    stop(sprintf(
      "%d of %d workers returned no result at all (groups %s); they were killed rather than erroring",
      length(malformed), expected, paste(utils::head(malformed, 10), collapse = ", ")
    ))
  }
  study_refuse_failures(results)
  out <- do.call(rbind, results)
  observed <- length(unique(do.call(paste, unname(as.list(out[key])))))
  if (observed != expected) {
    stop(sprintf(
      "wrote %d of %d replications; a silently dropped replication is not a shorter study",
      observed, expected
    ))
  }
  write.csv(out, output, row.names = FALSE, na = na)
  for (line in versions) cat(line, "\n", sep = "")
  cat("R ", R.version.string, "\n", sep = "")
  invisible(out)
}

study_version <- function(package) {
  paste(package, as.character(packageVersion(package)))
}
