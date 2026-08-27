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
# keeps the semantics it published under.  Nothing in this file consumes randomness or depends
# on how many workers ran, which is what lets the core count be decided per machine: every
# runner that sources this fits with deterministic learners.  A study whose search consumes
# randomness must seed inside its own worker before it can use this -- see
# tests/canonical/ctmle_selector/run_ctmle.R, which does, and says why.

study_arguments <- function(usage) {
  # The driver passes exactly samples, truths and output, in that order.  A runner that reads
  # them positionally without checking writes its results over one of its inputs the day the
  # contract changes.
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 3) stop(usage)
  list(samples = args[[1]], truths = args[[2]], output = args[[3]])
}

study_cores <- function(groups) {
  # Capped by memory as well as by cores.  `mclapply` forks, and a forked worker that the
  # kernel kills does not raise: it returns a non-data-frame that `rbind` drops without a
  # word, which is how a run once lost 86 of 3,200 replications while reporting success.
  # `study_collect` refuses that outcome; this cap is what stops it happening.
  requested <- suppressWarnings(as.integer(Sys.getenv("CLEVERLY_R_CORES", "")))
  cores <- if (is.na(requested) || requested < 1L) {
    max(1L, parallel::detectCores())
  } else {
    requested
  }
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
  cores
}

study_fitter <- function(groups, fit_one, progress_every = 10L) {
  # Returns the function `mclapply` is handed.  A replication that raises becomes a value with
  # a class on it rather than an error, because `mclapply` would otherwise hand back a
  # `try-error` whose message says nothing about which replication produced it.
  expected <- length(groups)
  function(index) {
    frame <- groups[[index]]
    tryCatch(
      {
        result <- fit_one(frame)
        if (index %% progress_every == 0) {
          cat(sprintf("completed %d/%d samples\n", index, expected))
        }
        result
      },
      error = function(condition) {
        structure(
          list(
            index = index,
            label = study_label(frame),
            message = conditionMessage(condition)
          ),
          class = "study_error"
        )
      }
    )
  }
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

study_refuse_failures <- function(results) {
  # Applied to a whole run by `study_collect` and to each batch by `study_stream`.  A streaming
  # study checks per batch on purpose: its runs are the long ones, and a replication that
  # raises in the first chunk should not be discovered an hour later.
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
  # in replicate-aligned chunks and fits each chunk across the granted cores, so only one chunk
  # is ever resident.  The last replication in a chunk is carried forward rather than fitted,
  # because a panel split across two reads is not a panel.
  #
  # No memory cap here: the chunk *is* the memory strategy, so measuring the footprint of one
  # chunk and dividing would cap the cores against a number this function already chose.
  requested <- suppressWarnings(as.integer(Sys.getenv("CLEVERLY_R_CORES", "")))
  cores <- if (is.na(requested) || requested < 1L) {
    max(1L, parallel::detectCores())
  } else {
    requested
  }
  cat(sprintf("fitting %d samples on %d cores\n", expected, cores))
  fit_group <- function(frame) {
    tryCatch(
      fit_one(frame),
      error = function(condition) {
        structure(
          list(index = NA_integer_, label = study_label(frame), message = conditionMessage(condition)),
          class = "study_error"
        )
      }
    )
  }
  connection <- gzfile(sample_path, open = "rt")
  on.exit(close(connection), add = TRUE)
  header <- readLines(connection, n = 1L)
  carry <- NULL
  results <- list()
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
      batch <- parallel::mclapply(groups, fit_group, mc.cores = cores, mc.preschedule = TRUE)
      study_refuse_failures(batch)
      results <- c(results, batch)
      completed <- completed + length(batch)
      cat(sprintf("completed %d/%d samples\n", completed, expected))
    }
    if (at_end) break
  }
  results
}

study_collect <- function(results, expected, output, versions, key = "replicate", na = "") {
  # Three refusals, and they are not the same refusal.  A worker that returned no result at
  # all was killed rather than having errored, so it carries no message and has to be reported
  # by index.  A worker that errored carries one.  A run where neither happened can still be
  # short, because `rbind` of a list with a NULL in it is simply narrower, so the replication
  # count is checked against what was drawn rather than against what came back.
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
