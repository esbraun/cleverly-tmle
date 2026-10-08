# RM19 rule 1: R's loop round count per draw, around the unchanged canonical runner.
#
# usage: Rscript count_rounds.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv
#
# The pinned image mounts tests/ at /fixture.  This wrapper counts the calls of drtmle's
# internal fluctuateG, one in each round of its targeting loop (drtmle.R:607-620), and records
# the count of each fit keyed by the scenario and replicate of the runner's fit_one.  It then
# sources run_drtmle.R unchanged, which reads the same three arguments, fits every draw, stops
# when a worker returns no result, and writes OUTPUT.csv.  The counts go to
# OUTPUT-rounds.csv; a draw without exactly one count stops the wrapper.
#
# Since RM39 the runner retries a group whose worker was killed.  A retried group would count
# its rounds twice, so the wrapper sets CLEVERLY_R_NO_RETRY=1, under which the first kill stops
# the run.  Each fit runs in its own fork, so the per-PID files below hold one draw each.
Sys.setenv(CLEVERLY_R_NO_RETRY = "1")
suppressPackageStartupMessages(library(drtmle))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: count_rounds.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
output <- args[[3]]
counts <- file.path(dirname(output), "round-counts")
dir.create(counts, showWarnings = FALSE)
unlink(file.path(counts, "*"))

ns <- asNamespace("drtmle")
tally <- new.env()
tally$rounds <- 0L
fluctuate_g <- get("fluctuateG", envir = ns)
utils::assignInNamespace("fluctuateG", function(...) {
  tally$rounds <- tally$rounds + 1L
  fluctuate_g(...)
}, ns = "drtmle")
fit_drtmle <- get("drtmle", envir = ns)
utils::assignInNamespace("drtmle", function(...) {
  # The caller is run_drtmle.R's fit_one, which holds the draw's scenario and replicate.
  caller <- parent.frame()
  tally$rounds <- 0L
  fit <- fit_drtmle(...)
  cat(sprintf("%s,%d,%d\n", get("scenario", envir = caller),
              as.integer(get("replicate", envir = caller)), tally$rounds),
      file = file.path(counts, sprintf("%d.csv", Sys.getpid())), append = TRUE)
  fit
}, ns = "drtmle")

source("/fixture/canonical/drtmle/run_drtmle.R")

rows <- read.csv(output, stringsAsFactors = FALSE)
parts <- list.files(counts, full.names = TRUE)
if (length(parts) == 0) stop("no round count was recorded")
rounds <- do.call(rbind, lapply(parts, function(path) {
  read.csv(path, header = FALSE, col.names = c("scenario", "replicate", "rounds"),
           stringsAsFactors = FALSE)
}))
draws <- unique(rows[c("scenario", "replicate")])
keys <- paste(rounds$scenario, rounds$replicate)
if (anyDuplicated(keys)) stop("a draw has more than one round count")
if (!setequal(keys, paste(draws$scenario, draws$replicate))) {
  stop(sprintf("round counts for %d draws, rows for %d", length(keys), nrow(draws)))
}
write.csv(rounds, sub("\\.csv$", "-rounds.csv", output), row.names = FALSE)
cat("rounds ", nrow(rounds), "\n", sep = "")
