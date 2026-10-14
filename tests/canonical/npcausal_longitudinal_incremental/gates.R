# The pairing gates of `longitudinal-incremental` (X19-plan.md 3.4, 6.1 and 6.8).  For each
# (scenario, replicate) of the sample archive, one fold-injected full-grid `ipsi` call, and its
# estimate, standard error, `calpha` and every unit's centred influence value.  The Python side
# (`gates.py`) compares them with the cleverly fit of the same rows.
#
# usage: gates.R SAMPLES.csv.gz DELTAS OUTPUT_ESTIMATES.csv OUTPUT_CURVES.csv.gz
suppressPackageStartupMessages(library(npcausal))
source("/fixture/npcausal_ipsi_adapter.R")
options(digits = 17)
set.seed(20261241)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) stop("usage: gates.R SAMPLES.csv.gz DELTAS OUTPUT_ESTIMATES.csv OUTPUT_CURVES.csv.gz")
samples <- read.csv(args[[1]], stringsAsFactors = FALSE, check.names = FALSE)
deltas <- as.numeric(strsplit(args[[2]], ",", fixed = TRUE)[[1]])
ipsi_copy <- ipsi_with_folds()

estimates <- list()
curves <- list()
for (key in unique(paste(samples$scenario, samples$replicate, sep = "\r"))) {
  frame <- samples[paste(samples$scenario, samples$replicate, sep = "\r") == key, ]
  frame <- frame[order(frame$row), ]
  arms <- grep("^A[0-9]+$", names(frame), value = TRUE)
  n_times <- sum(vapply(arms, function(name) !all(is.na(frame[[name]])), logical(1)))
  fit <- ipsi_fit_grid(frame, deltas, frame$fold + 1L, n_times, ipsi_copy)
  estimates[[key]] <- data.frame(
    scenario = frame$scenario[[1]], replicate = frame$replicate[[1]], delta = deltas,
    estimate = fit$estimate, sigma = fit$sigma, calpha = fit$calpha, n = nrow(frame)
  )
  curves[[key]] <- data.frame(
    scenario = frame$scenario[[1]], replicate = frame$replicate[[1]],
    row = rep(frame$row, length(deltas)), delta = rep(deltas, each = nrow(frame)),
    ic = as.numeric(fit$ic)
  )
}
write.csv(do.call(rbind, estimates), args[[3]], row.names = FALSE)
connection <- gzfile(args[[4]], "w")
write.csv(do.call(rbind, curves), connection, row.names = FALSE)
close(connection)
