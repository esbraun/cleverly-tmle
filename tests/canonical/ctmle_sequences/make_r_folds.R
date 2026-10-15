# R's selection folds of each binary_learners replicate of `canonical-ctmle-sequences`.
#
# usage: make_r_folds.R SEEDS.csv N OUTPUT.csv.gz
#
# SEEDS.csv is the seed table the regeneration driver hands the runner (scenario, replicate,
# seed).  For each binary_learners replicate r this script sets that replicate's seed and draws
# the folds exactly as cv_general does (functions_general.R:287) and as the runner does first
# in fit_one:
#
#   set.seed(s_r); by(sample(1:n, n), rep(1:5, length = n), list)
#
# The output holds one row per sample row: replicate, row (0-based, in sample order) and fold
# (1 to 5).  The Python side scores its candidates on these folds, and the runner stops on any
# replicate whose redrawn folds differ from them.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("usage: make_r_folds.R SEEDS.csv N OUTPUT.csv.gz")
seeds <- read.csv(
  args[[1]], colClasses = c("character", "integer", "integer"), stringsAsFactors = FALSE
)
if (!identical(names(seeds), c("scenario", "replicate", "seed"))) {
  stop("the seed table must hold the columns scenario, replicate, seed")
}
n <- as.integer(args[[2]])
if (is.na(n) || n < 5L) stop("N must be an integer of at least 5")
V <- 5L

seeds <- seeds[seeds$scenario == "binary_learners", ]
seeds <- seeds[order(seeds$replicate), ]
if (nrow(seeds) == 0L) stop("the seed table has no binary_learners rows")

draw <- function(replicate, seed) {
  set.seed(seed)
  folds <- by(sample(1:n, n), rep(1:V, length = n), list)
  labels <- integer(n)
  for (k in seq_along(folds)) {
    labels[as.integer(unlist(folds[[k]], use.names = FALSE))] <- k
  }
  if (any(labels == 0L)) stop(sprintf("replicate %d: a row has no fold", replicate))
  data.frame(replicate = replicate, row = seq_len(n) - 1L, fold = labels)
}

table <- do.call(rbind, Map(draw, seeds$replicate, seeds$seed))
connection <- gzfile(args[[3]], "w")
write.csv(table, connection, row.names = FALSE, quote = FALSE)
close(connection)
cat(sprintf("wrote %d replicates of %d rows to %s\n", nrow(seeds), n, args[[3]]))
