suppressPackageStartupMessages({
  library(ctmle)
  library(glmnet)
})
options(digits = 17)

# The S1 cross-check of the X16 plan (3.4) on one witness search seed: `rebuild_type1`, the
# type-1 adapter of the registered study, must agree exactly with `ctmleGlmnet` wherever
# `ctmleGlmnet` completes.  `regenerate.py` runs this script on each seed where `witness.R`
# completed and records the result in `manifest.json`.  The call repeats section 5 of
# `witness.R`, with the same arguments.  The folds are given, so neither call draws from R's
# generator (`functions_glmnet.R`, lines 308 to 309), and the script sets no seed.
glmnet.control(fdev = 0, devmax = 1)
source("/shared/rebuild_type1.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: rebuild_check.R INPUT_DIR OUTPUT_DIR")
input_dir <- args[[1]]
output_dir <- args[[2]]

frame <- read.csv(file.path(input_dir, "inputs.csv"), stringsAsFactors = FALSE)
grid <- read.csv(file.path(input_dir, "grid.csv"))$lambda
names_w <- grep("^W[0-9]+$", names(frame), value = TRUE)
x <- as.matrix(frame[names_w])
folds <- lapply(sort(unique(frame$fold)), function(f) which(frame$fold == f))
arguments <- list(
  Y = frame$Y, A = frame$A, W = data.frame(x), Q = cbind(frame$Q0W, frame$Q1W),
  lambdas = grid, ctmletype = 1, folds = folds, V = length(folds), alpha = 0.995,
  family = "binomial", gbound = 0.025, like_type = "loglik", fluctuation = "logistic",
  PEN = FALSE, detailed = TRUE
)

lasso <- do.call(ctmleGlmnet, arguments)
rebuilt <- do.call(rebuild_type1, arguments)

write.csv(
  data.frame(
    best_k = as.integer(lasso$best_k),
    rebuild_best_k = as.integer(rebuilt$cv.res$best_k),
    est_identical = as.integer(identical(
      rebuilt$candidates$results.all$est, lasso$candidates$results.all$est
    )),
    lambdas_identical = as.integer(identical(
      rebuilt$candidates$candidates$lambdas, lasso$candidates$candidates$lambdas
    ))
  ),
  file.path(output_dir, "rebuild.csv"),
  row.names = FALSE
)
