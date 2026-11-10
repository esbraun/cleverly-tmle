# The declared hte3 comparator of the ep-cate study (P1 plan 5.2), PENDING.
#
# Written in a session without R and never run: the owner's local Study queue runs it, and the
# change that adds its output registers the comparison.  The declaration it must keep is
# tests/studies/ep_cate.py:COMPARATOR.  The hte3 and sl3 function and argument names below are
# read from the package's documentation as recorded in the P1 sources, not from a run, and the
# first run must check each one against hte3 at eac64673 before its output is read.
#
# Usage: Rscript hte3_ep_runner.R DRAWS.csv POINTS.csv OUTPUT.csv
#
# DRAWS.csv holds one replicate per (group, replicate): the columns group, replicate, n, k, the
# covariates W1..Wd, A and Y, exported from the property draws of the run (the same streams).
# For each replicate the script fits hte3's Lrnr_cate_EP at hte3 commit eac64673 with
# targeting_style = "dr", no lasso screening, an additive cosine sieve (interaction order 1) with
# basisN = k cosines per modifier on the declared domain [-1, 1], 5 outer folds, and an sl3
# stand-in for the plan's ensemble (Lrnr_gam and Lrnr_glmnet with nnls).  It writes the
# curve at the study's ISE points as OUTPUT.csv with columns group, replicate, point, theta.
# The ISE and the comparison with cleverly's EP are computed in Python, as a different-estimator
# comparison (hte3 debiases by the linear Method 2, cleverly by the logistic Method 1).

suppressPackageStartupMessages({
  library(hte3)
  library(sl3)
  library(data.table)
})
options(digits = 17)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("usage: hte3_ep_runner.R DRAWS.csv POINTS.csv OUTPUT.csv")
draws <- fread(args[[1]])
points <- fread(args[[2]])
out <- list()

nuisance <- Lrnr_sl$new(
  learners = list(Lrnr_gam$new(), Lrnr_glmnet$new()),
  metalearner = Lrnr_nnls$new()
)

for (key in unique(draws[, paste(group, replicate)])) {
  frame <- draws[paste(group, replicate) == key]
  covariates <- grep("^W[0-9]+$", names(frame), value = TRUE)
  k <- frame$k[[1]]
  learner <- Lrnr_cate_EP$new(
    sieve_num_basis = k * length(covariates),
    sieve_interaction_order = 1,
    screen_basis_with_lasso = FALSE,
    targeting_style = "dr"
  )
  task <- make_hte3_Task_tx(
    frame,
    modifiers = covariates,
    confounders = covariates,
    treatment = "A",
    outcome = "Y",
    learner_pi = nuisance,
    learner_mu = nuisance,
    cross_fit_and_cv = TRUE,
    folds = 5
  )
  fitted <- learner$train(task)
  group_points <- points[group == frame$group[[1]]]
  theta <- fitted$predict(sl3_Task$new(group_points, covariates = covariates))
  out[[key]] <- data.table(
    group = frame$group[[1]],
    replicate = frame$replicate[[1]],
    point = seq_len(nrow(group_points)),
    theta = theta
  )
}

fwrite(rbindlist(out), args[[3]])
