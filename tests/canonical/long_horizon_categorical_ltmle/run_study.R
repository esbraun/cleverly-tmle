# Comparator runner of the long-horizon categorical LTMLE study (X59, S2): R lmtp_tmle at
# lmtp 9996b04 (1.5.5).
#
# The fits are long_horizon_lmtp_adapter.R's, with the plan-stratified outcome learner of
# lmtp_horizon_learners.R.  The plans and estimands are read from the truth table.

suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/study_harness.R")
source("/fixture/lmtp_sdr_adapter.R")
source("/fixture/lmtp_9996b04_crossfit_adapter.R")
source("/fixture/lmtp_9996b04_policy_adapter.R")
source("/fixture/lmtp_horizon_learners.R")
source("/fixture/long_horizon_lmtp_adapter.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
if (as.character(packageVersion("lmtp")) != "1.5.5") stop("this runner needs lmtp 1.5.5 at 9996b04")

fit_one <- function(frame) horizon_lmtp_replicate(frame, truths, "lmtp_tmle")

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(results, expected, paths$output, versions = study_version("lmtp"))
