suppressPackageStartupMessages({
  library(data.table)
  library(sl3)
  library(tmle3)
  library(ctmle3)
})
source("/fixture/study_harness.R")
source("/fixture/multi_arm_helpers.R")
options(digits = 17)

paths <- study_arguments(
  "usage: run_multi_arm_ctmle3_oat.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv"
)
samples <- read.csv(gzfile(paths$samples), stringsAsFactors = FALSE, check.names = FALSE)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
truth_key <- paste(truths$scenario, truths$replicate, sep = "|")
truth_columns <- grep("^truth_", names(truths), value = TRUE)
labels <- c("high", "low", "medium")
learners <- list(A = sl3::Lrnr_mean$new(), Y = sl3::Lrnr_glm$new())

fit_one <- function(frame) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  row <- match(paste(scenario, replicate, sep = "|"), truth_key)
  if (is.na(row)) stop(sprintf("no truth for %s replicate %s", scenario, replicate))
  truth <- as.numeric(truths[row, truth_columns])
  names(truth) <- sub("^truth_", "", truth_columns)
  data <- frame[c("W1", "W2", "W3", "A_code", "Y")]
  names(data)[names(data) == "A_code"] <- "A"
  fit <- tmle3(
    tmle_oat_TSM_all(), data,
    list(W = c("W1", "W2", "W3"), A = "A", Y = "Y"), learners
  )
  multi_arm_rows_from_tmle(
    fit, labels, truth, "ctmle3-multi-arm-oat", scenario, replicate, nrow(frame)
  )
}

groups <- split(samples, interaction(samples$scenario, samples$replicate, drop = TRUE))
expected <- length(groups)
rm(samples)
invisible(gc())
results <- parallel::mclapply(
  seq_along(groups), study_fitter(groups, fit_one),
  mc.cores = study_cores(groups), mc.preschedule = FALSE
)
study_collect(
  results, expected, paths$output,
  versions = c(study_version("ctmle3"), study_version("tmle3"), study_version("sl3")),
  key = c("scenario", "replicate"), na = "NA"
)
