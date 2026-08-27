suppressPackageStartupMessages({
  library(sl3)
  library(tmle3)
})
source("/fixture/study_harness.R")
source("/fixture/multi_arm_helpers.R")
options(digits = 17)

paths <- study_arguments(
  "usage: run_multi_arm_tmle3.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv"
)
samples <- read.csv(gzfile(paths$samples), stringsAsFactors = FALSE, check.names = FALSE)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
truth_key <- paste(truths$scenario, truths$replicate, sep = "|")
truth_columns <- grep("^truth_", names(truths), value = TRUE)
labels <- c("high", "low", "medium")

# This sl3 snapshot has no categorical logistic learner: Lrnr_glm asks R for a
# nonexistent multinomial() family and Lrnr_mean would erase the confounding that makes
# targeting load-bearing.  The base-recommended nnet package supplies the same softmax
# regression as sklearn; the adapter only packs its probability matrix into sl3's old
# categorical prediction representation.
Lrnr_multinom_fixed <- R6::R6Class(
  "Lrnr_multinom_fixed",
  inherit = sl3::Lrnr_base,
  public = list(
    initialize = function() {
      super$initialize(params = list(), name = "multinomial logistic regression")
    }
  ),
  private = list(
    .properties = c("categorical", "weights"),
    .required_packages = "nnet",
    .train = function(task) {
      outcome_type <- self$get_outcome_type(task)
      data <- as.data.frame(task$X)
      data$.outcome <- factor(task$Y, levels = outcome_type$levels)
      list(
        model = nnet::multinom(
          .outcome ~ ., data = data, weights = task$weights, trace = FALSE
        ),
        levels = outcome_type$levels
      )
    },
    .predict = function(task) {
      probabilities <- predict(
        self$fit_object$model, newdata = as.data.frame(task$X), type = "probs"
      )
      probabilities <- as.matrix(probabilities)
      stopifnot(ncol(probabilities) == length(self$fit_object$levels))
      colnames(probabilities) <- self$fit_object$levels
      sl3::pack_predictions(probabilities)
    }
  )
)

learners <- list(A = Lrnr_multinom_fixed$new(), Y = sl3::Lrnr_mean$new())

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
    tmle_TSM_all(), data, list(W = c("W1", "W2", "W3"), A = "A", Y = "Y"), learners
  )
  multi_arm_rows_from_tmle(
    fit, labels, truth, "tmle3-multi-arm", scenario, replicate, nrow(frame)
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
  versions = c(study_version("tmle3"), study_version("sl3")),
  key = c("scenario", "replicate"), na = "NA"
)
