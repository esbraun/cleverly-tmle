suppressPackageStartupMessages(library(drtmle))
source("/fixture/study_harness.R")
source("/fixture/multi_arm_helpers.R")
options(digits = 17)

args <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(args$samples), stringsAsFactors = FALSE, check.names = FALSE)
truths <- read.csv(args$truths, stringsAsFactors = FALSE, check.names = FALSE)

# The labels in arm-code order. multi_arm_rows_from_moments takes the first as the
# reference: "0" at two arms, and "high" (arm code 0) at three.
three_arm <- "three_arm_mar_outcome_and_treatment"
binary_names <- c(
  "ey[0]" = "ey0", "ey[1]" = "ey1", "ate[1 vs 0]" = "ate", "rr[1 vs 0]" = "rr",
  "or[1 vs 0]" = "or"
)
# The composite TMLE's contrast, read from out$tmle, and the name the study publishes it under.
tmle_rows <- list(
  binary_mar_outcome_and_treatment = c(source = "ate[1 vs 0]", name = "tmle_ate"),
  three_arm_mar_outcome_and_treatment = c(source = "ate[mid vs high]", name = "tmle_ate_mid")
)

fit_one <- function(frame) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  rows <- truths[truths$scenario == scenario & truths$replicate == replicate, ]
  if (nrow(rows) == 0) stop(sprintf("no truth for %s replicate %s", scenario, replicate))
  truth <- rows$truth
  names(truth) <- rows$estimand
  arms <- if (scenario == three_arm) 3L else 2L
  labels <- if (arms == 3L) c("high", "low", "mid") else c("0", "1")
  internal <- truth
  if (arms == 2L) {
    inverse <- stats::setNames(names(binary_names), binary_names)
    known <- names(internal) %in% binary_names
    names(internal)[known] <- inverse[names(internal)[known]]
  }
  codes <- seq_len(arms) - 1L
  qn <- lapply(codes, function(code) frame[[paste0("qn", code)]])
  gn <- lapply(codes, function(code) frame[[paste0("gn", code)]])
  # A is NA where the treatment is unrecorded, so drtmle's default DeltaA = !is.na(A)
  # holds the treatment observation indicator. gn is the oracle composite
  # P(A = a, DeltaA = 1, DeltaY = 1 | W), so estimateG is skipped. cvFolds = 1 fits in-sample.
  fit <- drtmle::drtmle(
    Y = frame$Y,
    A = frame$A_code,
    W = frame["W"],
    DeltaY = frame$Delta,
    a_0 = codes,
    family = binomial(),
    Qn = qn,
    gn = gn,
    glm_Qr = "gn",
    glm_gr = "Qn",
    guard = c("Q", "g"),
    reduction = "univariate",
    maxIter = 100,
    tolIC = 1e-8,
    tolg = 0.01,
    Qsteps = 2,
    cvFolds = 1,
    se_cv = "none",
    returnModels = FALSE,
    returnNuisance = TRUE,
    use_future = FALSE
  )
  initial <- vapply(qn, mean, numeric(1))
  out <- multi_arm_rows_from_moments(
    as.numeric(fit$drtmle$est), fit$drtmle$cov, initial, labels, internal,
    "drtmle-r-composite", scenario, replicate, nrow(frame)
  )
  if (scenario %in% names(tmle_rows)) {
    pairing <- tmle_rows[[scenario]]
    tmle <- multi_arm_rows_from_moments(
      as.numeric(fit$tmle$est), fit$tmle$cov, initial, labels, internal,
      "drtmle-r-composite", scenario, replicate, nrow(frame)
    )
    row <- tmle[tmle$estimand == pairing[["source"]], ]
    if (nrow(row) != 1) stop(sprintf("no out$tmle row %s", pairing[["source"]]))
    row$estimand <- pairing[["name"]]
    row$truth <- unname(truth[[pairing[["name"]]]])
    out <- rbind(out, row)
  }
  if (arms == 2L) {
    known <- out$estimand %in% names(binary_names)
    out$estimand[known] <- binary_names[out$estimand[known]]
  }
  out
}

groups <- split(samples, interaction(samples$scenario, samples$replicate, drop = TRUE))
cores <- study_cores(groups)
results <- parallel::mclapply(
  seq_along(groups),
  study_fitter(groups, fit_one),
  mc.cores = cores,
  mc.preschedule = TRUE
)
# Three scenarios share replicate indices, so a replication is a (scenario, replicate) pair.
study_collect(
  results,
  expected = length(groups),
  output = args$output,
  key = c("scenario", "replicate"),
  versions = c(
    study_version("drtmle"),
    paste("commit", Sys.getenv("DRTMLE_COMMIT"))
  )
)
