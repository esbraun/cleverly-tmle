suppressPackageStartupMessages(library(tmle))
source("/fixture/study_harness.R")
source("/fixture/tmle_point_adapter.R")
options(digits = 17)

args <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(args$samples), stringsAsFactors = FALSE)
truths <- read.csv(args$truths, stringsAsFactors = FALSE)

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  fit_level <- function(level) {
    scenario <- if (level == 0L) "binary_cde_z0_mar" else "binary_cde_z1_mar"
    original_z0 <- cbind(frame$qn_z0_a0, frame$qn_z0_a1)
    original_z1 <- cbind(frame$qn_z1_a0, frame$qn_z1_a1)
    # Recode the requested level to zero so Q is both the exact observed offset on targeted
    # rows and the exact counterfactual regression. Q.Z1 remains the exact other level.
    run_z <- if (level == 0L) frame$Z else 1L - frame$Z
    q_z0 <- if (level == 0L) original_z0 else original_z1
    q_z1 <- if (level == 0L) original_z1 else original_z0
    p_z1 <- if (level == 0L) {
      cbind(frame$pzn_a0, frame$pzn_a1)
    } else {
      cbind(1 - frame$pzn_a0, 1 - frame$pzn_a1)
    }
    # tmle orders these as (Z=0,A=0), (Z=0,A=1), (Z=1,A=0), (Z=1,A=1).
    # The declared MAR law excludes Z, so each arm's column repeats at the other level.
    p_delta1 <- cbind(frame$pin_a0, frame$pin_a1, frame$pin_a0, frame$pin_a1)
    fit <- tmle::tmle(
      Y = frame$Y,
      A = frame$A,
      W = frame["W"],
      Z = run_z,
      Delta = frame$Delta,
      Q = q_z0,
      Q.Z1 = q_z1,
      g1W = frame$gn1,
      pZ1 = p_z1,
      pDelta1 = p_delta1,
      family = "binomial",
      fluctuation = "logistic",
      Qbounds = c(0.001, 0.999),
      gbound = c(0.01, 0.99),
      cvQinit = FALSE,
      evalATT = FALSE,
      verbose = FALSE
    )
    if (!inherits(fit, "tmle.list") || length(fit) != 2L) {
      stop("tmle did not return one controlled direct-effect fit per level")
    }
    tmle_point_rows(
      fit[[1L]],
      q_z0,
      rep(1, nrow(frame)),
      truths,
      scenario,
      replicate,
      implementation = "tmle-r-cde"
    )
  }
  do.call(rbind, lapply(0:1, fit_level))
}

groups <- split(samples, samples$replicate)
cores <- study_cores(groups)
results <- parallel::mclapply(
  seq_along(groups),
  study_fitter(groups, fit_one),
  mc.cores = cores,
  mc.preschedule = TRUE
)
study_collect(
  results,
  expected = 2L * length(groups),
  output = args$output,
  versions = c(
    study_version("tmle"),
    paste("source sha256", Sys.getenv("TMLE_SHA256"))
  ),
  key = c("scenario", "replicate")
)
