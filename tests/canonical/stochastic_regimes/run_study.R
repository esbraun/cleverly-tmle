suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_point_adapter.R")
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)
scenario <- "binary_known_stochastic"
z <- qnorm(0.975)

# The generating mechanism and the declared intervention density, both indexed by W.  Supplied
# rather than fitted for the same reason the deterministic runner supplies its own: the study
# compares two targeting steps, not two mechanism-fitting pipelines.
G1 <- c(0.40, 0.60, 0.25)
TILT1 <- c(0.25, 0.50, 0.75)

truth_for <- function(replicate, estimand) {
  selected <- truths$replicate == replicate & truths$scenario == scenario & truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows", sum(selected)))
  truths$truth[selected]
}

# `lmtp`'s public interface takes a shift function of the natural treatment value, which is why
# the study page used to record that no comparator existed for a regime that ignores the
# natural value entirely.  The point adapter does not go through that interface: it hands
# `cf_density_ratios` a ratio and `cf_tmle` a shifted frame directly, and neither one requires
# the shifted column to be a function of the observed one.  A known stochastic regime is
# therefore expressible, and this is what it looks like.
fit_policy <- function(frame, label) {
  level <- as.integer(frame$W) + 1L
  g1 <- G1[level]
  if (label == "never") {
    assigned <- rep(0, nrow(frame))
    # A degenerate density: the ratio is the reciprocal of the observed arm probability on the
    # followed path and exactly zero off it, which is the deterministic path `mtp = FALSE`
    # checks the supplied ratio against.
    ratio <- ifelse(frame$A == 0, 1 / (1 - g1), 0)
    mtp <- FALSE
  } else {
    # One draw from the declared density per unit.  `lmtp` evaluates the shifted regression at
    # a realised treatment value rather than integrating over one, so the reference estimates
    # `E[m(A*, W)]` with `A*` drawn from q, which is unbiased for the integral and carries the
    # variance of the draw.  That variance is real and is reported: it belongs to the
    # estimator this study is comparing against, not to the parameter.
    assigned <- rbinom(nrow(frame), 1, TILT1[level])
    tilt_observed <- ifelse(frame$A == 1, TILT1[level], 1 - TILT1[level])
    natural_observed <- ifelse(frame$A == 1, g1, 1 - g1)
    ratio <- tilt_observed / natural_observed
    mtp <- TRUE
  }
  shifted <- frame[c("W", "A", "Y")]
  shifted$A <- assigned
  fit <- lmtp_point_tmle(
    frame[c("W", "A", "Y")],
    shifted,
    ratio,
    mtp = mtp,
    outcome_type = "binomial"
  )
  list(estimate = fit$estimate@x, initial = fit$initial, ic = fit$estimate@eif)
}

row_for <- function(replicate, name, fit, n) {
  truth <- truth_for(replicate, name)
  standard_error <- sd(fit$ic) / sqrt(n)
  low <- fit$estimate - z * standard_error
  high <- fit$estimate + z * standard_error
  data.frame(
    implementation = "lmtp",
    scenario = scenario,
    replicate = replicate,
    n = n,
    estimand = name,
    truth = truth,
    estimate = fit$estimate,
    inference_estimate = fit$estimate,
    std_error = standard_error,
    ci_lower = low,
    ci_upper = high,
    inference_scale = "identity",
    covered = as.integer(low <= truth && truth <= high),
    initial_estimate = fit$initial,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  # Seeded before the first draw, so replication k of a short probe is the same fit as
  # replication k of the declared run.  Nothing else in this runner consumes randomness.
  set.seed(20260902L + as.integer(replicate))
  never <- fit_policy(frame, "never")
  tilt <- fit_policy(frame, "tilt")
  contrast <- list(
    estimate = tilt$estimate - never$estimate,
    initial = tilt$initial - never$initial,
    ic = tilt$ic - never$ic
  )
  rbind(
    row_for(replicate, "ey_regime[never]", never, nrow(frame)),
    row_for(replicate, "ey_regime[tilt]", tilt, nrow(frame)),
    row_for(replicate, "ate_regime[tilt vs never]", contrast, nrow(frame))
  )
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(results, expected, paths$output, versions = study_version("lmtp"))
