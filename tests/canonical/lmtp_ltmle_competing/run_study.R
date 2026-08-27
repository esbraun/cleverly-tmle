suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_crossfit_adapter.R")
source("/fixture/lmtp_competing_adapter.R")
source("/fixture/study_harness.R")
source("/fixture/ltmle_regimen_adapter.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)

scenario <- "censored_competing_risk_curve"
causes <- c("relapse", "death")
z <- qnorm(0.975)


# The law's own mechanism, node by node, given to lmtp because it has no `gform` argument to be
# handed one through.  Cleverly receives the same numbers via KnownCompetingMechanism, so the
# paired comparison measures the competing-risk recursion, the targeting and the influence curve
# rather than two mechanism-fitting pipelines.
#
# Every conditional is a lookup over history cells rather than a closed form, because this law
# is a table.  Written as `array()` plus an index matrix rather than as nested `ifelse` chains:
# the arrays below are transcriptions of `G1`, `C1`, `G2` and `C2` in
# `tests/discrete_law_competing.py` and can be read against them cell by cell, whereas a chain
# that enumerates seven of eight cells and falls through to a bare literal for the eighth hides
# the one that was left out.  `lmtp_competing_adapter.R` re-checks the result against lmtp's own
# fitted ratios on every run.
#
# `array()` fills in column-major order, so the first index varies fastest.  Each array is
# indexed [w + 1, a1 + 1, ...] to match the Python arrays' [w, a1, ...].
G1 <- array(c(0.50, 0.25), dim = c(2))                       # P(A1 = 1 | W)
C1 <- array(c(0.75, 0.75, 0.50, 0.75), dim = c(2, 2))        # P(C1 = 1 | W, A1)
G2 <- array(                                                 # P(A2 = 1 | W, A1, L2)
  c(0.50, 0.75, 0.25, 0.50, 0.75, 0.50, 0.50, 0.25),
  dim = c(2, 2, 2)
)
C2 <- array(                                                 # P(C2 = 1 | W, A1, L2, A2)
  c(
    0.75, 0.75, 0.50, 0.75,
    0.75, 0.50, 0.75, 0.75,
    0.50, 0.75, 0.75, 0.50,
    0.75, 0.75, 0.50, 0.75
  ),
  dim = c(2, 2, 2, 2)
)

exact_ratios <- function(frame, arms, horizon) {
  # Per node, not cumulative: column t is zero exactly where the unit left the path at that
  # node, and lmtp multiplies the columns downstream.  The adapter re-checks both halves.
  w <- as.integer(frame$W)
  a1 <- as.integer(arms[, 1])
  p_a1 <- G1[w + 1]
  g1 <- ifelse(a1 == 1, p_a1, 1 - p_a1)
  c1 <- C1[cbind(w + 1, a1 + 1)]
  followed1 <- frame$A1 == a1 & frame$C1 == 1
  first <- ifelse(followed1, 1 / (g1 * c1), 0)
  if (horizon == 1) return(matrix(first, ncol = 1))

  l2 <- as.integer(ifelse(is.na(frame$L2), 0, frame$L2))
  a2 <- as.integer(arms[, 2])
  p_a2 <- G2[cbind(w + 1, a1 + 1, l2 + 1)]
  g2 <- ifelse(a2 == 1, p_a2, 1 - p_a2)
  c2 <- C2[cbind(w + 1, a1 + 1, l2 + 1, a2 + 1)]
  # A unit that had an event of either cause at the first node has no second-node arm.  ``A2``
  # is ``NA`` there, which is what removes it from the second column rather than a separate mask.
  followed2 <- !is.na(frame$A2) & frame$A2 == a2 & !is.na(frame$C2) & frame$C2 == 1
  cbind(first, ifelse(followed2, 1 / (g2 * c2), 0))
}

fit_plan <- function(frame, label, cause, horizon) {
  target <- if (cause == "relapse") c("R1", "R2") else c("D1", "D2")
  other <- if (cause == "relapse") c("D1", "D2") else c("R1", "R2")
  if (horizon == 1) {
    natural <- frame[c("W", "A1", "C1", target[[1]])]
    trt <- "A1"
    outcome <- target[[1]]
    compete <- NULL
    time_vary <- list(NULL)
    cens <- "C1"
  } else {
    natural <- frame[c(
      "W", "A1", "C1", target[[1]], other[[1]], "L2", "A2", "C2",
      target[[2]], other[[2]]
    )]
    trt <- c("A1", "A2")
    outcome <- target
    compete <- other
    time_vary <- list(NULL, "L2")
    cens <- c("C1", "C2")
  }
  shifted <- natural
  # This panel's dynamic plan reads L2 == 1 rather than L2 > 0: its second covariate is
  # three-valued, so "positive" and "equal to one" are different plans here.
  arms <- regimen_arms(
    frame, label, horizon,
    rule = function(l2) as.numeric(!is.na(l2) & l2 == 1)
  )
  shifted$A1 <- arms[, 1]
  if (horizon == 2) shifted$A2 <- arms[, 2]
  fit <- lmtp_competing_tmle_with_folds(
    natural,
    shifted,
    trt = trt,
    outcome = outcome,
    compete = compete,
    baseline = "W",
    time_vary = time_vary,
    cens = cens,
    outcome_type = if (horizon == 1) "binomial" else "survival",
    fold_assignment = frame$fold,
    learners_outcome = "SL.mean",
    learners_trt = "SL.glm",
    density_ratios = exact_ratios(frame, arms, horizon),
    control = lmtp_control(
      .trim = 1,
      .learners_outcome_folds = 2,
      .learners_trt_folds = 2,
      .return_full_fits = TRUE
    )
  )
  if (!identical(fit$fold_assignment, as.integer(frame$fold))) {
    stop("lmtp did not retain the supplied fold assignment")
  }
  if (horizon == 1) {
    list(estimate = fit$estimate@x, initial = mean(fit$initial), ic = fit$estimate@eif)
  } else {
    list(
      estimate = 1 - fit$estimate@x,
      initial = 1 - mean(fit$initial),
      ic = -fit$estimate@eif
    )
  }
}

truth_for <- function(replicate, estimand) {
  selected <- truths$replicate == replicate & truths$scenario == scenario & truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows", sum(selected)))
  truths$truth[selected]
}

row_for <- function(replicate, name, estimate, initial, ic, n) {
  truth <- truth_for(replicate, name)
  standard_error <- sd(ic) / sqrt(n)
  low <- estimate - z * standard_error
  high <- estimate + z * standard_error
  data.frame(
    implementation = "lmtp",
    scenario = scenario,
    replicate = replicate,
    n = n,
    estimand = name,
    truth = truth,
    estimate = estimate,
    inference_estimate = estimate,
    std_error = standard_error,
    ci_lower = low,
    ci_upper = high,
    inference_scale = "identity",
    covered = as.integer(low <= truth && truth <= high),
    initial_estimate = initial,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  rows <- list()
  for (cause in causes) {
    fits <- list(
      "never @ t=1" = fit_plan(frame, "never", cause, 1),
      "never @ t=2" = fit_plan(frame, "never", cause, 2),
      "always @ t=1" = fit_plan(frame, "always", cause, 1),
      "always @ t=2" = fit_plan(frame, "always", cause, 2),
      "continue_if_l2 @ t=2" = fit_plan(frame, "continue_if_l2", cause, 2)
    )
    for (key in names(fits)) {
      fit <- fits[[key]]
      pieces <- strsplit(key, " @ t=", fixed = TRUE)[[1]]
      rows[[length(rows) + 1]] <- row_for(
        replicate,
        sprintf("cif_regimen[%s, %s @ t=%s]", pieces[[1]], cause, pieces[[2]]),
        fit$estimate,
        fit$initial,
        fit$ic,
        nrow(frame)
      )
    }
    comparisons <- list(
      c("always", "1", "always @ t=1", "never @ t=1"),
      c("always", "2", "always @ t=2", "never @ t=2"),
      c("continue_if_l2", "2", "continue_if_l2 @ t=2", "never @ t=2")
    )
    for (spec in comparisons) {
      left <- fits[[spec[[3]]]]
      right <- fits[[spec[[4]]]]
      rows[[length(rows) + 1]] <- row_for(
        replicate,
        sprintf(
          "ate_regimen[%s vs never, %s @ t=%s]", spec[[1]], cause, spec[[2]]
        ),
        left$estimate - right$estimate,
        left$initial - right$initial,
        left$ic - right$ic,
        nrow(frame)
      )
    }
  }
  do.call(rbind, rows)
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected, chunk_lines = 512000L)
study_collect(results, expected, paths$output, versions = study_version("lmtp"))
