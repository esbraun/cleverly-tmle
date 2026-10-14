suppressPackageStartupMessages(library(drtmle))
source("/fixture/study_harness.R")
options(digits = 17)

# R drtmle 1.1.2, outcome-adaptive, by the X17 binary recode per arm, with a missing outcome,
# a missing treatment and baseline strata (X29).  For arm a the treatment is
# A_a = 1{A recorded, A = a}, never NA, with a_0 = c(1, 0) and Qn = list(qa, qa), where qa is
# this package's initial outcome regression at arm a.  DeltaY and DeltaA are always passed:
# both are all ones on a scenario without that missingness, and drtmle then fits no model for
# them.  adapt_g = TRUE makes the design the Qn columns, so with stratify = TRUE
#   gn_DeltaA = P(DeltaA = 1 | qa),
#   gn_A      = P(A_a = 1 | DeltaA = 1, qa),
#   gn_DeltaY = P(DeltaY = 1 | A_a = 1, DeltaA = 1, qa),
# each by glm on the same cubic in logit q as the Python learner, and gn = their product,
# floored at tolg.  That is this package's per-arm factor product.  On the strata scenario the
# same call runs on each stratum's rows.
#
# A is never NA in the recode: drtmle's two-level branch tests length(unique(A)), which counts
# an NA, and an NA sends the call into the multi-level branch.  drtmle returns no TMLE curve,
# so fit_arm rebuilds the targeted regression on its one-dimensional submodel at the reported
# estimate and refuses the fit unless that curve reproduces tmle$cov[1, 1] to 1e-8.  Contrasts
# use the delta method on the stacked curves.  A stratum curve is embedded on every row as
# 1{S = s} / P_n(S = s) times the subset curve, and the marginal is the P_n(S = s) mixture of
# the stratum points, whose curve adds the mass term psi_s - psi.
#
# Each sample row carries A as the arm position 0..K-1 (NA when unrecorded), dA, dY, S (0
# without strata), q<j> the initial regression of arm j and g<j> this package's bounded factor
# product of arm j, which the runner compares with drtmle's gn.

args <- study_arguments(
  "usage: run_drtmle_oat_per_arm_strata_missing.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv"
)
samples <- read.csv(gzfile(args$samples), stringsAsFactors = FALSE, check.names = FALSE)
truths <- read.csv(args$truths, stringsAsFactors = FALSE, check.names = FALSE)
truth_key <- paste(truths$scenario, truths$replicate, truths$estimand, sep = "|")
critical <- qnorm(0.975)
tolg <- 1e-4
logit_q <- "qlogis(pmin(pmax(Q1W, 1e-6), 1 - 1e-6))"
basis <- sprintf("%s + I(%s^2) + I(%s^3)", logit_q, logit_q, logit_q)
implementation <- "drtmle-r-oat-per-arm-strata-missing"
`%||%` <- function(value, default) if (is.null(value)) default else value

fit_arm <- function(frame, a) {
  qa <- frame[[sprintf("q%d", a)]]
  recorded <- frame$dA == 1
  hit_a <- ifelse(is.na(frame$A), 0, as.numeric(recorded & frame$A == a))
  observed_y <- ifelse(frame$dY == 1, frame$Y, NA)
  fit <- study_stage("drtmle", drtmle::drtmle(
    W = frame[c("W1", "W2", "W3")],
    A = hit_a,
    Y = observed_y,
    DeltaY = frame$dY,
    DeltaA = frame$dA,
    a_0 = c(1, 0),
    family = binomial(),
    adapt_g = TRUE,
    stratify = TRUE,
    Qn = list(qa, qa),
    glm_g = list(A = basis, DeltaY = basis, DeltaA = basis),
    cvFolds = 1,
    guard = NULL,
    targeted_se = TRUE,
    tolg = tolg,
    maxIter = 100,
    tolIC = 1e-10,
    returnNuisance = TRUE,
    se_cv = "none",
    use_future = FALSE
  ))
  if (length(fit$tmle$est) != 2) stop("the recode did not return two levels")
  psi <- fit$tmle$est[[1]]
  se <- sqrt(fit$tmle$cov[1, 1])
  gn <- fit$nuisance_drtmle$gnStar[[1]]
  # The composite indicator C_a = DeltaA DeltaY 1{A = a}: the rows the TMLE fluctuates on.
  hit <- hit_a * frame$dY
  outcome <- ifelse(hit == 1, frame$Y, 0)
  offset <- qlogis(qa)
  eps <- uniroot(
    function(value) mean(plogis(offset + value / gn)) - psi, c(-20, 20), tol = 1e-15
  )$root
  targeted <- plogis(offset + eps / gn)
  ic <- hit / gn * (outcome - targeted) + targeted - psi
  rebuilt <- sqrt(stats::var(ic) / nrow(frame))
  if (abs(rebuilt / se - 1) > 1e-8) {
    stop(sprintf("the rebuilt TMLE curve gives se %.12g against the reported %.12g", rebuilt, se))
  }
  mine <- frame[[sprintf("g%d", a)]]
  list(
    psi = psi,
    ic = ic,
    initial = mean(qa),
    gap = max(abs(gn - mine)),
    at_floor = sum(gn <= tolg)
  )
}

row_for <- function(frame, estimand, psi, curve, initial, scale, gap) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  truth_row <- match(paste(scenario, replicate, estimand, sep = "|"), truth_key)
  if (is.na(truth_row)) stop(sprintf("missing truth for %s/%s/%s", scenario, replicate, estimand))
  truth <- truths$truth[[truth_row]]
  se <- sqrt(stats::var(curve) / nrow(frame))
  if (scale == "log") {
    centre <- log(psi)
    low <- exp(centre - critical * se)
    high <- exp(centre + critical * se)
  } else {
    centre <- psi
    low <- psi - critical * se
    high <- psi + critical * se
  }
  data.frame(
    implementation = implementation,
    scenario = scenario,
    replicate = replicate,
    n = nrow(frame),
    estimand = estimand,
    truth = truth,
    estimate = psi,
    inference_estimate = centre,
    std_error = se,
    ci_lower = low,
    ci_upper = high,
    inference_scale = scale,
    covered = as.integer(low <= truth && truth <= high),
    initial_estimate = initial,
    gn_max_abs_diff = gap,
    stringsAsFactors = FALSE
  )
}

# One arm's point, curve and plug-in on every row: on the whole sample, or as the P_n(S = s)
# mixture of the stratum fits.  ``strata`` is NULL without strata.
arm_on_rows <- function(frame, a, strata) {
  if (is.null(strata)) {
    fit <- fit_arm(frame, a)
    return(list(total = fit, parts = list(), gap = fit$gap))
  }
  n <- nrow(frame)
  psi <- 0
  initial <- 0
  centred <- numeric(n)
  parts <- list()
  gap <- 0
  for (s in strata) {
    rows <- frame$S == s
    mass <- mean(rows)
    fit <- fit_arm(frame[rows, , drop = FALSE], a)
    embedded <- numeric(n)
    embedded[rows] <- fit$ic / mass
    parts[[as.character(s)]] <- list(psi = fit$psi, ic = embedded, initial = fit$initial)
    psi <- psi + mass * fit$psi
    initial <- initial + mass * fit$initial
    centred[rows] <- fit$ic + fit$psi
    gap <- max(gap, fit$gap)
  }
  list(total = list(psi = psi, ic = centred - psi, initial = initial), parts = parts, gap = gap)
}

contrast_rows <- function(frame, add, fits, arms, tail, pair) {
  base <- fits[[1]]
  odds <- function(p) p / (1 - p)
  for (a in seq_len(arms - 1)) {
    fit <- fits[[a + 1]]
    add(pair("ate", a, tail), fit$psi - base$psi, fit$ic - base$ic, fit$initial - base$initial,
        "identity")
    if (is.null(tail)) {
      add(pair("rr", a, tail), fit$psi / base$psi, fit$ic / fit$psi - base$ic / base$psi,
          fit$initial / base$initial, "log")
      log_odds_curve <- fit$ic / (fit$psi * (1 - fit$psi)) - base$ic / (base$psi * (1 - base$psi))
      add(pair("or", a, tail), odds(fit$psi) / odds(base$psi), log_odds_curve,
          odds(fit$initial) / odds(base$initial), "log")
    }
  }
}

fit_one <- function(frame) {
  scenario <- frame$scenario[[1]]
  arms <- if (scenario == "three_arm_missing") 3 else 2
  strata <- if (scenario == "strata") sort(unique(frame$S)) else NULL
  results <- lapply(seq_len(arms) - 1, function(a) arm_on_rows(frame, a, strata))
  gap <- max(vapply(results, function(result) result$gap, numeric(1)))
  rows <- list()
  add <- function(estimand, psi, curve, initial, scale) {
    rows[[length(rows) + 1]] <<- row_for(frame, estimand, psi, curve, initial, scale, gap)
  }
  mean_name <- function(a, tail) {
    if (arms == 2) sprintf("ey%d%s", a, tail %||% "") else sprintf("ey[%d]", a)
  }
  pair <- function(stem, a, tail) {
    if (arms == 2) sprintf("%s%s", stem, tail %||% "") else sprintf("%s[%d vs 0]", stem, a)
  }
  totals <- lapply(results, function(result) result$total)
  for (a in seq_len(arms) - 1) {
    total <- totals[[a + 1]]
    add(mean_name(a, NULL), total$psi, total$ic, total$initial, "identity")
  }
  if (is.null(strata)) {
    contrast_rows(frame, add, totals, arms, NULL, pair)
  } else {
    # The strata primary reports ey0, ey1 and ate, marginal and in each stratum.
    contrast_rows(frame, add, totals, arms, "", pair)
    for (s in strata) {
      tail <- sprintf("[S=%d]", as.integer(s))
      parts <- lapply(results, function(result) result$parts[[as.character(s)]])
      for (a in seq_len(arms) - 1) {
        part <- parts[[a + 1]]
        add(mean_name(a, tail), part$psi, part$ic, part$initial, "identity")
      }
      contrast_rows(frame, add, parts, arms, tail, pair)
    }
  }
  do.call(rbind, rows)
}


groups <- split(samples, interaction(samples$scenario, samples$replicate, drop = TRUE))
cores <- study_cores(groups)
# Forced here, in the parent: with `mc.preschedule = TRUE`, `mclapply` would force it in each
# child, and `study_collect` would then find no registered groups to retry or assemble.
fit_group <- study_fitter(groups, fit_one)
results <- parallel::mclapply(
  seq_along(groups),
  fit_group,
  mc.cores = cores,
  mc.preschedule = TRUE
)
# Four scenarios share replicate indices, so a replication is a (scenario, replicate) pair.
study_collect(
  results,
  expected = length(groups),
  output = args$output,
  versions = c(
    study_version("drtmle"),
    paste("commit", Sys.getenv("DRTMLE_COMMIT"))
  ),
  key = c("scenario", "replicate")
)
