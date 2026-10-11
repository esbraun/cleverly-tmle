suppressPackageStartupMessages(library(drtmle))
source("/fixture/study_harness.R")
options(digits = 17)

# R drtmle 1.1.2, outcome-adaptive, by a binary recode per arm.  For arm a the treatment is
# A_a = 1{A = a} with a_0 = c(1, 0) and Qn = list(qa, qa), where qa is this package's initial
# outcome regression at arm a.  That enters the two-level glm branch of estimate_g, whose design
# names the columns Q1W and Q0W, so level 1 is exactly P(A = a | Qbar(a, W)).  glm_g is the same
# cubic in logit q as the Python learner.  adapt_g = TRUE turns the extra targeting off.  The
# contrasts are formed from the per-arm TMLE curves, which fit_arm rebuilds and checks.
# tolg = 0.01 truncates below only; the laws keep the projection inside (0.01, 0.99).
#
# Each sample row carries A as the arm position 0..K-1, q<j> the initial regression of arm j and
# g<j> this package's untruncated per-arm mechanism, which the runner compares with its own.

args <- study_arguments("usage: run_drtmle_oat_per_arm.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(args$samples), stringsAsFactors = FALSE)
truths <- read.csv(args$truths, stringsAsFactors = FALSE)
truth_key <- paste(truths$scenario, truths$replicate, truths$estimand, sep = "|")
critical <- qnorm(0.975)
tolg <- 0.01
lower <- 0.01
upper <- 0.99
logit_q <- "qlogis(pmin(pmax(Q1W, 1e-6), 1 - 1e-6))"
basis <- sprintf("%s + I(%s^2) + I(%s^3)", logit_q, logit_q, logit_q)

fit_arm <- function(frame, a) {
  qa <- frame[[sprintf("q%d", a)]]
  fit <- drtmle::drtmle(
    W = frame[c("W1", "W2", "W3")],
    A = as.numeric(frame$A == a),
    Y = frame$Y,
    a_0 = c(1, 0),
    family = binomial(),
    adapt_g = TRUE,
    Qn = list(qa, qa),
    glm_g = basis,
    cvFolds = 1,
    guard = NULL,
    targeted_se = TRUE,
    tolg = tolg,
    maxIter = 100,
    tolIC = 1e-10,
    returnNuisance = TRUE,
    se_cv = "none",
    use_future = FALSE
  )
  if (length(fit$tmle$est) != 2) stop("the recode did not return two levels")
  psi <- fit$tmle$est[[1]]
  se <- sqrt(fit$tmle$cov[1, 1])
  gn <- fit$nuisance_drtmle$gnStar[[1]]
  # drtmle returns no TMLE curve: ic_drtmle is the curve of its own drtmle stage, whose
  # outcome regression is not the TMLE's.  The TMLE's targeted regression is the one point of
  # its one-dimensional logistic submodel, logit Q* = logit qa + eps / gn, whose mean is the
  # reported estimate, so the runner rebuilds it there and refuses the fit unless the rebuilt
  # curve reproduces the reported variance.
  hit <- as.numeric(frame$A == a)
  offset <- qlogis(qa)
  eps <- uniroot(
    function(value) mean(plogis(offset + value / gn)) - psi, c(-20, 20), tol = 1e-15
  )$root
  targeted <- plogis(offset + eps / gn)
  ic <- hit / gn * (frame$Y - targeted) + targeted - psi
  rebuilt <- sqrt(stats::var(ic) / nrow(frame))
  if (abs(rebuilt / se - 1) > 1e-8) {
    stop(sprintf("the rebuilt TMLE curve gives se %.12g against the reported %.12g", rebuilt, se))
  }
  mine <- pmin(pmax(frame[[sprintf("g%d", a)]], lower), upper)
  list(
    psi = psi,
    se = se,
    ic = ic,
    initial = mean(qa),
    gap = max(abs(gn - mine))
  )
}

row_for <- function(frame, estimand, psi, se, initial, scale, gap) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  truth_row <- match(paste(scenario, replicate, estimand, sep = "|"), truth_key)
  if (is.na(truth_row)) stop(sprintf("missing truth for %s/%s/%s", scenario, replicate, estimand))
  truth <- truths$truth[[truth_row]]
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
    implementation = "drtmle-r-oat-per-arm",
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

fit_one <- function(frame) {
  scenario <- frame$scenario[[1]]
  arms <- if (scenario == "three_arm_active") 3 else 2
  n <- nrow(frame)
  fits <- lapply(seq_len(arms) - 1, function(a) fit_arm(frame, a))
  gap <- max(vapply(fits, function(fit) fit$gap, numeric(1)))
  se_of <- function(curve) sqrt(stats::var(curve) / n)
  odds <- function(p) p / (1 - p)
  rows <- list()
  add <- function(estimand, psi, se, initial, scale) {
    rows[[length(rows) + 1]] <<- row_for(frame, estimand, psi, se, initial, scale, gap)
  }
  mean_name <- function(a) if (arms == 2) sprintf("ey%d", a) else sprintf("ey[%d]", a)
  pair <- function(stem, a) if (arms == 2) stem else sprintf("%s[%d vs 0]", stem, a)
  for (a in seq_len(arms) - 1) {
    fit <- fits[[a + 1]]
    add(mean_name(a), fit$psi, fit$se, fit$initial, "identity")
  }
  base <- fits[[1]]
  for (a in seq_len(arms - 1)) {
    fit <- fits[[a + 1]]
    add(pair("ate", a), fit$psi - base$psi, se_of(fit$ic - base$ic), fit$initial - base$initial,
        "identity")
    add(pair("rr", a), fit$psi / base$psi, se_of(fit$ic / fit$psi - base$ic / base$psi),
        fit$initial / base$initial, "log")
    log_odds_curve <- fit$ic / (fit$psi * (1 - fit$psi)) - base$ic / (base$psi * (1 - base$psi))
    add(pair("or", a), odds(fit$psi) / odds(base$psi), se_of(log_odds_curve),
        odds(fit$initial) / odds(base$initial), "log")
  }
  if (arms == 2) {
    # The population-intervention contrasts read the observed mean, whose curve is Y - mean(Y)
    # on complete data.
    observed <- mean(frame$Y)
    observed_curve <- frame$Y - observed
    add("par", observed - base$psi, se_of(observed_curve - base$ic), observed - base$initial,
        "identity")
    paf_curve <- base$ic / observed - base$psi * observed_curve / observed^2
    add("paf", 1 - base$psi / observed, se_of(-paf_curve), 1 - base$initial / observed,
        "identity")
  }
  do.call(rbind, rows)
}

groups <- split(samples, interaction(samples$scenario, samples$replicate, drop = TRUE))
cores <- study_cores(groups)
results <- parallel::mclapply(
  seq_along(groups),
  study_fitter(groups, fit_one),
  mc.cores = cores,
  mc.preschedule = TRUE
)
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
