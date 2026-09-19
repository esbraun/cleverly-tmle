# Exact-equality probe of the planted-row scale workaround, on every continuous replication.
#
# R tmle takes a continuous outcome's scale from the range of every non-NA Y (tmle.R line
# 1120).  The study runner therefore sets Y to the two q_bounds on two rows whose response
# indicator is zero.  This probe checks five things for each continuous fit the runner makes:
#
#   1. with the planted rows, the scale tmle takes is q_bounds exactly;
#   2. without them, the scale differs, so the workaround is not a no-op;
#   3. moving the planted rows to two other eligible rows leaves the point and every
#      influence-curve value bitwise unchanged;
#   4. an independent rebuild at the q_bounds scale, which fits the fluctuation on the
#      respondents only, reproduces tmle's point and curve; and
#   5. the fit R would run without the workaround reports a different point, so the
#      workaround is not a no-op on the published estimate (unplanted_point_difference > 0).
#      The shift is small for this law, and the targeting-ratio parity check alone would
#      admit it; the witness records its size for a separate check against |cleverly - R|.
#
# The file is deliberately not named run_*.R: tests/unit/test_canonical_runner_parity.py
# holds those to the published-row contract, and this probe publishes a check table.

suppressPackageStartupMessages(library(tmle))
source("/fixture/study_harness.R")
options(digits = 17)

args <- study_arguments("usage: probe_scale_workaround.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- as.data.frame(data.table::fread(cmd = paste("gzip -dc", shQuote(args$samples))))

q_bounds <- c(0, 1)
gbound <- 0.01
alpha <- 0.9995
rebuild_tolerance <- 1e-12
scenario <- "continuous_mar_natural_course"

bound <- function(x, bounds) {
  x[x > max(bounds)] <- max(bounds)
  x[x < min(bounds)] <- min(bounds)
  x
}

plant_at <- function(y, rows) {
  y[rows[[1]]] <- q_bounds[[1]]
  y[rows[[2]]] <- q_bounds[[2]]
  y
}

scale_of <- function(y, a, q, delta) {
  as.numeric(tmle:::.initStage1(y, a, q, NULL, delta, q_bounds, alpha, TRUE, "gaussian")$ab)
}

fit_at <- function(y, a, w, delta, q, g1, p) {
  fit <- tmle::tmle(
    Y = y, A = a, W = w, Delta = delta, Q = q, g1W = g1, pDelta1 = p,
    family = "gaussian", fluctuation = "logistic", Qbounds = q_bounds, gbound = gbound,
    alpha = alpha, cvQinit = FALSE, prescreenW.g = FALSE, target.gwt = FALSE, B = 1,
    evalATT = FALSE, verbose = FALSE
  )
  if (fit$Qinit$type != "user-supplied values" ||
      fit$g$type != "user-supplied values" ||
      fit$g.Delta$type != "user-supplied values") {
    stop("tmle refitted a nuisance instead of using the supplied predictions")
  }
  if (is.null(fit$estimates$EY1) || !is.null(fit$estimates$EY0) ||
      !is.null(fit$estimates$ATE)) {
    stop("tmle did not select its scalar population-mean result shape")
  }
  fit
}

# The published quantities of one population-mean fit: the point, then the curve.
published <- function(fit) {
  c(fit$estimates$EY1$psi, fit$estimates$IC$IC.EY1)
}

# The population-mean fluctuation written out at the q_bounds scale, in tmle's order of
# operations.  The synthetic treatment is constant, so h0 is zero and its coefficient is NA.
rebuild <- function(y, a, delta, q, g1, p) {
  ab <- q_bounds
  n <- length(y)
  ystar <- bound(y, q_bounds)
  ystar[is.na(ystar)] <- 0
  ystar <- (ystar - ab[[1]]) / diff(ab)
  initial <- cbind(QAW = (1 - a) * q[, 1] + a * q[, 2], Q0W = q[, 1], Q1W = q[, 2])
  initial <- (bound(initial, q_bounds) - ab[[1]]) / diff(ab)
  logits <- qlogis(bound(initial, c(alpha, 1 - alpha)))
  g1t <- bound(g1 * p, c(gbound, 1))
  g0t <- bound((1 - g1) * p, c(gbound, 1))
  h1 <- a / g1t
  h0 <- (1 - a) / g0t
  keep <- delta == 1
  epsilon <- suppressWarnings(coef(glm(
    ystar ~ -1 + offset(logits[, "QAW"]) + h0 + h1,
    family = "binomial", weights = rep(1, n), subset = keep
  )))
  epsilon[is.na(epsilon)] <- 0
  targeted <- logits + cbind(epsilon[[1]] * h0 + epsilon[[2]] * h1, epsilon[[1]] / g0t, epsilon[[2]] / g1t)
  targeted <- plogis(targeted) * diff(ab) + ab[[1]]
  outcome <- ystar * diff(ab) + ab[[1]]
  mu1 <- mean(targeted[, "Q1W"])
  residual <- delta * (outcome - targeted[, "QAW"])
  c(mu1, residual / g1t + targeted[, "Q1W"] - mu1)
}

probe_one <- function(frame) {
  if (frame$scenario[[1]] != scenario) stop("the probe received a non-continuous replication")
  replicate <- frame$replicate[[1]]
  n <- nrow(frame)
  y <- frame$Y
  delta <- frame$Delta
  a <- rep(1, n)
  w <- data.frame(A_original = frame$A, W = frame$W)
  q <- cbind(frame$qn, frame$qn)
  g1 <- rep(1, n)
  p <- cbind(frame$pin, frame$pin)
  eligible <- which(delta == 0)
  if (length(eligible) < 4L) stop("the probe needs four rows with a zero response indicator")
  first <- plant_at(y, eligible[1:2])
  last <- plant_at(y, rev(eligible)[1:2])
  planted <- scale_of(first, a, q, delta)
  unplanted <- scale_of(y, a, q, delta)
  at_first <- fit_at(first, a, w, delta, q, g1, p)
  at_last <- fit_at(last, a, w, delta, q, g1, p)
  at_unplanted <- fit_at(y, a, w, delta, q, g1, p)
  reference <- published(at_first)
  moved <- max(abs(published(at_last) - reference))
  rebuilt <- max(abs(rebuild(first, a, delta, q, g1, frame$pin) - reference))
  unplanted_point <- abs(
    as.numeric(at_unplanted$estimates$EY1$psi) - as.numeric(at_first$estimates$EY1$psi)
  )
  scale_exact <- identical(planted, as.numeric(q_bounds))
  unplanted_differs <- !identical(unplanted, as.numeric(q_bounds))
  data.frame(
    scenario = scenario,
    replicate = replicate,
    planted_scale_lower = planted[[1]],
    planted_scale_upper = planted[[2]],
    unplanted_scale_lower = unplanted[[1]],
    unplanted_scale_upper = unplanted[[2]],
    scale_exact = scale_exact,
    unplanted_scale_differs = unplanted_differs,
    moved_rows_difference = moved,
    rebuild_difference = rebuilt,
    rebuild_tolerance = rebuild_tolerance,
    unplanted_point_difference = unplanted_point,
    passed = scale_exact && unplanted_differs && moved == 0 &&
      rebuilt <= rebuild_tolerance && unplanted_point > 0,
    stringsAsFactors = FALSE
  )
}

continuous <- samples[samples$scenario == scenario, ]
rm(samples)
if (nrow(continuous) == 0L) stop(sprintf("the probe replications of %s are missing", scenario))
groups <- split(continuous, continuous$replicate)
rm(continuous)
cores <- study_cores(groups)
results <- parallel::mclapply(
  seq_along(groups),
  study_fitter(groups, probe_one),
  mc.cores = cores,
  mc.preschedule = TRUE
)
study_collect(
  results,
  expected = length(groups),
  output = args$output,
  key = c("scenario", "replicate"),
  versions = c(study_version("tmle"))
)
