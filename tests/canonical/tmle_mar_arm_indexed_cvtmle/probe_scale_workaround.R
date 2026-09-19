# Exact-equality probe of the planted-row scale workaround, on replication 0.
#
# R tmle takes a continuous outcome's scale from the range of every non-NA Y (tmle.R line
# 1120).  The study runner therefore sets Y to the two q_bounds on two rows whose response
# indicator is zero.  This probe checks four things for each continuous fit the runner makes:
#
#   1. with the planted rows, the scale tmle takes is q_bounds exactly;
#   2. without them, the scale differs, so the workaround is not a no-op;
#   3. moving the planted rows to two other eligible rows leaves every point and every
#      influence-curve value bitwise unchanged; and
#   4. an independent rebuild at the q_bounds scale, which fits the fluctuation on the
#      respondents only, reproduces tmle's fluctuation, points, and curves.
#
# The file is deliberately not named run_*.R: tests/unit/test_canonical_runner_parity.py
# holds those to the published-row contract, and this probe publishes a check table.

suppressPackageStartupMessages(library(tmle))
source("/fixture/study_harness.R")
options(digits = 17)

args <- study_arguments("usage: probe_scale_workaround.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- as.data.frame(data.table::fread(cmd = paste("gzip -dc", shQuote(args$samples))))

q_bounds <- c(-2, 8)
gbound <- 0.001
alpha <- 0.9995
rebuild_tolerance <- 1e-12
probe_replicate <- 0
three_arm_labels <- c("high", "low", "mid")
continuous <- list(
  continuous_mar_two_arm_stacked = 2L,
  continuous_mar_three_arm_stacked = 3L
)

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
  tmle::tmle(
    Y = y, A = a, W = w, Delta = delta, Q = q, g1W = g1, pDelta1 = p,
    family = "gaussian", fluctuation = "logistic", Qbounds = q_bounds, gbound = gbound,
    alpha = alpha, cvQinit = FALSE, prescreenW.g = FALSE, target.gwt = FALSE, B = 1,
    evalATT = FALSE, verbose = FALSE
  )
}

# The published quantities of one fit: every point, then every influence-curve value.
published <- function(fit, two_arm) {
  if (two_arm) {
    c(
      fit$estimates$EY0$psi, fit$estimates$EY1$psi, fit$estimates$ATE$psi,
      fit$estimates$IC$IC.EY0, fit$estimates$IC$IC.EY1, fit$estimates$IC$IC.ATE
    )
  } else {
    c(fit$estimates$EY1$psi, fit$estimates$IC$IC.EY1)
  }
}

# The stacked fluctuation written out at the q_bounds scale, in tmle's order of operations.
rebuild <- function(y, a, delta, q, g1, p0, p1, two_arm) {
  ab <- q_bounds
  n <- length(y)
  ystar <- bound(y, q_bounds)
  ystar[is.na(ystar)] <- 0
  ystar <- (ystar - ab[[1]]) / diff(ab)
  initial <- cbind(QAW = (1 - a) * q[, 1] + a * q[, 2], Q0W = q[, 1], Q1W = q[, 2])
  initial <- (bound(initial, q_bounds) - ab[[1]]) / diff(ab)
  logits <- qlogis(bound(initial, c(alpha, 1 - alpha)))
  g1t <- bound(g1 * p1, c(gbound, 1))
  g0t <- bound((1 - g1) * p0, c(gbound, 1))
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
  if (!two_arm) {
    return(c(mu1, residual / g1t + targeted[, "Q1W"] - mu1))
  }
  mu0 <- mean(targeted[, "Q0W"])
  c(
    mu0, mu1, mu1 - mu0,
    (1 - a) / g0t * residual + targeted[, "Q0W"] - mu0,
    a / g1t * residual + targeted[, "Q1W"] - mu1,
    (a / g1t - (1 - a) / g0t) * residual + targeted[, "Q1W"] - targeted[, "Q0W"] - (mu1 - mu0)
  )
}

probe <- function(scenario, arm, y, a, w, delta, q, g1, p0, p1, two_arm) {
  eligible <- which(delta == 0)
  if (length(eligible) < 4L) stop("the probe needs four rows with a zero response indicator")
  first <- plant_at(y, eligible[1:2])
  last <- plant_at(y, rev(eligible)[1:2])
  planted <- scale_of(first, a, q, delta)
  unplanted <- scale_of(y, a, q, delta)
  pdelta <- cbind(p0, p1)
  at_first <- fit_at(first, a, w, delta, q, g1, pdelta)
  at_last <- fit_at(last, a, w, delta, q, g1, pdelta)
  for (fit in list(at_first, at_last)) {
    if (fit$Qinit$type != "user-supplied values" ||
        fit$g$type != "user-supplied values" ||
        fit$g.Delta$type != "user-supplied values") {
      stop("tmle refitted a nuisance instead of using the supplied predictions")
    }
  }
  reference <- published(at_first, two_arm)
  moved <- max(abs(published(at_last, two_arm) - reference))
  rebuilt <- max(abs(rebuild(first, a, delta, q, g1, p0, p1, two_arm) - reference))
  scale_exact <- identical(planted, as.numeric(q_bounds))
  unplanted_differs <- !identical(unplanted, as.numeric(q_bounds))
  data.frame(
    scenario = scenario,
    replicate = probe_replicate,
    arm = arm,
    planted_scale_lower = planted[[1]],
    planted_scale_upper = planted[[2]],
    unplanted_scale_lower = unplanted[[1]],
    unplanted_scale_upper = unplanted[[2]],
    scale_exact = scale_exact,
    unplanted_scale_differs = unplanted_differs,
    moved_rows_difference = moved,
    rebuild_difference = rebuilt,
    rebuild_tolerance = rebuild_tolerance,
    passed = scale_exact && unplanted_differs && moved == 0 && rebuilt <= rebuild_tolerance,
    stringsAsFactors = FALSE
  )
}

rows <- list()
for (scenario in names(continuous)) {
  arms <- continuous[[scenario]]
  frame <- samples[samples$scenario == scenario & samples$replicate == probe_replicate, ]
  if (nrow(frame) == 0L) stop(sprintf("the probe replication of %s is missing", scenario))
  n <- nrow(frame)
  if (arms == 2L) {
    q <- cbind(frame$q0, frame$q1)
    rows[[length(rows) + 1L]] <- probe(
      scenario, "both", frame$Y, frame$A, data.frame(W = frame$W), frame$Delta, q,
      frame$g1, frame$pi0, frame$pi1, TRUE
    )
  } else {
    for (code in 0:2) {
      q <- frame[[paste0("q", code)]]
      p <- frame[[paste0("pi", code)]]
      delta <- as.numeric(frame$A == code & frame$Delta == 1)
      y <- ifelse(delta == 1, frame$Y, NA_real_)
      rows[[length(rows) + 1L]] <- probe(
        scenario, three_arm_labels[[code + 1L]], y, rep(1, n),
        data.frame(A_original = frame$A, W = frame$W), delta, cbind(q, q),
        frame[[paste0("g", code)]], p, p, FALSE
      )
    }
  }
}
out <- do.call(rbind, rows)
write.csv(out, args$output, row.names = FALSE)
print(out)
cat(study_version("tmle"), "\n", sep = "")
