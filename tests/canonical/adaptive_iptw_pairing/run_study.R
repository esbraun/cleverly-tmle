suppressPackageStartupMessages(library(drtmle))
source("/fixture/study_harness.R")
options(digits = 17)

# R drtmle 1.1.2 `adaptive_iptw` on the samples and mechanisms the Python phase wrote
# (tests/studies/canonical_adaptive_iptw_pairing.py).  One call per key, with a_0 a single
# level, so each key runs its own targeting loop, as the package's does.  `gn` is the package's
# initial mechanism at the key; drtmle fits only the reduced regression Qr, on the key's
# polynomial in qlogis(gn).  Every recode is built here from the raw columns, and each key's
# mechanism is composed here from the raw fitted factors the Python phase shipped (the treatment
# factor, the outcome and treatment observation factors, and their truncation rules).  The
# composition check stops the run when it differs from the mechanism cleverly divided by.
#
# Folds.  adaptive_iptw draws them inline with split(sample(seq_len(n)), rep(1:K, length = n)).
# A cross-fitted call runs a copy of adaptive_iptw whose enclosing environment is a child of the
# drtmle namespace holding one binding, `sample`, which returns the permutation the Python phase
# dealt its folds from.  The body and formals are drtmle's.
#
# Checks on every call: the fold check (drtmle's initial Qr equals a default-control glm rebuilt
# here on the supplied folds and rows, 1e-10, fatal), the nuisance gate (the package's initial Qr
# against that glm run to convergence, 1e-8; a miss is a recorded failure of the replicate) and
# the curve check (each rebuilt curve's variance against drtmle's cov, relative 1e-10, fatal).
# Before the call, the composition check (the composed mechanism against cleverly's, 1e-12,
# fatal) and, after it, the range check (drtmle's final mechanism inside [tolg, 1 - tolg]).

NUISANCE_GATE <- 1e-8
FOLD_CHECK <- 1e-10
CURVE_CHECK <- 1e-10
COMPOSITION_CHECK <- 1e-12
N_FOLDS <- 10L
TOL_IC <- 1e-8
MAX_ITER <- 100L
CRITICAL <- qnorm(0.975)

# Scenario -> the degree of the reduced basis (tests/studies/canonical_adaptive_iptw.py DEGREE).
DEGREE <- c(
  pd_known_arms = 3L, pd_known_arms_one_step = 3L, pd_known_regimes = 3L,
  pd_known_regimes_one_step = 3L, st_known = 1L, p_insample = 1L, m_insample = 3L,
  c_l1_insample = 3L, c_l3_insample = 3L, pd_regimes_insample = 3L, cn_insample = 5L,
  rm = 2L, ma_arms = 2L, ma_regime = 2L, p_rate_insample = 1L, m_rate_insample = 3L,
  c_l1_crossfit = 3L, c_l1_crossfit_one_step = 3L, c_l3_crossfit = 3L, cn_crossfit = 5L, ma_crossfit = 2L
)
ONE_STEP <- c("pd_known_arms_one_step", "pd_known_regimes_one_step", "c_l1_crossfit_one_step")
CROSS_FITTED <- c(
  "pd_known_arms", "pd_known_arms_one_step", "pd_known_regimes", "pd_known_regimes_one_step",
  "st_known", "c_l1_crossfit", "c_l1_crossfit_one_step", "c_l3_crossfit", "cn_crossfit", "ma_crossfit"
)
# The arm labels of a string-coded treatment, by code (ARM_LABELS in the Python module).
ARM_LABELS <- list(
  m_insample = c("high", "low", "medium"),
  m_rate_insample = c("high", "low", "medium"),
  c_l3_insample = c("high", "low", "mid"),
  c_l3_crossfit = c("high", "low", "mid")
)
LOG_SCALE <- c("rr", "or")

pd_rule <- function(frame) as.numeric(frame$W1 > 0)
pd_stochastic_one <- function(frame) plogis(frame$W1)
rm_rule <- function(frame) as.numeric(frame$W != 0)
rm_stochastic_one <- function(frame) c(0.2, 0.8, 0.8)[as.integer(frame$W) + 1L]

args <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(args$truths, stringsAsFactors = FALSE)
truth_key <- paste(truths$scenario, truths$replicate, truths$estimand, sep = "|")

# ------------------------------------------------------------------------------- the call

with_folds <- function(permutation) {
  # A copy of adaptive_iptw that draws `permutation` where it would call sample(seq_len(n)).
  copy <- drtmle::adaptive_iptw
  if (!identical(sort(permutation), seq_along(permutation))) {
    study_fatal_error("the fold permutation is not a permutation of the rows")
  }
  calls <- 0L
  scope <- new.env(parent = asNamespace("drtmle"))
  scope$calls <- function() calls
  scope$sample <- function(x, ...) {
    if (length(list(...))) study_fatal_error("the fold stub got arguments beyond x")
    if (!identical(as.integer(x), seq_along(permutation))) {
      study_fatal_error("the fold stub was not called as sample(seq_len(n))")
    }
    calls <<- calls + 1L
    if (calls > 1L) study_fatal_error("adaptive_iptw called sample twice")
    permutation
  }
  environment(copy) <- scope
  if (!identical(sort(ls(scope)), c("calls", "sample")) || SAMPLE_CALLS != 1L) {
    study_fatal_error("the fold stub's scope or drtmle's use of sample() is not as written")
  }
  if (!identical(body(copy), body(drtmle::adaptive_iptw)) ||
    !identical(formals(copy), formals(drtmle::adaptive_iptw))) {
    study_fatal_error("the fold copy of adaptive_iptw differs from drtmle's")
  }
  copy
}

# adaptive_iptw draws its folds with the one call to sample() in its body; a second call, or none,
# would make the stub's permutation not the folds.
count_calls <- function(expression, name) {
  if (is.call(expression)) {
    head <- expression[[1]]
    here <- as.integer(is.name(head) && identical(as.character(head), name))
    return(here + sum(vapply(as.list(expression), count_calls, integer(1), name = name)))
  }
  if (is.pairlist(expression) || is.list(expression)) {
    return(sum(vapply(as.list(expression), count_calls, integer(1), name = name)))
  }
  0L
}
SAMPLE_CALLS <- count_calls(body(drtmle::adaptive_iptw), "sample")

basis <- function(degree) {
  terms <- c("qlogis(gn)", if (degree >= 2L) sprintf("I(qlogis(gn)^%d)", 2:degree))
  paste(terms, collapse = " + ")
}

fit_rows <- function(A, Y, a) {
  # The rows drtmle's glm_Qr branch fits on: A == a with an observed outcome.
  hit <- A == a & !is.na(Y)
  hit[is.na(hit)] <- FALSE
  hit
}

degree_for <- function(gn, rows, declared) {
  # The package caps the degree at the number of distinct training values minus one.
  distinct <- length(unique(round(qlogis(gn[rows]), 10)))
  max(1L, min(declared, distinct - 1L))
}

refit_by_hand <- function(gn, target, rows, folds, formula, family, control) {
  # The initial Qr as drtmle's glm_Qr branch fits it: one glm per fold on the training rows
  # (every row in sample), predicted at the fold's rows.
  out <- rep(NA_real_, length(gn))
  for (valid in folds) {
    train <- if (length(valid) == length(gn)) valid else setdiff(seq_along(gn), valid)
    keep <- train[rows[train]]
    model <- stats::glm(
      stats::as.formula(paste0("Qrn ~", formula)),
      data = data.frame(Qrn = target[keep], gn = gn[keep]),
      family = family,
      control = control
    )
    out[valid] <- stats::predict(model, newdata = data.frame(gn = gn[valid]), type = "response")
  }
  out
}

run_call <- function(label, A, Y, a, gn, qr_python, lower, family, declared, permutation) {
  # One key.  Returns the per-row Horvitz-Thompson terms and corrections of both forms.
  n <- length(A)
  rows <- fit_rows(A, Y, a)
  degree <- if (length(unique(gn[rows])) == 1L) 0L else degree_for(gn, rows, declared)
  formula <- if (degree == 0L) "1" else basis(degree)
  cross <- !is.null(permutation)
  caller <- if (cross) with_folds(permutation) else drtmle::adaptive_iptw
  fit <- study_stage(sprintf("adaptive_iptw %s", label), caller(
    W = data.frame(row = seq_len(n)),
    A = A,
    Y = Y,
    family = family,
    a_0 = a,
    gn = list(gn),
    glm_Qr = formula,
    returnModels = FALSE,
    maxIter = MAX_ITER,
    tolIC = TOL_IC,
    tolg = lower,
    cvFolds = if (cross) N_FOLDS else 1L
  ))
  if (cross && environment(caller)$calls() != 1L) {
    study_fatal_error(sprintf("%s: adaptive_iptw did not draw its folds once", label))
  }
  qr0 <- as.numeric(fit$iptw_os_nuisance$Qrn[[1]])
  final <- as.numeric(fit$iptw_tmle_nuisance$gn[[1]])
  if (any(final < lower | final > 1 - lower)) {
    study_fatal_error(sprintf("%s: drtmle's final mechanism left [tolg, 1 - tolg]", label))
  }
  folds <- if (cross) split(permutation, rep(seq_len(N_FOLDS), length = n)) else list(seq_len(n))
  # The fold check: drtmle's initial Qr is the default-control glm on the supplied folds and
  # rows, so drtmle fitted what this runner says it fitted.
  by_hand <- refit_by_hand(gn, Y, rows, folds, formula, family, stats::glm.control())
  if (max(abs(by_hand - qr0)) > FOLD_CHECK) {
    study_fatal_error(sprintf("%s: drtmle's initial Qr is not the supplied folds' fit", label))
  }
  # The nuisance gate: the package's initial Qr against the same regression run to convergence.
  # drtmle's glm stops at glm.control()'s relative deviance change of 1e-8, which can leave its
  # predictions about 1e-6 from the converged fit; that gap is the comparator's own, and the
  # fold check above already ties drtmle to this regression.
  converged <- refit_by_hand(
    gn, Y, rows, folds, formula, family, stats::glm.control(epsilon = 1e-15, maxit = 100)
  )
  study_stage(sprintf("nuisance_gate %s", label), {
    gap <- max(abs(converged - qr_python))
    if (!is.finite(gap) || gap > NUISANCE_GATE) {
      stop(sprintf("initial Qr of %s differs from the package's by %.3g", label, gap))
    }
  })
  observed <- as.numeric(!is.na(A) & A == a & !is.na(Y))
  y <- ifelse(observed == 1, Y, 0)
  g_star <- as.numeric(fit$iptw_tmle_nuisance$gn[[1]])
  qr_star <- as.numeric(fit$iptw_tmle_nuisance$QrnStar[[1]])
  g0 <- as.numeric(fit$iptw_os_nuisance$gn[[1]])
  out <- list(
    ht_tmle = observed / g_star * y,
    correction_tmle = qr_star / g_star * (observed - g_star),
    ht_os = observed / g0 * y,
    correction_os = qr0 / g0 * (observed - g0),
    tmle = as.numeric(fit$iptw_tmle$est),
    os = as.numeric(fit$iptw_os$est),
    untargeted = as.numeric(fit$iptw$est)
  )
  # The curve check: drtmle's covariances are var(D) / n with denominator n - 1.
  for (form in c("tmle", "os")) {
    reported <- as.numeric(if (form == "tmle") fit$iptw_tmle$cov else fit$iptw_os$cov)
    centre <- if (form == "tmle") out$tmle else out$untargeted
    curve <- out[[paste0("ht_", form)]] - centre - out[[paste0("correction_", form)]]
    rebuilt <- stats::var(curve) / n
    if (abs(rebuilt - reported) > CURVE_CHECK * reported) {
      study_fatal_error(sprintf("%s: the rebuilt %s curve does not give drtmle's variance", label, form))
    }
  }
  if (abs(mean(out$ht_tmle) - out$tmle) > 1e-12 * max(1, abs(out$tmle))) {
    study_fatal_error(sprintf("%s: the rebuilt targeted mean is not drtmle's", label))
  }
  out
}

# ------------------------------------------------------------------- the composition

clip <- function(x, lower, upper) pmin(pmax(x, lower), upper)

arm_mechanism <- function(frame) {
  # The (n, K) mechanism cleverly divides an arm by, from the raw factors: the treatment factor
  # truncated into g_bounds (two arms on the simplex through g1, arm 0 its complement; else
  # column by column), times each observation factor floored at `floor` and capped at 1.
  arms <- Filter(function(code) !all(is.na(frame[[sprintf("tr_a%d", code)]])), 0:2)
  lower <- frame$g_lo[[1]]
  upper <- frame$g_hi[[1]]
  floor <- frame$floor[[1]]
  treatment <- sapply(arms, function(code) frame[[sprintf("tr_a%d", code)]])
  if (length(arms) == 2L && frame$simplex[[1]] == 1) {
    one <- clip(treatment[, 2], lower, upper)
    out <- cbind(1 - one, one)
  } else {
    out <- clip(treatment, lower, upper)
  }
  for (code in arms) {
    pi <- frame[[sprintf("pi_a%d", code)]]
    if (!all(is.na(pi))) out[, code + 1L] <- out[, code + 1L] * clip(pi, floor, 1)
  }
  if (!all(is.na(frame$pA))) out <- out * clip(frame$pA, floor, 1)
  out
}

composed <- function(frame, label, rule) {
  # One key's mechanism: an arm's column, the rule's arm for a deterministic regime, the
  # observation factor at the realised arm for the natural course; clipped into the key's
  # bounds, as cleverly clips it before targeting.
  lower <- column(frame, "lb", label)[[1]]
  upper <- column(frame, "ub", label)[[1]]
  if (label == "obs") {
    arm <- as.integer(frame$A)
    pi <- vapply(seq_len(nrow(frame)), function(i) {
      # A row with no recorded arm reads 0, which the floor then clips, as cleverly reads it.
      if (is.na(arm[[i]])) 0 else frame[[sprintf("pi_a%d", arm[[i]])]][[i]]
    }, numeric(1))
    return(clip(pi, lower, upper))
  }
  g <- arm_mechanism(frame)
  code <- switch(label,
    never = rep(0L, nrow(frame)),
    d = as.integer(rule(frame)),
    s0 = rep(0L, nrow(frame)),
    s1 = rep(1L, nrow(frame)),
    rep(as.integer(sub("^a", "", label)), nrow(frame))
  )
  clip(g[cbind(seq_len(nrow(frame)), code + 1L)], lower, upper)
}

# --------------------------------------------------------------------------- the means

mean_of <- function(calls, form) {
  # The estimate of a sum of keys, its curve, and its untargeted value.  Each key's estimate is
  # drtmle's own; its curve is the row term minus the estimate.  The one-step estimate is the
  # mean of its row terms exactly, and the targeted one up to drtmle's score at its stop.
  total <- 0
  curve <- 0
  initial <- 0
  for (call in calls) {
    estimate <- if (form == "tmle") call$tmle else call$os
    total <- total + estimate
    curve <- curve + call[[paste0("ht_", form)]] - call[[paste0("correction_", form)]] - estimate
    initial <- initial + call$untargeted
  }
  list(estimate = total, curve = curve, initial = initial)
}

pooled <- function(parts, n) {
  # One arm mean over strata: each stratum's terms in its rows, divided by the whole n.
  estimate <- 0
  initial <- 0
  for (part in parts) {
    estimate <- estimate + part$estimate * part$size / n
    initial <- initial + part$initial * part$size / n
  }
  curve <- unlist(lapply(parts, function(part) part$curve + part$estimate)) - estimate
  list(estimate = estimate, curve = curve, initial = initial)
}

contrast <- function(left, right) {
  list(
    estimate = left$estimate - right$estimate,
    curve = left$curve - right$curve,
    initial = left$initial - right$initial
  )
}

ratio <- function(one, zero, kind) {
  if (kind == "rr") {
    estimate <- one$estimate / zero$estimate
    log_curve <- one$curve / one$estimate - zero$curve / zero$estimate
    initial <- one$initial / zero$initial
  } else {
    odds <- function(p) p / (1 - p)
    estimate <- odds(one$estimate) / odds(zero$estimate)
    log_curve <- one$curve / (one$estimate * (1 - one$estimate)) -
      zero$curve / (zero$estimate * (1 - zero$estimate))
    initial <- odds(one$initial) / odds(zero$initial)
  }
  list(estimate = estimate, curve = log_curve, initial = initial, log = TRUE)
}

attributable_fraction <- function(observed, unexposed) {
  list(
    estimate = (observed$estimate - unexposed$estimate) / observed$estimate,
    curve = unexposed$estimate / observed$estimate^2 * observed$curve -
      unexposed$curve / observed$estimate,
    initial = (observed$initial - unexposed$initial) / observed$initial
  )
}

# ---------------------------------------------------------------------- the scenarios

column <- function(frame, prefix, label) frame[[sprintf("%s_%s", prefix, label)]]

RULES <- list(
  pd_known_regimes = pd_rule, pd_known_regimes_one_step = pd_rule, pd_regimes_insample = pd_rule,
  rm = rm_rule, ma_regime = rm_rule
)

one_call <- function(frame, label, A, Y, a, family, permutation) {
  name <- sprintf("%s/%s/%s", frame$scenario[[1]], frame$replicate[[1]], label)
  gn <- composed(frame, label, RULES[[frame$scenario[[1]]]])
  gap <- max(abs(gn - column(frame, "gn", label)))
  if (!is.finite(gap) || gap > COMPOSITION_CHECK) {
    study_fatal_error(sprintf("%s: the composed mechanism differs from cleverly's by %.3g", name, gap))
  }
  run_call(
    name,
    A, Y, a,
    gn = gn,
    qr_python = column(frame, "qr", label),
    lower = column(frame, "lb", label)[[1]],
    family = family,
    declared = DEGREE[[frame$scenario[[1]]]],
    permutation = permutation
  )
}

permutation_of <- function(frame, rows = NULL) {
  position <- if (is.null(rows)) frame$position else frame$position[rows]
  if (anyNA(position)) study_fatal_error("a cross-fitted sample has no fold position")
  order(position)
}

arm_means <- function(frame, codes, form, permutation) {
  out <- lapply(codes, function(code) {
    call <- one_call(frame, sprintf("a%d", code), frame$A, frame$Y, code, binomial(), permutation)
    mean_of(list(call), form)
  })
  names(out) <- as.character(codes)
  out
}

regime_means <- function(frame, rule, stochastic_one, form, permutation, never = TRUE) {
  rule_arm <- rule(frame)
  followed <- as.numeric(frame$A == rule_arm)
  s_one <- stochastic_one(frame)
  out <- list(
    d = mean_of(list(one_call(frame, "d", followed, frame$Y, 1, binomial(), permutation)), form),
    s = mean_of(
      list(
        one_call(frame, "s0", frame$A, (1 - s_one) * frame$Y, 0, gaussian(), permutation),
        one_call(frame, "s1", frame$A, s_one * frame$Y, 1, gaussian(), permutation)
      ),
      form
    )
  )
  if (never) {
    out$never <- mean_of(
      list(one_call(frame, "never", frame$A, frame$Y, 0, binomial(), permutation)), form
    )
  }
  out
}

keys_of <- function(frame) {
  scenario <- frame$scenario[[1]]
  form <- if (scenario %in% ONE_STEP) "os" else "tmle"
  cross <- scenario %in% CROSS_FITTED
  if (cross != !anyNA(frame$position)) {
    study_fatal_error(sprintf("%s: fold positions do not match the declared cross-fitting", scenario))
  }
  permutation <- if (cross && scenario != "st_known") permutation_of(frame)
  if (scenario %in% c("pd_known_arms", "pd_known_arms_one_step")) {
    m <- arm_means(frame, 0:1, form, permutation)
    return(list(
      ey0 = m[["0"]], ey1 = m[["1"]], ate = contrast(m[["1"]], m[["0"]]),
      rr = ratio(m[["1"]], m[["0"]], "rr"), or = ratio(m[["1"]], m[["0"]], "or")
    ))
  }
  if (scenario %in% c("pd_known_regimes", "pd_known_regimes_one_step", "pd_regimes_insample")) {
    m <- regime_means(frame, pd_rule, pd_stochastic_one, form, permutation)
    return(list(
      "ey_regime[never]" = m$never, "ey_regime[d]" = m$d, "ey_regime[s]" = m$s,
      "ate_regime[d vs never]" = contrast(m$d, m$never),
      "ate_regime[s vs never]" = contrast(m$s, m$never)
    ))
  }
  if (scenario == "st_known") {
    parts <- list("0" = list(), "1" = list())
    out <- list()
    for (level in sort(unique(frame$V))) {
      rows <- which(frame$V == level)
      stratum <- frame[rows, , drop = FALSE]
      m <- arm_means(stratum, 0:1, form, permutation_of(stratum))
      out[[sprintf("ate[V=%d]", as.integer(level))]] <- contrast(m[["1"]], m[["0"]])
      for (code in c("0", "1")) {
        parts[[code]][[length(parts[[code]]) + 1L]] <- c(m[[code]], size = length(rows))
      }
    }
    one <- pooled(parts[["1"]], nrow(frame))
    zero <- pooled(parts[["0"]], nrow(frame))
    return(c(list(ate = contrast(one, zero)), out))
  }
  if (scenario %in% c(
    "p_insample", "p_rate_insample", "c_l1_insample", "ma_arms", "c_l1_crossfit",
    "c_l1_crossfit_one_step", "ma_crossfit"
  )) {
    m <- arm_means(frame, 0:1, form, permutation)
    return(list(ey0 = m[["0"]], ey1 = m[["1"]], ate = contrast(m[["1"]], m[["0"]])))
  }
  if (scenario %in% names(ARM_LABELS)) {
    labels <- ARM_LABELS[[scenario]]
    m <- arm_means(frame, seq_along(labels) - 1L, form, permutation)
    names(m) <- labels
    out <- list()
    for (label in labels) out[[sprintf("ey[%s]", label)]] <- m[[label]]
    for (label in labels[-1]) {
      out[[sprintf("ate[%s vs %s]", label, labels[[1]])]] <- contrast(m[[label]], m[[labels[[1]]]])
    }
    return(out)
  }
  if (scenario %in% c("cn_insample", "cn_crossfit")) {
    natural <- one_call(frame, "obs", frame$Delta, frame$Y, 1, binomial(), permutation)
    obs <- mean_of(list(natural), form)
    zero <- mean_of(
      list(one_call(frame, "a0", frame$A, frame$Y, 0, binomial(), permutation)), form
    )
    return(list(
      ey_obs = obs, ey0 = zero, par = contrast(obs, zero),
      paf = attributable_fraction(obs, zero)
    ))
  }
  if (scenario == "rm") {
    m <- regime_means(frame, rm_rule, rm_stochastic_one, form, NULL, never = FALSE)
    return(list(
      "ey_regime[d]" = m$d, "ey_regime[s]" = m$s, "ate_regime[s vs d]" = contrast(m$s, m$d)
    ))
  }
  if (scenario == "ma_regime") {
    followed <- as.numeric(frame$A == rm_rule(frame))
    d <- mean_of(list(one_call(frame, "d", followed, frame$Y, 1, binomial(), NULL)), form)
    return(list("ey_regime[d]" = d))
  }
  stop(sprintf("unknown scenario %s", scenario))
}

fit_one <- function(group) {
  frame <- readRDS(group$path[[1]])
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  keys <- keys_of(frame)
  n <- nrow(frame)
  rows <- lapply(names(keys), function(estimand) {
    value <- keys[[estimand]]
    truth_row <- match(paste(scenario, replicate, estimand, sep = "|"), truth_key)
    if (is.na(truth_row)) stop(sprintf("missing truth for %s/%s/%s", scenario, replicate, estimand))
    truth <- truths$truth[[truth_row]]
    log_scale <- estimand %in% LOG_SCALE
    se <- sqrt(stats::var(value$curve) / length(value$curve))
    centre <- if (log_scale) log(value$estimate) else value$estimate
    low <- centre - CRITICAL * se
    high <- centre + CRITICAL * se
    if (log_scale) {
      low <- exp(low)
      high <- exp(high)
    }
    data.frame(
      implementation = "drtmle",
      scenario = scenario,
      replicate = replicate,
      n = n,
      estimand = estimand,
      truth = truth,
      estimate = value$estimate,
      inference_estimate = centre,
      std_error = se,
      ci_lower = low,
      ci_upper = high,
      inference_scale = if (log_scale) "log" else "identity",
      covered = as.integer(low <= truth && truth <= high),
      initial_estimate = value$initial,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

# ------------------------------------------------------------------------- the samples

split_samples <- function(path, directory, chunk_lines = 200000L) {
  # Reads the archive in chunks and saves each (scenario, replicate) block as its own file, so
  # the parent never holds the whole archive and each forked worker reads one block.  A block
  # cut by a chunk boundary is carried into the next chunk.
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  connection <- gzfile(path, open = "rt")
  on.exit(close(connection), add = TRUE)
  header <- readLines(connection, n = 1L)
  carry <- NULL
  index <- list()
  save_block <- function(block) {
    key <- sprintf("%s|%s", block$scenario[[1]], block$replicate[[1]])
    if (!is.null(index[[key]])) study_fatal_error(sprintf("block %s is not contiguous", key))
    file <- file.path(directory, sprintf("%06d.rds", length(index) + 1L))
    saveRDS(block, file)
    index[[key]] <<- data.frame(
      scenario = block$scenario[[1]], replicate = block$replicate[[1]], path = file,
      stringsAsFactors = FALSE
    )
  }
  repeat {
    lines <- readLines(connection, n = chunk_lines)
    at_end <- length(lines) == 0L
    current <- if (at_end) {
      NULL
    } else {
      utils::read.csv(text = c(header, lines), stringsAsFactors = FALSE, na.strings = c("", "NA"))
    }
    frame <- if (is.null(carry)) current else if (is.null(current)) carry else rbind(carry, current)
    carry <- NULL
    if (is.null(frame) || !nrow(frame)) break
    block_of <- paste(frame$scenario, frame$replicate, sep = "|")
    starts <- c(TRUE, block_of[-1] != block_of[-length(block_of)])
    block_id <- cumsum(starts)
    if (!at_end) {
      last <- block_id == max(block_id)
      carry <- frame[last, , drop = FALSE]
      frame <- frame[!last, , drop = FALSE]
      block_id <- block_id[!last]
    }
    for (block in split(frame, block_id)) save_block(block)
    if (at_end) break
  }
  unname(index)
}

blocks <- split_samples(args$samples, file.path(tempdir(), "adaptive_iptw_pairing_blocks"))
groups <- lapply(blocks, identity)
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
