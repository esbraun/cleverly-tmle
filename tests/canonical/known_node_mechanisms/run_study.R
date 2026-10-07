suppressPackageStartupMessages(library(ltmle))
source("/fixture/study_harness.R")
options(digits = 17)

# R ltmle 1.3-0 on the two-node SMART with the declared treatment and censoring factors as a
# numeric gform.  Column j of gform is P(A node j = 1) or P(C node j uncensored) given the
# observed past, which is what the Python phase declares, so neither side fits a mechanism.
# A row censored at C1 has no second-node probability; it is filled in range because
# nothing downstream of the censoring reads it, and the cumulative-bound check below refuses
# any fit where a bound binds.

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(paths$samples), stringsAsFactors = FALSE, check.names = FALSE)
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)

plans <- list(always = c(1, 1), never = c(0, 0))
qforms <- list(
  smart_q_correct = c(L1 = "Q.kplus1 ~ L0", Y = "Q.kplus1 ~ L0 + L1"),
  smart_q_wrong = c(L1 = "Q.kplus1 ~ 1", Y = "Q.kplus1 ~ L0")
)
z <- qnorm(0.975)

fit_plan <- function(frame, scenario, label) {
  plan <- plans[[label]]
  abar <- matrix(rep(plan, each = nrow(frame)), ncol = 2)
  g2 <- ifelse(is.na(frame$g2_1), 0.5, frame$g2_1)
  r2 <- ifelse(is.na(frame$r2), 1, frame$r2)
  gform <- cbind(frame$g1_1, frame$r1, g2, r2)
  data <- data.frame(
    L0 = frame$L0,
    A1 = frame$A1,
    C1 = BinaryToCensoring(is.uncensored = frame$C1),
    L1 = frame$L1,
    A2 = frame$A2,
    C2 = BinaryToCensoring(is.uncensored = ifelse(is.na(frame$C2), 0, frame$C2)),
    Y = frame$Y
  )
  targeted <- ltmle(
    data = data,
    Anodes = c("A1", "A2"),
    Cnodes = c("C1", "C2"),
    Lnodes = "L1",
    Ynodes = "Y",
    survivalOutcome = FALSE,
    Qform = qforms[[scenario]],
    gform = gform,
    abar = abar,
    gbounds = c(1e-8, 1),
    SL.library = "glm",
    stratify = TRUE,
    variance.method = "ic"
  )
  if (any(abs(targeted$cum.g - targeted$cum.g.unbounded) > 1e-12, na.rm = TRUE)) {
    stop(sprintf("%s activated a cumulative g bound", label))
  }
  first_q <- targeted$fit$Q[[1]]
  initial <- if (inherits(first_q, "no.Y.variation")) {
    unname(first_q$Y.value)
  } else {
    coefficients <- first_q[, "Estimate"]
    design <- model.matrix(~L0, data = frame)
    unname(mean(plogis(design[, names(coefficients), drop = FALSE] %*% coefficients)))
  }
  list(
    estimate = unname(targeted$estimates[["tmle"]]),
    initial = initial,
    ic = as.numeric(targeted$IC$tmle)
  )
}

row_for <- function(scenario, replicate, name, estimate, initial, ic, n) {
  selected <- truths$scenario == scenario & truths$replicate == replicate & truths$estimand == name
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows", sum(selected)))
  truth <- truths$truth[selected]
  standard_error <- sd(ic) / sqrt(n)
  low <- estimate - z * standard_error
  high <- estimate + z * standard_error
  data.frame(
    implementation = "ltmle-known-gform",
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
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  if (!scenario %in% names(qforms)) stop(sprintf("unknown scenario %s", scenario))
  fits <- setNames(lapply(names(plans), function(label) fit_plan(frame, scenario, label)), names(plans))
  n <- nrow(frame)
  rows <- lapply(names(plans), function(label) {
    fit <- fits[[label]]
    row_for(scenario, replicate, sprintf("ey_regimen[%s]", label), fit$estimate, fit$initial, fit$ic, n)
  })
  rows[[length(rows) + 1]] <- row_for(
    scenario,
    replicate,
    "ate_regimen[always vs never]",
    fits$always$estimate - fits$never$estimate,
    fits$always$initial - fits$never$initial,
    fits$always$ic - fits$never$ic,
    n
  )
  do.call(rbind, rows)
}

groups <- split(samples, interaction(samples$scenario, samples$replicate, drop = TRUE))
expected <- length(groups)
rm(samples)
invisible(gc())
results <- parallel::mclapply(
  seq_along(groups),
  study_fitter(groups, fit_one),
  mc.cores = study_cores(groups),
  mc.preschedule = FALSE
)
study_collect(
  results,
  expected = expected,
  output = paths$output,
  versions = study_version("ltmle"),
  key = c("scenario", "replicate")
)
