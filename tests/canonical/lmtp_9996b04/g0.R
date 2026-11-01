# Gate G0 of X59 (X59-plan.md section 4.1): the lmtp 9996b04 image and its adapters.
#
# Run from the repository root by tests/canonical/x59_smoke.py g0, which compares the outputs
# across the two images:
#
#   docker run --rm -v "$PWD/tests/canonical:/fixture:ro" -v "$OUT:/out" IMAGE \
#     /fixture/lmtp_9996b04/g0.R MODE /out
#
# MODE is one of
#
# * image: on cleverly-lmtp:9996b04 only.  The version assertion, every lmtp internal the
#   adapters locate, the structure guard on cf_tmle's fits, and the five-fold validation-row
#   score of the fluctuation, which 9996b04 fits on each fold's validation rows (X39 audit,
#   verdict 2): its largest absolute value is written to image.csv and must be at most 1e-8.
# * adapters: on both images.  One fixed sample through the supplied-fold lmtp_tmle adapter and
#   the known-policy adapter (the shipped ones on 1.5.4, the 9996b04 ones on 1.5.5) at one fold,
#   and through the shipped lmtp_sdr_with_folds at one and five folds.  The estimates and the
#   unit influence curves are written at 17 digits, so the comparison is byte identity.
# * learners: on cleverly-lmtp:9996b04.  Reads /out/learner-input.csv, which the smoke writes
#   with cleverly's own node fits beside it, and writes R's predictions to learners.csv.
# * packages: on both images.  Writes the installed package list, so a cross-image identity
#   failure can be told from dependency drift.

suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/study_harness.R")
options(digits = 17)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: g0.R MODE OUTPUT_DIR")
mode <- args[[1]]
out <- args[[2]]
version <- as.character(packageVersion("lmtp"))
pinned <- identical(version, "1.5.5")
if (!version %in% c("1.5.4", "1.5.5")) stop(sprintf("lmtp %s is neither pinned commit", version))

write_csv <- function(frame, name) {
  write.csv(frame, file.path(out, name), row.names = FALSE)
}

fixture <- function(n = 400L, censored = TRUE) {
  # A three-node binary law with censoring, a deterministic plan, and a coin plan whose
  # probabilities are multiples of one quarter.  The supplied ratios are the law's own.
  set.seed(20261101)
  W <- sample(0:3, n, replace = TRUE)
  frame <- data.frame(W = W)
  alive <- rep(TRUE, n)
  ratio_always <- matrix(0, n, 3)
  ratio_coin <- matrix(0, n, 3)
  for (t in 1:3) {
    previous <- if (t == 1) rep(0, n) else frame[[paste0("A", t - 1)]]
    if (t > 1) {
      frame[[paste0("L", t)]] <- ifelse(alive, rbinom(n, 1, plogis(-0.3 + 0.4 * previous)), NA)
    }
    covariate <- if (t > 1) ifelse(is.na(frame[[paste0("L", t)]]), 0, frame[[paste0("L", t)]]) else 0
    g <- plogis(0.2 * W - 0.4 + 0.5 * covariate)
    frame[[paste0("A", t)]] <- ifelse(alive, rbinom(n, 1, g), NA)
    stay <- if (censored) plogis(2.5 + 0.2 * covariate) else rep(1, n)
    frame[[paste0("C", t)]] <- ifelse(alive, if (censored) rbinom(n, 1, stay) else 1L, NA)
    arm <- frame[[paste0("A", t)]]
    observed <- alive & frame[[paste0("C", t)]] == 1
    observed[is.na(observed)] <- FALSE
    g_arm <- ifelse(arm == 1, g, 1 - g)
    ratio_always[, t] <- ifelse(observed & arm == 1, 1 / (g * stay), 0)
    ratio_coin[, t] <- ifelse(observed, 0.5 / (g_arm * stay), 0)
    alive <- observed
  }
  frame$Y <- ifelse(alive, rbinom(n, 1, plogis(-0.2 + 0.3 * frame$L3 + 0.5 * frame$A3)), NA)
  frame$..unit <- seq_len(n)
  coin <- lapply(1:3, function(t) matrix(rep(c(0, 0, 1, 1), n), nrow = n, byrow = TRUE))
  list(frame = frame, ratio_always = ratio_always, ratio_coin = ratio_coin, coin = coin)
}

nodes <- list(
  trt = paste0("A", 1:3), cens = paste0("C", 1:3), time_vary = list(NULL, "L2", "L3"),
  baseline = "W", outcome = "Y"
)

if (mode == "packages") {
  packages <- installed.packages()[, c("Package", "Version")]
  write_csv(as.data.frame(packages[order(packages[, "Package"]), ]), sprintf("packages-%s.csv", version))
} else if (mode == "image") {
  if (!pinned) stop("the image gate runs on lmtp 1.5.5 (9996b04) only")
  source("/fixture/lmtp_9996b04_crossfit_adapter.R")
  for (name in c(
    "LmtpTask", "cf_tmle", "cf_sdr", "cf_density_ratios", "theta_dr", "get_folded_data",
    "current_trt", "run_ensemble"
  )) lmtp_internal(name)
  data <- fixture()
  frame <- data$frame
  shifted <- frame
  for (a in nodes$trt) shifted[[a]] <- 1
  assignment <- rep(0:4, length.out = nrow(frame))
  Task <- lmtp_internal("LmtpTask")
  task <- Task$new(
    data = frame, shifted = shifted, A = nodes$trt, Y = nodes$outcome, L = nodes$time_vary,
    W = nodes$baseline, C = nodes$cens, D = NULL, k = Inf, id = "..unit",
    outcome_type = "binomial", bounds = NULL, folds = 5L, weights = NULL
  )
  task$folds <- fold_list(assignment)
  control <- lmtp_control(.trim = 1, .learners_outcome_folds = 2)
  regressions <- lmtp_internal("cf_tmle")(task, data$ratio_always, "SL.glm", control, function(...) NULL)
  lmtp_9996b04_fits_guard(regressions$fits)
  # The fluctuation's score on each fold's validation rows, node by node.
  cumulative <- t(apply(data$ratio_always, 1, cumprod))
  scores <- c()
  for (fold in seq_along(task$folds)) {
    rows <- task$folds[[fold]]$validation_set
    for (t in 1:3) {
      residual <- regressions$shifted[rows, t + 1] - regressions$natural[rows, t]
      weight <- cumulative[rows, t]
      keep <- weight > 0 & !is.na(residual)
      scores <- c(scores, sum(weight[keep] * residual[keep]) / length(rows))
    }
  }
  write_csv(data.frame(
    lmtp = version, fits_class = class(regressions$fits)[[1]], fits_rows = nrow(regressions$fits),
    max_validation_score = max(abs(scores))
  ), "image.csv")
} else if (mode == "adapters") {
  source("/fixture/lmtp_sdr_adapter.R")
  data <- fixture()
  frame <- data$frame
  shifted <- frame
  for (a in nodes$trt) shifted[[a]] <- 1
  variables <- c("W", "A1", "C1", "L2", "A2", "C2", "L3", "A3", "C3", "Y", "..unit")
  one <- rep(0L, nrow(frame))
  # The shipped policy adapter takes no censoring node, so the policy pair runs on the
  # uncensored version of the fixture on both images.
  plain <- fixture(censored = FALSE)
  policy_columns <- c("W", "A1", "L2", "A2", "L3", "A3", "Y")
  if (pinned) {
    source("/fixture/lmtp_9996b04_policy_adapter.R")
    tmle <- lmtp_9996b04_tmle_with_folds(
      frame[variables], shifted[variables], trt = nodes$trt, outcome = "Y", baseline = "W",
      time_vary = nodes$time_vary, cens = nodes$cens, id = "..unit", fold_assignment = one,
      density_ratios = data$ratio_always
    )
    policy <- lmtp_9996b04_policy_tmle(
      plain$frame[policy_columns], plain$coin, plain$ratio_coin, trt = nodes$trt,
      outcome = "Y", baseline = "W", time_vary = nodes$time_vary, cens = NULL,
      fold_assignment = one, copies = 4L
    )
  } else {
    source("/fixture/lmtp_policy_adapter.R")
    shipped <- lmtp_tmle_with_folds(
      frame[variables], shifted[variables], trt = nodes$trt, outcome = "Y", baseline = "W",
      time_vary = nodes$time_vary, cens = nodes$cens, id = "..unit", fold_assignment = one,
      density_ratios = data$ratio_always,
      control = lmtp_control(
        .trim = 1, .learners_outcome_folds = 2, .learners_trt_folds = 2, .return_full_fits = TRUE
      )
    )
    tmle <- list(estimate = shipped$estimate@x, ic = as.numeric(shipped$estimate@eif))
    policy <- lmtp_policy_tmle(
      plain$frame[policy_columns], plain$coin, plain$ratio_coin, 0.5 * (plain$ratio_coin > 0),
      trt = nodes$trt, outcome = "Y", baseline = "W", time_vary = nodes$time_vary,
      fold_assignment = one, copies = 4L, refuse_bound = FALSE
    )
  }
  rows <- list(
    data.frame(fit = "tmle", unit = seq_along(tmle$ic), estimate = tmle$estimate, ic = tmle$ic),
    data.frame(fit = "policy", unit = seq_along(policy$ic), estimate = policy$estimate, ic = policy$ic)
  )
  for (folds in c(1L, 5L)) {
    for (a in nodes$trt) frame[[sprintf("shift__always__%s__1", sub("A", "", a))]] <- 1
    for (t in 1:3) frame[[sprintf("ratio__always__%d", t)]] <- data$ratio_always[, t]
    frame$fold <- if (folds == 1L) one else rep(0:4, length.out = nrow(frame))
    sdr <- lmtp_sdr_with_folds(
      frame, "always", trt = nodes$trt, outcome = "Y", baseline = "W",
      time_vary = nodes$time_vary, cens = nodes$cens
    )
    rows[[length(rows) + 1L]] <- data.frame(
      fit = sprintf("sdr_%d", folds), unit = seq_along(sdr$ic), estimate = sdr$estimate, ic = sdr$ic
    )
  }
  write_csv(do.call(rbind, rows), sprintf("adapters-%s.csv", version))
} else if (mode == "learners") {
  if (!pinned) stop("the learner gate runs on lmtp 1.5.5 (9996b04) only")
  source("/fixture/lmtp_sdr_adapter.R")
  source("/fixture/lmtp_horizon_learners.R")
  input <- read.csv(file.path(out, "learner-input.csv"), check.names = FALSE)
  # One row per training or prediction row: case, family, role (fit or predict), the target,
  # and the design columns x1..xp.  The smoke writes cleverly's prediction beside each
  # prediction row, and compares R's.
  predictions <- lapply(split(input, input$case), function(case) {
    columns <- grep("^x[0-9]+$", names(case), value = TRUE)
    train <- case[case$role == "fit", ]
    test <- case[case$role == "predict", ]
    design <- as.matrix(train[columns])
    family <- unique(case$family)
    coefficient <- if (family == "gaussian") {
      horizon_lstsq(cbind(1, design), train$target)
    } else {
      horizon_irls(design, train$target)
    }
    eta <- as.numeric(cbind(1, as.matrix(test[columns])) %*% coefficient)
    data.frame(case = test$case, row = test$row, r_prediction = if (family == "gaussian") eta else plogis(eta))
  })
  write_csv(do.call(rbind, predictions), "learners.csv")
} else {
  stop(sprintf("unknown G0 mode %s", mode))
}
cat(sprintf("g0 %s: lmtp %s ok\n", mode, version))
