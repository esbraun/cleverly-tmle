suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_crossfit_adapter.R")
source("/fixture/study_harness.R")
source("/fixture/ltmle_regimen_adapter.R")
options(digits = 17)

# The clustered counterpart of lmtp_ltmle/run_study.R: the same panel, regimens, exact
# per-node density ratios and SL.glm sequential regressions, with id = "id" and the realized
# grouped fold column. The standard error is ife's clustered one, and each contrast subtracts
# two ife objects that keep the same id vector, as lmtp_clustered_tmle/run_study.R does.

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)

scenario <- "clustered_end_of_study"
dynamic_label <- "treat then continue if l2 positive"
plans <- c("never", "always", dynamic_label)

exact_ratios <- function(frame, arms) {
  # The law's own per-node density ratios, as lmtp_ltmle/run_study.R writes them. The cluster
  # component of the study law enters the final outcome only, so the mechanism is unchanged.
  a1 <- arms[, 1]
  a2 <- arms[, 2]
  l2 <- ifelse(is.na(frame$L2), 0, frame$L2)

  p_a1 <- plogis(0.3 * frame$W1 - 0.4 * frame$W2)
  g1 <- ifelse(a1 == 1, p_a1, 1 - p_a1)
  c1 <- plogis(2.2 + 0.3 * frame$W1 - 0.3 * a1)

  p_a2 <- plogis(0.5 * l2 + 0.6 * a1 - 0.2 * frame$W2)
  g2 <- ifelse(a2 == 1, p_a2, 1 - p_a2)
  c2 <- plogis(2.4 + 0.2 * l2)

  followed1 <- frame$A1 == a1 & frame$C1 == 1
  followed2 <- !is.na(frame$A2) & frame$A2 == a2 & !is.na(frame$C2) & frame$C2 == 1
  cbind(
    ifelse(followed1, 1 / (g1 * c1), 0),
    ifelse(followed2, 1 / (g2 * c2), 0)
  )
}

cluster_sum_se <- function(eif, cluster) {
  # The cluster-sum rule this package reports: J var(S_c) / n^2 over the row curve.
  sums <- tapply(eif, cluster, sum)
  sqrt(length(sums) * var(sums) / length(eif)^2)
}

fit_plan <- function(frame, label) {
  natural <- frame[c("W1", "W2", "A1", "C1", "L2", "A2", "C2", "Y", "id")]
  shifted <- natural
  arms <- regimen_arms(frame, label)
  shifted$A1 <- arms[, 1]
  shifted$A2 <- arms[, 2]
  fit <- lmtp_tmle_with_folds(
    natural,
    shifted,
    trt = c("A1", "A2"),
    outcome = "Y",
    baseline = c("W1", "W2"),
    time_vary = list(NULL, "L2"),
    cens = c("C1", "C2"),
    id = "id",
    outcome_type = "binomial",
    fold_assignment = frame$fold,
    learners_outcome = "SL.glm",
    learners_trt = "SL.glm",
    density_ratios = exact_ratios(frame, arms),
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
  if (!identical(fit$estimate@id, as.character(frame$id))) {
    stop("the ife estimate did not retain the cluster identifier")
  }
  # The identity the paired standard errors rest on: at equal cluster sizes ife's cluster-mean
  # rule equals the cluster-sum rule on lmtp's own row curve.
  hand <- cluster_sum_se(fit$estimate@eif, frame$id)
  if (abs(fit$estimate@std_error - hand) > 1e-10) {
    stop(sprintf(
      "ife's clustered SE %.17g differs from the cluster-sum SE %.17g", fit$estimate@std_error, hand
    ))
  }
  list(estimate = fit$estimate, initial = mean(fit$initial))
}

truth_for <- function(replicate, estimand) {
  selected <- truths$replicate == replicate & truths$scenario == scenario & truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows", sum(selected)))
  truths$truth[selected]
}

row_for <- function(replicate, name, fit, n) {
  truth <- truth_for(replicate, name)
  low <- fit$estimate@conf_int[[1]]
  high <- fit$estimate@conf_int[[2]]
  data.frame(
    implementation = "lmtp",
    scenario = scenario,
    replicate = replicate,
    n = n,
    estimand = name,
    truth = truth,
    estimate = fit$estimate@x,
    inference_estimate = fit$estimate@x,
    std_error = fit$estimate@std_error,
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
  fits <- setNames(lapply(plans, function(label) fit_plan(frame, label)), plans)
  rows <- lapply(plans, function(label) {
    row_for(replicate, sprintf("ey_regimen[%s]", label), fits[[label]], nrow(frame))
  })
  for (label in c("always", dynamic_label)) {
    contrast <- list(
      estimate = fits[[label]]$estimate - fits[["never"]]$estimate,
      initial = fits[[label]]$initial - fits[["never"]]$initial
    )
    rows[[length(rows) + 1]] <- row_for(
      replicate, sprintf("ate_regimen[%s vs never]", label), contrast, nrow(frame)
    )
  }
  do.call(rbind, rows)
}

expected <- length(unique(truths$replicate))
results <- study_stream(paths$samples, fit_one, expected)
study_collect(
  results, expected, paths$output,
  versions = c(study_version("lmtp"), study_version("ife"))
)
