suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_crossfit_adapter.R")
source("/fixture/lmtp_weighted_glm_adapter.R")

design <- weighted_glm_design(
  data.frame(W = c(-1, 1), obs_weight_aux = c(2, 3)),
  data.frame(W = c(0, 2), obs_weight_aux = c(4, 5))
)
stopifnot(
  identical(names(design$X), "W"),
  identical(names(design$newX), "W"),
  identical(design$weights, c(2, 3))
)

set.seed(20261102)
n <- 240L
W1 <- rnorm(n)
W2 <- rnorm(n)
A <- rbinom(n, 1, plogis(0.3 * W1 - 0.4 * W2))
Y <- rbinom(n, 1, plogis(-0.6 + 0.5 * A + 0.4 * W1 - 0.2 * W2))
selection <- ifelse(W1 > 0, 0.3, 0.9)
weights <- 1 / selection
natural <- data.frame(W1, W2, A, Y, obs_weight_aux = weights)
shifted <- natural
shifted$A <- 1
assignment <- rep(0:4, length.out = n)
p_a <- plogis(0.3 * W1 - 0.4 * W2)
density_ratios <- matrix(ifelse(A == 1, 1 / p_a, 0), ncol = 1L)

fit <- lmtp_tmle_with_folds(
  natural,
  shifted,
  trt = "A",
  outcome = "Y",
  baseline = c("W1", "W2", "obs_weight_aux"),
  weights = weights,
  outcome_type = "binomial",
  fold_assignment = assignment,
  learners_outcome = "SL.weighted.glm",
  learners_trt = "SL.glm",
  density_ratios = density_ratios,
  control = lmtp_control(
    .trim = 1,
    .learners_outcome_folds = 2,
    .learners_trt_folds = 2,
    .return_full_fits = TRUE
  )
)
normalized <- weights / mean(weights)
weighted_eif <- normalized * fit$estimate@eif
ht_standard_error <- sd(weighted_eif) / sqrt(n)
hajek_standard_error <- sd(normalized * (fit$estimate@eif - fit$estimate@x)) / sqrt(n)
stopifnot(
  isTRUE(all.equal(as.numeric(fit$estimate@weights), normalized, tolerance = 1e-12)),
  abs(mean(weighted_eif) - fit$estimate@x) < 0.02,
  abs(mean(weighted_eif)) > 0.1,
  isTRUE(all.equal(fit$estimate@std_error, ht_standard_error, tolerance = 1e-12)),
  is.finite(hajek_standard_error),
  abs(ht_standard_error - hajek_standard_error) > 1e-8
)

bad_weight_message <- tryCatch(
  {
    lmtp_tmle_with_folds(
      natural,
      shifted,
      trt = "A",
      outcome = "Y",
      baseline = c("W1", "W2", "obs_weight_aux"),
      weights = replace(weights, 1, 0),
      fold_assignment = assignment
    )
    ""
  },
  error = function(condition) conditionMessage(condition)
)
stopifnot(identical(bad_weight_message, "weights must be finite positive numbers"))
cat(sprintf("lmtp %s weighted adapter smoke passed\n", packageVersion("lmtp")))
