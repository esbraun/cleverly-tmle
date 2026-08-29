suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_crossfit_adapter.R")

set.seed(20260929)
n_clusters <- 30L
cluster_size <- 4L
n <- n_clusters * cluster_size
cluster <- rep(seq_len(n_clusters), each = cluster_size)
W1 <- rnorm(n)
W2 <- rnorm(n)
A <- rbinom(n, 1, plogis(0.3 * W1 + 0.6 * W2))
Y <- 1 + A + 0.8 * W1 + 0.5 * W2 + rep(rnorm(n_clusters), each = cluster_size) + rnorm(n)
natural <- data.frame(W1, W2, A, Y, cluster)
shifted <- natural
shifted$A <- 1
assignment <- rep(0:4, each = 6L)[cluster]
p1 <- plogis(0.3 * W1 + 0.6 * W2)
ratio <- matrix(ifelse(A == 1, 1 / p1, 0), ncol = 1L)

fit <- lmtp_tmle_with_folds(
  natural,
  shifted,
  trt = "A",
  outcome = "Y",
  baseline = c("W1", "W2"),
  id = "cluster",
  outcome_type = "continuous",
  fold_assignment = assignment,
  density_ratios = ratio,
  control = lmtp_control(
    .trim = 1,
    .learners_outcome_folds = 2,
    .learners_trt_folds = 2,
    .return_full_fits = TRUE
  )
)
stopifnot(
  is.finite(fit$estimate@x),
  is.finite(fit$estimate@std_error),
  identical(fit$estimate@id, as.character(cluster)),
  identical(fit$fold_assignment, as.integer(assignment))
)

split_assignment <- assignment
split_assignment[[1]] <- (split_assignment[[1]] + 1L) %% 5L
message <- tryCatch(
  {
    lmtp_tmle_with_folds(
      natural,
      shifted,
      trt = "A",
      outcome = "Y",
      baseline = c("W1", "W2"),
      id = "cluster",
      outcome_type = "continuous",
      fold_assignment = split_assignment,
      density_ratios = ratio
    )
    ""
  },
  error = function(condition) conditionMessage(condition)
)
stopifnot(grepl("split across folds", message, fixed = TRUE))
cat(sprintf("lmtp %s clustered adapter smoke passed\n", packageVersion("lmtp")))
