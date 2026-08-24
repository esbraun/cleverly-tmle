suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_crossfit_adapter.R")

set.seed(20260824)
n <- 240L
W <- rnorm(n)
A1 <- rbinom(n, 1, plogis(0.2 * W))
L2 <- rnorm(n, 0.3 * W + 0.4 * A1)
A2 <- rbinom(n, 1, plogis(0.2 * W + 0.4 * L2 + 0.3 * A1))
Y <- rbinom(n, 1, plogis(-0.4 + 0.2 * W + 0.3 * L2 + 0.4 * A2))
natural <- data.frame(W, A1, L2, A2, Y)
shifted <- natural
shifted$A1 <- 1
shifted$A2 <- 1
assignment <- rep(0:4, length.out = n)

fit <- lmtp_tmle_with_folds(
  natural,
  shifted,
  trt = c("A1", "A2"),
  outcome = "Y",
  baseline = "W",
  time_vary = list(NULL, "L2"),
  fold_assignment = assignment,
  control = lmtp_control(
    .trim = 1,
    .learners_outcome_folds = 2,
    .learners_trt_folds = 2
  )
)
summary <- ife::tidy(fit$estimate)
stopifnot(nrow(summary) == 1L, is.finite(summary$estimate), is.finite(summary$std.error))
cat(sprintf("lmtp %s adapter smoke passed\n", packageVersion("lmtp")))
