# Disposable smoke for `lmtp_competing_tmle_with_folds`.
#
# `smoke.R` covers the shared adapter and reaches none of what the competing wrapper adds: a
# populated `D`, the one-fold training-equals-validation branch the ordinary row runs on, and
# the supplied-density-ratio screens.  Each of those is a place a study can go wrong quietly,
# so each gets a case here.  Run it before a regeneration, as the fixture READMEs say.

suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_crossfit_adapter.R")
source("/fixture/lmtp_competing_adapter.R")

set.seed(20260825)
n <- 2000L

# The competing law's own missingness convention: a unit that had an event of either cause at
# the first node has no second node at all, so every later column is NA rather than carried
# forward.  That is the pattern `tests/discrete_law_competing.frame` writes and the study feeds
# to both implementations.
W <- rbinom(n, 1, 0.75)
A1 <- rbinom(n, 1, ifelse(W == 0, 0.50, 0.25))
C1 <- rbinom(n, 1, ifelse(W == 0 & A1 == 1, 0.50, 0.75))
J1 <- ifelse(C1 == 1, sample(0:2, n, replace = TRUE, prob = c(0.5, 0.25, 0.25)), NA)
R1 <- ifelse(is.na(J1), NA, as.numeric(J1 == 1))
D1 <- ifelse(is.na(J1), NA, as.numeric(J1 == 2))

at_risk <- !is.na(J1) & J1 == 0
blank <- rep(NA_real_, n)
L2 <- blank; A2 <- blank; C2 <- blank; R2 <- blank; D2 <- blank
m <- sum(at_risk)
L2[at_risk] <- rbinom(m, 1, 0.5)
A2[at_risk] <- rbinom(m, 1, 0.5)
C2[at_risk] <- rbinom(m, 1, 0.75)
uncensored <- at_risk & !is.na(C2) & C2 == 1
k <- sum(uncensored)
J2 <- sample(0:2, k, replace = TRUE, prob = c(0.5, 0.25, 0.25))
R2[uncensored] <- as.numeric(J2 == 1)
D2[uncensored] <- as.numeric(J2 == 2)

natural <- data.frame(W, A1, C1, R1, D1, L2, A2, C2, R2, D2)
shifted <- natural
shifted$A1 <- 1
shifted$A2 <- ifelse(is.na(natural$L2), NA, as.numeric(natural$L2 == 1))

arms <- cbind(rep(1, n), ifelse(is.na(natural$L2), 0, as.numeric(natural$L2 == 1)))
followed1 <- natural$A1 == arms[, 1] & natural$C1 == 1
g1 <- ifelse(W == 0, 0.50, 0.25)
c1 <- ifelse(W == 0 & A1 == 1, 0.50, 0.75)
first <- ifelse(followed1, 1 / (g1 * c1), 0)
followed2 <- !is.na(natural$A2) & natural$A2 == arms[, 2] &
  !is.na(natural$C2) & natural$C2 == 1
second <- ifelse(followed2, 1 / (0.5 * 0.75), 0)
ratios <- cbind(first, second)

control <- lmtp_control(
  .trim = 1, .learners_outcome_folds = 2, .learners_trt_folds = 2, .return_full_fits = TRUE
)

fit_with <- function(assignment, density_ratios = ratios) {
  lmtp_competing_tmle_with_folds(
    natural, shifted,
    trt = c("A1", "A2"),
    outcome = c("R1", "R2"),
    compete = c("D1", "D2"),
    baseline = "W",
    time_vary = list(NULL, "L2"),
    cens = c("C1", "C2"),
    outcome_type = "survival",
    fold_assignment = assignment,
    learners_outcome = "SL.mean",
    learners_trt = "SL.glm",
    density_ratios = density_ratios,
    control = control
  )
}

five <- rep(0:4, length.out = n)
fit <- fit_with(five)
summary <- ife::tidy(fit$estimate)
stopifnot(
  nrow(summary) == 1L,
  is.finite(summary$estimate),
  is.finite(summary$std.error),
  length(fit$initial) == n,
  !anyNA(fit$initial),
  identical(fit$fold_assignment, five)
)
cat("five-fold competing fit passed\n")

# The branch `fold_list` refuses, and the one the ordinary row runs on.
single <- rep(0L, n)
one <- fit_with(single)
stopifnot(
  is.finite(ife::tidy(one$estimate)$estimate),
  !anyNA(one$initial),
  identical(one$fold_assignment, single)
)
cat("one-fold competing fit passed\n")

# The screens have to reject, or they are decoration.  A ratio built from the wrong arm keeps
# the shape and moves the zero pattern; a dropped censoring factor keeps the zero pattern and
# moves the weight.  Each is a mistake the runner's lookup table can make.
rejects <- function(label, density_ratios) {
  failed <- inherits(try(fit_with(five, density_ratios), silent = TRUE), "try-error")
  if (!failed) stop(sprintf("the supplied-ratio screens accepted %s", label))
  cat(sprintf("rejected %s\n", label))
}
wrong_arm <- ratios
wrong_arm[, 1] <- ifelse(natural$A1 == 0 & natural$C1 == 1, 1 / (g1 * c1), 0)
rejects("a first-node ratio built from the opposite arm", wrong_arm)
rejects("a ratio with the censoring factor dropped", cbind(first * c1, second))

cat(sprintf("lmtp %s competing adapter smoke passed\n", packageVersion("lmtp")))
