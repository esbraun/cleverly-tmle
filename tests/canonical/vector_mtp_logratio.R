# The ordered law's induced density ratio and the stacked classifier on its logarithm, for
# vector-mtp's unsupplied pair.
#
# ordered_ratio transcribes tests/studies/vector_mtp_common.py:ordered_ratio for the plans
# "ordered" and "full" (and returns 1 under "natural course").  vector_mtp_parity.R evaluates it
# on the grid that tests/canonical/vector_mtp_parity.py writes, and the Python side compares it
# with ordered_ratio at rtol 1e-12, before the declared run.
#
# SL.logratio.<plan> is the R twin of tests/studies/vector_mtp_common.py:OracleLogOdds: an
# unpenalized binomial glm of the stack label on one feature, log(max(r(a, h), 1e-13)), which it
# computes from the named columns F, D1, D2, W1 and W2 of each row it is handed.  lmtp's
# stack_data replaces the node's treatment columns in the shifted half and copies the history,
# so each copy of a unit reads its own treatment value.  A data column holding the feature would
# carry the natural row's value into the shifted copy.  One plan's feature per fit: the
# comparator fits (natural course, ordered) with SL.logratio.ordered and (natural course, full)
# with SL.logratio.full, as the package fits each pair with its own learner.

VECTOR_MTP <- list(
  lower = 0, upper = 6, cap = 5.5, step = 0.5, factor = 1.25, knee = 3, floor = 1e-13
)
VECTOR_MTP$scale_held <- VECTOR_MTP$cap / VECTOR_MTP$factor
VECTOR_MTP$shift_held <- VECTOR_MTP$cap - VECTOR_MTP$step

vector_mtp_truncated_pdf <- function(a, mean) {
  mass <- pnorm(VECTOR_MTP$upper - mean) - pnorm(VECTOR_MTP$lower - mean)
  ifelse(a >= VECTOR_MTP$lower & a <= VECTOR_MTP$upper, dnorm(a - mean) / mass, 0)
}

vector_mtp_ordered_density <- function(c, x1, x2, w1, w2) {
  # g(c, x1, x2 | w): the flag, D1 given the flag, and D2 given D1 and the flag.
  flag <- plogis(0.1 + 0.6 * w1 - 0.4 * w2)
  p <- if (c == 1) flag else 1 - flag
  p *
    vector_mtp_truncated_pdf(x1, 2.4 + 0.6 * c + 0.4 * w1 + 0.3 * w2) *
    vector_mtp_truncated_pdf(x2, 1 + 0.4 * x1 + 0.5 * c - 0.3 * w1)
}

ordered_ratio <- function(plan, f, b1, b2, w1, w2) {
  if (plan == "natural course") return(rep(1, length(b1)))
  if (!(plan %in% c("ordered", "full"))) stop(sprintf("no transcribed ratio for plan %s", plan))
  v <- VECTOR_MTP
  observed <- ifelse(
    f == 1,
    vector_mtp_ordered_density(1, b1, b2, w1, w2),
    vector_mtp_ordered_density(0, b1, b2, w1, w2)
  )
  reached <- rep(0, length(b1))
  for (c in c(0, 1)) {
    scaled <- plan == "ordered" || (plan == "full" && c == 1)
    pieces1 <- if (scaled) {
      list(
        list(beta = b1 / v$factor, jacobian = 1 / v$factor, valid = b1 <= v$cap),
        list(beta = b1, jacobian = 1, valid = b1 > v$scale_held)
      )
    } else {
      list(list(beta = b1, jacobian = 1, valid = rep(TRUE, length(b1))))
    }
    for (piece1 in pieces1) {
      beta1 <- piece1$beta
      image <- if (plan == "full") ifelse(beta1 >= v$knee, 1, c) else rep(c, length(b1))
      hit <- piece1$valid & (image == f)
      small <- beta1 <= v$knee
      pieces2 <- list(
        list(beta = b2 - v$step, valid = small & (b2 <= v$cap)),
        list(beta = b2, valid = small & (b2 > v$shift_held)),
        list(beta = b2, valid = !small)
      )
      for (piece2 in pieces2) {
        term <- vector_mtp_ordered_density(c, beta1, piece2$beta, w1, w2) * piece1$jacobian
        reached <- reached + ifelse(hit & piece2$valid, term, 0)
      }
    }
  }
  safe <- ifelse(observed > 0, observed, 1)
  ifelse(observed > 0, reached / safe, 0)
}

ordered_log_ratio <- function(plan, frame) {
  # The feature of each row, from its own named columns.
  ratio <- ordered_ratio(
    plan, frame[["F"]], frame[["D1"]], frame[["D2"]], frame[["W1"]], frame[["W2"]]
  )
  log(pmax(ratio, VECTOR_MTP$floor))
}

logratio_learner <- function(plan) {
  force(plan)
  feature <- function(frame) data.frame(logratio = ordered_log_ratio(plan, frame))
  function(Y, X, newX, family, obsWeights, ...) {
    fit <- stats::glm(
      Y ~ logratio, data = feature(X), family = family, weights = obsWeights,
      control = stats::glm.control(epsilon = 1e-10, maxit = 100)
    )
    if (!fit$converged) stop(sprintf("the %s log-ratio classifier did not converge", plan))
    pred <- stats::predict(fit, newdata = feature(newX), type = "response")
    list(pred = pred, fit = structure(list(glm = fit, feature = feature), class = "SL.logratio"))
  }
}

predict.SL.logratio <- function(object, newdata, ...) {
  stats::predict(object$glm, newdata = object$feature(newdata), type = "response")
}

SL.logratio.ordered <- logratio_learner("ordered")
SL.logratio.full <- logratio_learner("full")
