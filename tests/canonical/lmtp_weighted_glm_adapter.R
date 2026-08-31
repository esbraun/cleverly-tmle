# SuperLearner GLM adapter for lmtp 1.5.4's sampling-weight boundary.
#
# lmtp passes task weights to targeting, plug-in averaging, and covariance, but its nuisance
# calls do not forward them as SuperLearner obsWeights. The weighted longitudinal comparison
# therefore carries the fixed design weight in an auxiliary predictor. This adapter consumes
# that column as the GLM loss weight and removes it from both predictor designs.

LMTP_AUX_WEIGHT <- "obs_weight_aux"

weighted_glm_design <- function(X, newX, weight_column = LMTP_AUX_WEIGHT) {
  X <- as.data.frame(X)
  newX <- as.data.frame(newX)
  if (!weight_column %in% names(X) || !weight_column %in% names(newX)) {
    stop(sprintf("weighted GLM requires auxiliary column %s", weight_column))
  }
  weights <- X[[weight_column]]
  if (!is.numeric(weights) || any(!is.finite(weights)) || any(weights <= 0)) {
    stop("auxiliary learner weights must be finite positive numbers")
  }
  list(
    X = X[, setdiff(names(X), weight_column), drop = FALSE],
    newX = newX[, setdiff(names(newX), weight_column), drop = FALSE],
    weights = as.numeric(weights)
  )
}

SL.weighted.glm <- function(Y, X, newX, family, obsWeights, id, ...) {
  design <- weighted_glm_design(X, newX)
  out <- SuperLearner::SL.glm(
    Y = Y,
    X = design$X,
    newX = design$newX,
    family = family,
    obsWeights = design$weights,
    id = id,
    ...
  )
  class(out$fit) <- c("SL.weighted.glm", class(out$fit))
  out
}

predict.SL.weighted.glm <- function(object, newdata, ...) {
  newdata <- as.data.frame(newdata)
  if (!LMTP_AUX_WEIGHT %in% names(newdata)) {
    stop(sprintf("weighted GLM prediction requires auxiliary column %s", LMTP_AUX_WEIGHT))
  }
  predictors <- newdata[, setdiff(names(newdata), LMTP_AUX_WEIGHT), drop = FALSE]
  stats::predict(object$object, newdata = predictors, type = "response")
}
