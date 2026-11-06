# The outcome learners of the long-horizon lmtp comparators (X59): the regressions cleverly fits,
# stated for lmtp's run_ensemble hook.
#
# cleverly fits the node-t regression of a plan on the units that follow the plan through t,
# with W and L2..Lt as numeric columns and, at each known stochastic (policy) node s <= t, the
# drop-first dummies of A_s (tests/studies/long_horizon_common.py, _design).  lmtp pools every
# observed row and puts A1..At in its design.  The two coincide for a plan that is stochastic at
# every node (mix, coin), where every observed row follows.  For every other plan they are two
# regressions of one target, so this file states cleverly's: it fits on the followers only and
# builds cleverly's design from the history lmtp hands it.  The followers are read from the
# supplied ratios: a unit follows through t exactly when its per-node ratios 1..t are all
# positive (tests/studies/long_horizon_reference.py, `ratios`).
#
# Learners, by the name a fit passes as learners_outcome:
#
# * SL.horizon.quasibinomial: S2.  The quasibinomial IRLS of
#   tests/studies/fractional_glm.py::QuasiBinomialGLM at every node, as cleverly's LTMLE fits it.
# * SL.horizon.sdr: S3.  The same IRLS at the horizon, where the target is the binary outcome,
#   and least squares before it, where the target is the SDR transform (cleverly's
#   pseudo_learner is LinearRegression).
# * SL.horizon.pooled: the declared fallback.  The quasibinomial IRLS on lmtp's own pooled
#   design, which is cleverly's regression for a plan that is stochastic at every node only.
#
# The node is read from the history: the regression of node t is handed A1..At and no later arm.
# horizon_learner_state() is set by the caller before each plan's fit, so the learner reads the
# plan's followers, its policy nodes and the declared mutation.  A fit that reads a state left by
# another plan is refused by the unit count check.
#
# Each learner returns an object with predict and summary methods.  lmtp 9996b04's tmle.R stores
# summary(fit, time, fold) where f04a2b4 stored the fit (X39-plan-review.md, finding 3), so the
# summary method is part of the learner's interface on the pinned commit.

horizon_state <- new.env(parent = emptyenv())

horizon_learner_state <- function(ratios, policy_nodes, m, horizon, mutation = 0L, folds = NULL) {
  # ratios: units x K per-node ratios of the plan about to be fitted, rows in unit order.
  # folds: each unit's outer fold, so a fit can count the followers its training folds hold.
  if (!is.matrix(ratios) || ncol(ratios) != horizon) {
    stop("the learner state needs one ratio column per node")
  }
  if (!is.null(folds) && length(folds) != nrow(ratios)) stop("one outer fold per unit is required")
  horizon_state$folds <- folds
  follow <- ratios > 0
  for (node in seq_len(horizon)[-1]) follow[, node] <- follow[, node] & follow[, node - 1L]
  horizon_state$followers <- follow
  horizon_state$policy <- as.integer(policy_nodes)
  horizon_state$m <- as.integer(m)
  horizon_state$horizon <- as.integer(horizon)
  horizon_state$mutation <- as.integer(mutation)
  invisible(TRUE)
}

horizon_policy_nodes <- function(mask, horizon) {
  # The bitmask the Python side writes (policy__<plan>): bit t - 1 marks node t.
  which(bitwAnd(as.integer(mask), bitwShiftL(1L, seq_len(horizon) - 1L)) > 0)
}

horizon_node_of <- function(data) {
  arms <- grep("^A[0-9]+$", names(data), value = TRUE)
  if (!length(arms)) stop("the regression's history holds no treatment column")
  max(as.integer(sub("^A", "", arms)))
}

horizon_number <- function(column) {
  value <- suppressWarnings(as.numeric(as.character(column)))
  value[is.na(value)] <- 0
  value
}

horizon_design <- function(data, node, stratified) {
  # cleverly's design of node `node`, without the intercept.
  m <- horizon_state$m
  columns <- list()
  # Declared mutation 2 (X59-plan.md section 4.1, M2): the node-K regression without W.
  if (!(horizon_state$mutation == 2L && node == horizon_state$horizon)) {
    columns$W <- horizon_number(data$W)
  }
  for (s in seq_len(node)[-1]) columns[[paste0("L", s)]] <- horizon_number(data[[paste0("L", s)]])
  arms <- if (stratified) horizon_state$policy[horizon_state$policy <= node] else seq_len(node)
  for (s in sort(arms)) {
    codes <- horizon_number(data[[paste0("A", s)]])
    for (k in seq_len(m - 1L)) columns[[sprintf("A%d_%d", s, k)]] <- as.numeric(codes == k)
  }
  matrix(unlist(columns, use.names = FALSE), nrow = nrow(data), dimnames = list(NULL, names(columns)))
}

horizon_lstsq <- function(design, response) {
  # numpy.linalg.lstsq(rcond=None): the minimum-norm least-squares solution, dropping singular
  # values below machine precision times the larger dimension of the design.
  decomposition <- svd(design)
  cutoff <- .Machine$double.eps * max(dim(design)) * max(decomposition$d)
  inverse <- ifelse(decomposition$d > cutoff, 1 / decomposition$d, 0)
  as.numeric(decomposition$v %*% (inverse * crossprod(decomposition$u, response)))
}

horizon_irls <- function(design, target, max_iter = 100L, tol = 1e-10) {
  # QuasiBinomialGLM.fit with unit weights, step for step: start at the mean's logit, solve each
  # working least-squares problem by minimum norm, stop when no coefficient moves by more than
  # tol.  A run that ends without converging is kept only when the deviance settled (R's
  # glm.control rule), as the Python learner keeps it.
  design <- cbind(1, design)
  average <- mean(target)
  coefficient <- numeric(ncol(design))
  coefficient[[1]] <- log(min(max(average, 1e-8), 1 - 1e-8) / min(max(1 - average, 1e-8), 1))
  deviance <- function(fitted) {
    xlogy <- function(x, y) ifelse(x == 0, 0, x * log(y))
    2 * sum(xlogy(target, target) - xlogy(target, fitted) +
      xlogy(1 - target, 1 - target) - xlogy(1 - target, 1 - fitted))
  }
  previous <- coefficient
  converged <- FALSE
  for (step in seq_len(max_iter)) {
    previous <- coefficient
    eta <- as.numeric(design %*% coefficient)
    fitted <- plogis(eta)
    variance <- pmax(fitted * (1 - fitted), 1e-10)
    working <- eta + (target - fitted) / variance
    root <- sqrt(variance)
    updated <- horizon_lstsq(design * root, working * root)
    if (max(abs(updated - coefficient)) <= tol) {
      coefficient <- updated
      converged <- TRUE
      break
    }
    coefficient <- updated
  }
  if (!converged) {
    before <- deviance(plogis(as.numeric(design %*% previous)))
    after <- deviance(plogis(as.numeric(design %*% coefficient)))
    if (abs(after - before) / (abs(after) + 0.1) >= 1e-8) {
      stop("quasibinomial IRLS did not converge")
    }
  }
  coefficient
}

horizon_unit_ids <- function(column) {
  # The callers' id is the integer row of the unit (..unit); anything else cannot index the state.
  if (!is.numeric(column) || anyNA(column) || any(column != round(column)) || any(column < 1)) {
    stop("the regression's unit id is not a positive integer row number")
  }
  as.integer(column)
}

horizon_log <- function(node, family, rows, units, coefficient) {
  # G0's follower-mapping check (x59_smoke.py g0) sets CLEVERLY_X59_FOLLOWER_LOG; runs do not.
  path <- study_env("CLEVERLY_X59_FOLLOWER_LOG")
  if (!nzchar(path)) return(invisible(FALSE))
  line <- data.frame(
    node = node, family = family, rows = rows, units = units,
    coefficients = paste(sprintf("%.17g", coefficient), collapse = ";")
  )
  write.table(
    line, path, sep = ",", row.names = FALSE, col.names = !file.exists(path),
    append = file.exists(path)
  )
  invisible(TRUE)
}

horizon_fit <- function(data, y, id, family, stratified) {
  node <- horizon_node_of(data)
  rows <- rep(TRUE, nrow(data))
  units <- horizon_unit_ids(data[[id]])
  if (stratified) {
    follow <- horizon_state$followers
    if (is.null(follow)) stop("the horizon learner was called before its state was set")
    if (max(units) > nrow(follow)) {
      stop("the regression's unit ids do not index the learner state; set it for this plan")
    }
    rows <- follow[units, node]
    # Every follower of the training folds must reach the fit: a unit lmtp dropped, or an id
    # that names another unit, changes this count.
    folds <- horizon_state$folds
    if (!is.null(folds)) {
      held <- unique(folds[unique(units)])
      expected <- sum(follow[folds %in% held, node])
      handed <- length(unique(units[rows]))
      if (handed != expected) {
        stop(sprintf(
          "node %d: the regression holds %d followers, its training folds %d", node, handed, expected
        ))
      }
    }
  }
  design <- horizon_design(data[rows, , drop = FALSE], node, stratified)
  target <- as.numeric(data[[y]][rows])
  if (nrow(design) <= ncol(design) + 1L) {
    stop(sprintf("node %d has %d followed rows for %d coefficients", node, nrow(design), ncol(design) + 1L))
  }
  coefficient <- if (family == "gaussian") {
    horizon_lstsq(cbind(1, design), target)
  } else {
    horizon_irls(design, target)
  }
  horizon_log(node, family, sum(rows), length(unique(units[rows])), coefficient)
  structure(
    list(coefficient = coefficient, node = node, family = family, stratified = stratified,
      columns = colnames(design), rows = sum(rows)),
    class = "lmtp_horizon_glm"
  )
}

predict.lmtp_horizon_glm <- function(object, newdata, tol = NULL, ...) {
  design <- horizon_design(newdata, object$node, object$stratified)
  if (!identical(colnames(design), object$columns)) stop("the prediction design differs from the fit's")
  eta <- as.numeric(cbind(1, design) %*% object$coefficient)
  if (object$family == "gaussian") return(eta)
  prediction <- plogis(eta)
  if (is.null(tol)) return(prediction)
  pmin(1 - tol, pmax(tol, prediction))
}

summary.lmtp_horizon_glm <- function(object, time = NA, fold = NA, ...) {
  # What lmtp 9996b04's tmle.R stores for each node and fold: no runner reads it.
  data.table::data.table(
    time = time, fold = fold, learner = paste("horizon", object$family), risk = NA_real_,
    coef = 1, rows = object$rows
  )
}

registerS3method("predict", "lmtp_horizon_glm", predict.lmtp_horizon_glm)
registerS3method("summary", "lmtp_horizon_glm", summary.lmtp_horizon_glm)

local({
  # Chained over whatever run_ensemble is bound now (lmtp_sdr_adapter.R installs its own
  # dispatch first), so every other learner name reaches lmtp's ensemble unchanged.
  namespace <- asNamespace("lmtp")
  previous <- get("run_ensemble", envir = namespace)
  dispatch <- function(data, y, learners, outcome_type, id, folds) {
    if (identical(learners, "SL.horizon.quasibinomial")) {
      return(horizon_fit(data, y, id, "quasibinomial", stratified = TRUE))
    }
    if (identical(learners, "SL.horizon.sdr")) {
      family <- if (horizon_node_of(data) == horizon_state$horizon) "quasibinomial" else "gaussian"
      return(horizon_fit(data, y, id, family, stratified = TRUE))
    }
    if (identical(learners, "SL.horizon.pooled")) {
      return(horizon_fit(data, y, id, "quasibinomial", stratified = FALSE))
    }
    previous(data, y, learners, outcome_type, id, folds)
  }
  unlockBinding("run_ensemble", namespace)
  assign("run_ensemble", dispatch, envir = namespace)
  lockBinding("run_ensemble", namespace)
})
