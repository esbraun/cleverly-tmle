suppressPackageStartupMessages({
  library(ctmle)
  library(glmnet)
  library(SuperLearner)
})
source("/fixture/study_harness.R")
options(digits = 17)

# R `ctmle` 0.1.2 at 18de559 for the registered study `canonical-ctmle-sequences`.  Line
# numbers cite the `ctmle` sources at that commit.
#
# binary_lasso:    ctmleGlmnet(ctmletype = 1) on the package's own selection folds, then R's
#                  type-1 final step from R's best_k and from the package's selected index.
# binary_learners: build_gn_seq on folds_R, then ctmleGeneral(ctmletype = 1) on folds_R.
#
# Beside each row the runner writes what the replay of the X16 plan, 3.4.2, reads: R's
# outcome and propensity coefficients, the accepted candidates, each candidate's estimate and
# loss, and the final steps.  A vector is one string of `%.17g` values joined by `;`.  Before
# it writes them, the runner asserts that `plogis(cbind(1, W) %*% coef)` reproduces each
# array R used to 1e-12, so the replay can rebuild R's arrays from the coefficients.
#
# The declared adapter setting: the default glmnet path runs to lambda_max * 1e-4 instead of
# stopping early on the deviance (X16 plan, 3.5 fact 3).  `predict(s = 0)` clamps the penalty 0
# to the minimum of the path it reads (quirk 4), and R reads paths on two row sets:
#   - `construct_candidates_glmnet` scores each try with a path fitted on the training rows
#     only (functions_glmnet.R:132-135).  In the full-sample construction those are all rows.
#     Inside cross-validation fold v they are the rows outside fold v.
#   - the accepted refit (functions_glmnet.R:205-208) and `evaluate_candidates_glmnet`
#     (functions_glmnet.R:22-25, quirk 3) fit the path on all rows.
# The runner records the full-sample minimum as `path_min_lambda` and the minimum of the path
# on the rows outside fold v as `path_min_lambda_fold_<v>`, for v = 1 to V.
glmnet.control(fdev = 0, devmax = 1)

args <- study_arguments("usage: run_ctmle_sequences.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(args$samples), stringsAsFactors = FALSE)
truths <- read.csv(args$truths, stringsAsFactors = FALSE)
truth_key <- paste(truths$scenario, truths$replicate, sep = "|")

V <- 5L
GBOUND <- 0.025
# R `ctmle`'s default `alpha`.  R bounds the initial Q and each accepted update into
# `c(alpha, 1 - alpha)` (ctmle_general.R:161-162; functions_general.R:218), but not a
# candidate's `Qstar` (functions_general.R:185-201).  The package bounds every targeted Q.
ALPHA <- 0.995
QBOUNDS <- c(ALPHA, 1 - ALPHA)
# Quirk 5: `evaluate_candidates_general` matches the next accepted column by squared distance
# below 1e-5 (functions_general.R:209), not by index.  Two candidate columns closer than that
# could be matched to each other, so the run stops on any such pair.
SEPARATION <- 1e-5

# The quirk-7 error of ctmleGlmnet type 1 at a selected last penalty (3.5, fact 6).
QUIRK_7_TEXT <- "length of 'dimnames' [1] not equal to array extent"
# The largest gap between an array R used and its rebuild from R's coefficients.
RECONSTRUCTION_TOLERANCE <- 1e-12
COVARIATES <- c("W1", "W2", "W3", "W4")
# The 1-based entries of the binary_learners grid the declared candidates use (3.4.1, N2).
LEARNER_ENTRIES <- c(2L, 4L, 16L, 19L)

# One grid per scenario, 20 penalties each, written by design.py.
grid <- read.csv("/fixture/ctmle_sequences/lambdas.csv", stringsAsFactors = FALSE)
if (!identical(names(grid), c("scenario", "entry", "lambda"))) {
  stop("lambdas.csv must hold the columns scenario, entry, lambda")
}
STUDY_LAMBDAS <- list()
for (scenario in c("binary_lasso", "binary_learners")) {
  rows <- grid[grid$scenario == scenario, ]
  if (nrow(rows) != 20L || !identical(as.integer(rows$entry), 1:20) ||
      any(diff(rows$lambda) >= 0) || any(rows$lambda <= 0)) {
    stop(sprintf(
      "lambdas.csv must hold entries 1 to 20 of positive, strictly decreasing penalties for %s",
      scenario
    ))
  }
  STUDY_LAMBDAS[[scenario]] <- rows$lambda
}
if (!setequal(unique(grid$scenario), names(STUDY_LAMBDAS))) {
  stop("lambdas.csv must hold exactly the grids of binary_lasso and binary_learners")
}
LEARNER_LAMBDAS <- STUDY_LAMBDAS[["binary_learners"]][LEARNER_ENTRIES]

# R's own folds for binary_learners, drawn by make_r_folds.R from the seed this runner sets
# before each replicate.  `row` is 0-based, in sample order.
committed_folds <- read.csv(gzfile("/fixture/ctmle_sequences/r_folds.csv.gz"))
if (!identical(names(committed_folds), c("replicate", "row", "fold"))) {
  stop("r_folds.csv.gz must hold the columns replicate, row, fold")
}
committed_folds <- committed_folds[order(committed_folds$replicate, committed_folds$row), ]
fold_lookup <- split(as.integer(committed_folds$fold), committed_folds$replicate)
rm(committed_folds)

# The seeds the harness sets before each replicate.  binary_learners sets the same seed again
# before ctmleGeneral, so cv_general's redraw at functions_general.R:287 is folds_R.
run_seeds <- study_seeds(study_env("CLEVERLY_R_SEEDS"))

# ---- the SuperLearner wrappers of README.md, lines 303-337, at LEARNER_LAMBDAS -------------
lasso_lambdas <- LEARNER_LAMBDAS

SL.glmnet_new <- function(Y, X, newX, family, obsWeights, id, alpha = 1,
                           nlambda = 100, lambda = 0,...){
      # browser()
      if (!is.matrix(X)) {
            X <- model.matrix(~-1 + ., X)
            newX <- model.matrix(~-1 + ., newX)
      }
      fit <- glmnet::glmnet(x = X, y = Y,
                            lambda = lambda,
                            family = family$family, alpha = alpha)
      pred <- predict(fit, newx = newX, type = "response")
      fit <- list(object = fit)
      class(fit) <- "SL.glmnet"
      out <- list(pred = pred, fit = fit)
      return(out)
}

SL.cv1lasso <- function (... , alpha = 1, lambda = lasso_lambdas[1]){
      SL.glmnet_new(... , alpha = alpha, lambda = lambda)
}

SL.cv2lasso <- function (... , alpha = 1, lambda = lasso_lambdas[2]){
      SL.glmnet_new(... , alpha = alpha, lambda = lambda)
}

SL.cv3lasso <- function (... , alpha = 1, lambda = lasso_lambdas[3]){
      SL.glmnet_new(... , alpha = alpha, lambda = lambda)
}

SL.cv4lasso <- function (... , alpha = 1, lambda = lasso_lambdas[4]){
      SL.glmnet_new(... , alpha = alpha, lambda = lambda)
}

SL.library = c('SL.cv1lasso', 'SL.cv2lasso', 'SL.cv3lasso', 'SL.cv4lasso', 'SL.glm')

# ---- the type-1 adapter and the final step from a declared start --------------------------
# `rebuild_type1` and `final_type1`, shared with the witness's rebuild check.
source("/fixture/ctmle_sequences/rebuild_type1.R")

# ---- shared pieces --------------------------------------------------------------------------

# The package's selection split, carried in the `selection_fold` column, asserted to be a
# partition that holds both arms in every fold, as run_ctmle.R does.
selection_folds <- function(frame) {
  assignment <- frame$selection_fold
  if (anyNA(assignment)) stop("the selection fold column has missing values")
  codes <- sort(unique(assignment))
  if (!identical(as.integer(codes), seq_len(V) - 1L)) {
    stop(sprintf("selection folds must be 0..%d; got %s", V - 1L, paste(codes, collapse = ",")))
  }
  folds <- lapply(codes, function(k) which(assignment == k))
  if (sum(lengths(folds)) != nrow(frame) || anyDuplicated(unlist(folds)) != 0L) {
    stop("the selection folds do not partition the sample")
  }
  for (index in seq_along(folds)) {
    if (length(unique(frame$A[folds[[index]]])) != 2L) {
      stop(sprintf("selection fold %s does not hold both treatment arms", codes[[index]]))
    }
  }
  folds
}

# Quirk 5, asserted on one block of candidate columns.
check_separation <- function(columns, block, label) {
  columns <- as.matrix(columns)
  if (ncol(columns) < 2L) return(invisible(NULL))
  for (i in seq_len(ncol(columns) - 1L)) {
    for (j in seq(i + 1L, ncol(columns))) {
      distance <- sum((columns[, i] - columns[, j])^2)
      if (!is.finite(distance) || distance < SEPARATION) {
        stop(sprintf(
          "%s: %s candidate columns %d and %d lie at squared distance %.3g, below %g (quirk 5)",
          label, block, i, j, distance, SEPARATION
        ))
      }
    }
  }
  invisible(NULL)
}

# The fold of each row, 1 to V, of a `by(sample(1:n, n), rep(1:V, length = n), list)` draw.
fold_labels <- function(folds, n) {
  labels <- integer(n)
  for (k in seq_along(folds)) {
    labels[as.integer(unlist(folds[[k]], use.names = FALSE))] <- k
  }
  if (any(labels == 0L)) stop("the drawn folds do not cover every row")
  labels
}

# ---- the two fits ---------------------------------------------------------------------------

# R's `bound` of the initial Q and of each accepted update: values above `max(bounds)` move
# to it, then values below `min(bounds)`.
bound_q <- function(x, bounds) {
  x[x > max(bounds)] <- max(bounds)
  x[x < min(bounds)] <- min(bounds)
  x
}

# The binary_learners candidate estimates and losses that the replay compares (review 2, B1).
# R's evaluation (functions_general.R:188-196) reads each candidate's `Qstar` unbounded.  The
# package bounds it into `[1 - alpha, alpha]`.  So the runner reads R's own `Qstars` and bounds
# them the way R bounds its accepted updates, and both sides compute the same quantity.  Where
# the bound leaves a candidate unchanged, the result must reproduce R's own `est` and
# `likelihood` (the check below).
bounded_candidates <- function(results, y, label) {
  qstars <- results$Qstars
  if (length(qstars) != length(results$est)) {
    stop(sprintf("%s: results.all holds %d Qstars for %d candidates", label,
                 length(qstars), length(results$est)))
  }
  est <- loss <- numeric(length(qstars))
  for (i in seq_along(qstars)) {
    raw <- as.matrix(qstars[[i]])
    if (!all(c("QAW", "Q0W", "Q1W") %in% colnames(raw)) || nrow(raw) != length(y)) {
      stop(sprintf("%s: Qstars[[%d]] is not an n x (QAW, Q0W, Q1W) matrix", label, i))
    }
    q <- bound_q(raw, QBOUNDS)
    est[i] <- mean(q[, "Q1W"]) - mean(q[, "Q0W"])
    loss[i] <- -sum(y * log(q[, "QAW"]) + (1 - y) * log(1 - q[, "QAW"]))
    if (identical(q, raw)) {
      gaps <- c(abs(est[i] - results$est[i]),
                abs(loss[i] - results$likelihood[i]) / abs(results$likelihood[i]))
      if (!all(is.finite(gaps)) || max(gaps) > RECONSTRUCTION_TOLERANCE) {
        stop(sprintf("%s: candidate %d recomputed from its Qstar misses R by %.3g", label, i,
                     max(gaps)))
      }
    }
  }
  list(est = est, loss = loss)
}

# One vector column: the values printed with `%.17g` and joined by `;`.
vector_text <- function(values) {
  if (is.null(values)) return(NA_character_)
  paste(sprintf("%.17g", as.numeric(values)), collapse = ";")
}

# Fatal, outside every stage: a broken reconstruction fails the run (3.4.2).
check_reconstruction <- function(used, coef, x, what, label) {
  rebuilt <- plogis(cbind(1, x) %*% as.matrix(coef))
  used <- as.matrix(used)
  if (!identical(dim(rebuilt), dim(used))) {
    stop(sprintf("%s: the rebuilt %s has dimensions %s, not %s", label, what,
                 paste(dim(rebuilt), collapse = "x"), paste(dim(used), collapse = "x")))
  }
  gap <- max(abs(rebuilt - used))
  if (!is.finite(gap) || gap > RECONSTRUCTION_TOLERANCE) {
    stop(sprintf("%s: the %s rebuilt from its coefficients misses by %.3g, above %g",
                 label, what, gap, RECONSTRUCTION_TOLERANCE))
  }
  invisible(NULL)
}

# R's type-1 final step from `start_k`, with its explicit-lambda array checked against its
# coefficients.  NULL when `start_k` is the last index, where the step does not exist.
final_columns <- function(arguments, start_k, label) {
  if (is.na(start_k) || start_k >= length(arguments$lambdas)) return(NULL)
  final <- study_stage("final_type1", final_type1(
    Y = arguments$Y, A = arguments$A, W = arguments$W, Q = arguments$Q,
    lambdas = arguments$lambdas, start_k = start_k, gbound = arguments$gbound,
    like_type = arguments$like_type
  ))
  check_reconstruction(final$response, final$coef, final$lassox,
                       sprintf("final-step array from start %d", start_k), label)
  if (length(final$loss) != length(arguments$lambdas) - start_k) {
    stop(sprintf("%s: the final step from %d scored %d options", label, start_k,
                 length(final$loss)))
  }
  final
}

final_fields <- function(final, prefix) {
  out <- list(
    start = NA_integer_, coef = NA_character_, loss = NA_character_,
    index = NA_integer_, est = NA_real_
  )
  if (!is.null(final)) {
    out <- list(
      start = final$start, coef = vector_text(final$coef), loss = vector_text(final$loss),
      index = final$index, est = final$est
    )
  }
  names(out) <- paste(prefix, names(out), sep = "_")
  out
}

fit_lasso <- function(frame, W, Q, label) {
  folds <- selection_folds(frame)
  lambdas <- c(STUDY_LAMBDAS[["binary_lasso"]], 0)
  K <- length(lambdas)
  # The default path of the fit's own construction (functions_glmnet.R:130-134): once on all
  # rows, and once on the training rows of each cross-validation fold.
  lassox <- model.matrix(~ W1 + W2 + W3 + W4, W)[, -1]
  default_path <- function(rows) {
    study_stage("default glmnet path", glmnet(
      x = lassox[rows, , drop = FALSE], y = as.factor(frame$A[rows]), family = "binomial"
    ))
  }
  full_path <- default_path(seq_len(nrow(frame)))
  path_min <- min(full_path$lambda)
  fold_path_min <- vapply(folds, function(held) min(default_path(-held)$lambda), numeric(1))
  # The arrays the full-sample construction and evaluation read (functions_glmnet.R:22-25 and
  # 132-135), and their coefficients, candidate-major.
  g_coef <- as.matrix(coef(full_path, s = lambdas))
  check_reconstruction(
    predict(full_path, newx = lassox, s = lambdas, type = "response"), g_coef, lassox,
    "lasso candidate array", label
  )
  arguments <- list(
    Y = frame$Y,
    A = frame$A,
    W = W,
    Q = Q,
    lambdas = lambdas,
    ctmletype = 1,
    family = "binomial",
    like_type = "loglik",
    gbound = GBOUND,
    PEN = FALSE,
    V = V,
    folds = folds,
    detailed = TRUE
  )
  # A ctmleGlmnet error comes back as a condition, so the adapter below can read it.  When
  # the harness records failures, the stage has already made it a study_fitting_failure that
  # keeps the original message.  A fatal error passes through.
  fit <- tryCatch(
    study_stage("ctmleGlmnet", do.call(ctmleGlmnet, arguments)),
    error = function(condition) {
      if (inherits(condition, "study_fatal_error")) stop(condition)
      condition
    }
  )
  # S1: the rebuild runs on every replicate.
  rebuilt <- study_stage("rebuild_type1", rebuild_type1(
    Y = arguments$Y, A = arguments$A, W = arguments$W, Q = arguments$Q,
    lambdas = arguments$lambdas, ctmletype = arguments$ctmletype, family = arguments$family,
    like_type = arguments$like_type, gbound = arguments$gbound, PEN = arguments$PEN,
    V = arguments$V, folds = arguments$folds, detailed = arguments$detailed
  ))
  rebuild_best_k <- as.integer(rebuilt$cv.res$best_k)
  if (inherits(fit, "error")) {
    # Only the quirk-7 failure is adapted: the message must hold the quirk-7 text and the
    # rebuilt selection must be the last penalty.  Otherwise the original ctmleGlmnet error
    # is recorded as this group's failure.
    matched <- grepl(QUIRK_7_TEXT, conditionMessage(fit), fixed = TRUE)
    if (!matched || !identical(rebuild_best_k, K)) stop(fit)
    adapter_error_matched <- as.integer(matched)
    candidates <- rebuilt$candidates
    results <- candidates$results.all
    best_k <- rebuild_best_k
    estimate <- as.numeric(results$est[best_k])
    std_error <- sqrt(as.numeric(results$varDstar_origscale[best_k]) / rebuilt$n)
    adapter_used <- 1L
    rebuild_agrees <- NA_integer_
  } else {
    best_k <- as.integer(fit$best_k)
    candidates <- fit$candidates
    if (!identical(rebuild_best_k, best_k) ||
        !identical(rebuilt$candidates$results.all$est, candidates$results.all$est) ||
        !identical(rebuilt$candidates$candidates$lambdas, candidates$candidates$lambdas)) {
      stop(sprintf(
        "%s: rebuild_type1 disagrees with ctmleGlmnet (best_k %d against %d)",
        label, rebuild_best_k, best_k
      ))
    }
    estimate <- as.numeric(fit$est)
    std_error <- sqrt(as.numeric(fit$var.psi))
    adapter_used <- 0L
    adapter_error_matched <- 0L
    rebuild_agrees <- 1L
  }
  # R's own final step from best_k, and the same step from the package's selected index.
  final_r <- final_columns(arguments, best_k, label)
  # A check of the verbatim copy: from best_k it reproduces ctmleGlmnet's own final step.
  if (!is.null(final_r)) {
    gaps <- abs(c(final_r$est - estimate, final_r$var.psi - as.numeric(fit$var.psi)))
    if (!all(is.finite(gaps)) || max(gaps) > RECONSTRUCTION_TOLERANCE) {
      stop(sprintf("%s: final_type1 from best_k misses ctmleGlmnet by %.3g", label, max(gaps)))
    }
  }
  package_selected <- as.integer(frame$package_selected[[1]])
  if (is.na(package_selected) || package_selected < 1L || package_selected > K) {
    stop(sprintf("%s: the package_selected column holds %s", label, package_selected))
  }
  final_pkg <- final_columns(arguments, package_selected, label)
  results <- candidates$results.all
  c(
    list(
      estimate = estimate,
      std_error = std_error,
      best_k = best_k,
      n_candidates = length(candidates$candidates$lambdas),
      adapter_used = adapter_used,
      path_min_lambda = path_min,
      fold_path_min_lambda = fold_path_min,
      adapter_error_matched = adapter_error_matched,
      rebuild_best_k = rebuild_best_k,
      rebuild_agrees = rebuild_agrees,
      g_coef = vector_text(g_coef),
      # R marks acceptance at full precision, by exact equality with the grid.
      r_accepted = vector_text(which(lambdas %in% candidates$candidates$lambdas)),
      cand_est = vector_text(results$est),
      cand_loss = vector_text(results$likelihood)
    ),
    final_fields(final_r, "final_r"),
    final_fields(final_pkg, "final_pkg")
  )
}

fit_learners <- function(frame, W, Q, seed, folds, label) {
  n <- nrow(frame)
  gn_seq <- study_stage("build_gn_seq", build_gn_seq(
    A = frame$A, W = W, SL.library = SL.library, folds = folds, verbose = FALSE
  ))
  full <- as.matrix(gn_seq$gn_candidates)
  cv <- as.matrix(gn_seq$gn_candidates_cv)
  if (ncol(full) != 5L || ncol(cv) != 5L) stop(sprintf("%s: build_gn_seq gave no 5 columns", label))
  check_separation(full, "full", label)
  check_separation(cv, "cv", label)
  # The full-sample fit of each library entry, in library order: four glmnet fits at one
  # penalty each, then SL.glm.
  fits <- gn_seq$details$fitLibrary
  if (length(fits) != 5L) stop(sprintf("%s: build_gn_seq kept %d fits, not 5", label, length(fits)))
  g_coef <- vapply(fits, function(fit) as.numeric(as.matrix(coef(fit$object))), numeric(5))
  check_reconstruction(full, g_coef, as.matrix(W), "learners candidate array", label)
  # The same seed again, so cv_general's redraw at functions_general.R:287 is folds_R.
  set.seed(seed)
  fit <- study_stage("ctmleGeneral", ctmleGeneral(
    Y = frame$Y,
    A = frame$A,
    W = W,
    Q = Q,
    ctmletype = 1,
    gn_candidates = full,
    gn_candidates_cv = cv,
    family = "binomial",
    like_type = "loglik",
    gbound = GBOUND,
    PEN = FALSE,
    V = V,
    folds = folds,
    detailed = TRUE
  ))
  best_k <- as.integer(fit$best_k)
  # The accepted candidates, matched by column, as the witness fixture matches them.
  same <- function(a, b) isTRUE(all(as.vector(a) == as.vector(b)))
  accepted <- apply(as.matrix(fit$candidates$candidates$gns), 2, function(column) {
    hits <- which(apply(full, 2, function(candidate) same(candidate, column)))
    if (length(hits) != 1L) stop(sprintf("%s: an accepted column matches no single candidate", label))
    hits
  })
  results <- fit$candidates$results.all
  bounded <- bounded_candidates(results, frame$Y, label)
  # R's plain D* variance at the selected candidate, never `var.psi`, which carries the
  # first-covariate ICg term (quirk 6).
  variance <- as.numeric(results$varDstar_origscale[best_k]) / n
  c(
    list(
      estimate = as.numeric(fit$est),
      std_error = sqrt(variance),
      best_k = best_k,
      n_candidates = NCOL(fit$candidates$candidates$gns),
      adapter_used = 0L,
      path_min_lambda = NA_real_,
      fold_path_min_lambda = rep(NA_real_, V),
      adapter_error_matched = NA_integer_,
      rebuild_best_k = NA_integer_,
      rebuild_agrees = NA_integer_,
      g_coef = vector_text(g_coef),
      r_accepted = vector_text(accepted),
      cand_est = vector_text(bounded$est),
      cand_loss = vector_text(bounded$loss)
    ),
    final_fields(NULL, "final_r"),
    final_fields(NULL, "final_pkg")
  )
}

fit_one <- function(frame) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  label <- sprintf("%s replicate %s", scenario, replicate)
  n <- nrow(frame)
  if (scenario == "binary_learners") {
    # First, before anything else reads the generator: the harness has just set this
    # replicate's seed, and make_r_folds.R drew folds_R right after the same call.
    seed <- study_seed_for(frame, run_seeds)
    if (is.null(seed)) stop(sprintf("%s: binary_learners needs the replication seeds", label))
    folds <- by(sample(1:n, n), rep(1:V, length = n), list)
    drawn <- fold_labels(folds, n)
    committed <- fold_lookup[[as.character(replicate)]]
    if (is.null(committed) || !identical(committed, drawn)) {
      stop(sprintf("%s: the folds drawn from the replication seed are not r_folds.csv.gz", label))
    }
    if (!identical(as.integer(frame$selection_fold) + 1L, drawn)) {
      stop(sprintf("%s: the package scored its candidates on another split", label))
    }
  } else if (scenario != "binary_lasso") {
    stop(sprintf("unknown scenario %s", scenario))
  }
  truth_row <- match(paste(scenario, replicate, sep = "|"), truth_key)
  if (is.na(truth_row)) stop(sprintf("no truth for %s replicate %s", scenario, replicate))
  truth <- truths$truth_ate[[truth_row]]
  W <- frame[COVARIATES]
  q_fit <- study_stage(
    "initial glm", glm(Y ~ A + W1 + W2 + W3 + W4, data = frame, family = binomial())
  )
  q0 <- predict(q_fit, transform(frame, A = 0), type = "response")
  q1 <- predict(q_fit, transform(frame, A = 1), type = "response")
  # The outcome coefficients read the design (A, W1, ..., W4), as the package's outcome
  # learner does.
  q_coef <- coef(q_fit)
  if (!identical(names(q_coef), c("(Intercept)", "A", COVARIATES))) {
    stop(sprintf("%s: the outcome glm has coefficients %s", label,
                 paste(names(q_coef), collapse = ", ")))
  }
  w_matrix <- as.matrix(W)
  check_reconstruction(fitted(q_fit), q_coef, cbind(frame$A, w_matrix), "outcome fit", label)
  check_reconstruction(q0, q_coef, cbind(0, w_matrix), "outcome fit at A = 0", label)
  check_reconstruction(q1, q_coef, cbind(1, w_matrix), "outcome fit at A = 1", label)
  Q <- cbind(q0, q1)
  result <- if (scenario == "binary_lasso") {
    fit_lasso(frame, W, Q, label)
  } else {
    fit_learners(frame, W, Q, seed, folds, label)
  }
  if (!is.finite(result$estimate) || !is.finite(result$std_error)) {
    stop(sprintf("%s: a non-finite estimate or standard error", label))
  }
  critical <- qnorm(0.975)
  low <- result$estimate - critical * result$std_error
  high <- result$estimate + critical * result$std_error
  # The five path_min_lambda_fold_<v> columns below assume V = 5.
  if (length(result$fold_path_min_lambda) != V || V != 5L) {
    stop(sprintf("%s: %d fold path minima, not 5", label, length(result$fold_path_min_lambda)))
  }
  data.frame(
    implementation = "r-ctmle",
    scenario = scenario,
    replicate = replicate,
    n = n,
    estimand = "ate",
    truth = truth,
    estimate = result$estimate,
    inference_estimate = result$estimate,
    std_error = result$std_error,
    ci_lower = low,
    ci_upper = high,
    inference_scale = "identity",
    covered = as.integer(low <= truth && truth <= high),
    initial_estimate = mean(q1 - q0),
    best_k = result$best_k,
    n_candidates = result$n_candidates,
    adapter_used = result$adapter_used,
    path_min_lambda = result$path_min_lambda,
    path_min_lambda_fold_1 = result$fold_path_min_lambda[[1]],
    path_min_lambda_fold_2 = result$fold_path_min_lambda[[2]],
    path_min_lambda_fold_3 = result$fold_path_min_lambda[[3]],
    path_min_lambda_fold_4 = result$fold_path_min_lambda[[4]],
    path_min_lambda_fold_5 = result$fold_path_min_lambda[[5]],
    adapter_error_matched = result$adapter_error_matched,
    rebuild_best_k = result$rebuild_best_k,
    rebuild_agrees = result$rebuild_agrees,
    q_coef = vector_text(q_coef),
    g_coef = result$g_coef,
    r_accepted = result$r_accepted,
    cand_est = result$cand_est,
    cand_loss = result$cand_loss,
    final_r_start = result$final_r_start,
    final_r_coef = result$final_r_coef,
    final_r_loss = result$final_r_loss,
    final_r_index = result$final_r_index,
    final_r_est = result$final_r_est,
    final_pkg_start = result$final_pkg_start,
    final_pkg_coef = result$final_pkg_coef,
    final_pkg_loss = result$final_pkg_loss,
    final_pkg_index = result$final_pkg_index,
    final_pkg_est = result$final_pkg_est,
    stringsAsFactors = FALSE
  )
}

groups <- split(samples, interaction(samples$scenario, samples$replicate, drop = TRUE))
cores <- study_cores(groups)
# Forced here, in the parent: with `mc.preschedule = TRUE`, `mclapply` would force it in each
# child, and `study_collect` would then find no registered groups to retry or assemble.
fit_group <- study_fitter(groups, fit_one)
results <- parallel::mclapply(
  seq_along(groups),
  fit_group,
  mc.cores = cores,
  mc.preschedule = TRUE
)
study_collect(
  results,
  expected = length(groups),
  output = args$output,
  versions = c(
    study_version("ctmle"),
    study_version("glmnet"),
    study_version("SuperLearner"),
    paste("commit", Sys.getenv("CTMLE_COMMIT"))
  ),
  key = c("scenario", "replicate")
)
