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
# binary_lasso:    ctmleGlmnet(ctmletype = 1) on the package's own selection folds.
# binary_learners: build_gn_seq on folds_R, then ctmleGeneral(ctmletype = 1) on folds_R.
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
# Quirk 5: `evaluate_candidates_general` matches the next accepted column by squared distance
# below 1e-5 (functions_general.R:209), not by index.  Two candidate columns closer than that
# could be matched to each other, so the run stops on any such pair.
SEPARATION <- 1e-5

grid <- read.csv("/fixture/ctmle_sequences/lambdas.csv")
if (!identical(names(grid), c("entry", "lambda")) || nrow(grid) != 20L ||
    !identical(as.integer(grid$entry), 1:20) || any(diff(grid$lambda) >= 0) ||
    any(grid$lambda <= 0)) {
  stop("lambdas.csv must hold entries 1 to 20 of positive, strictly decreasing penalties")
}
STUDY_LAMBDAS <- grid$lambda
LEARNER_LAMBDAS <- STUDY_LAMBDAS[c(4L, 8L, 12L, 16L)]

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

# ---- the type-1 adapter ---------------------------------------------------------------------
# ctmleGlmnet type 1 has no branch for a selected last penalty: `lambdas_remain` is empty at
# ctmle_glmnet.R:270, and the final step fails (quirk 7).  This function takes ctmleGlmnet's
# formals (lines 86-95), runs its preamble (lines 97-112) verbatim, and rebuilds R's own
# construction and cross-validation (lines 213-230).  It returns the pieces ctmleGlmnet would
# have used, with `varDstar_origscale` as lines 346-347 compute it.
rebuild_type1 <- function(Y, A, W, Wg = W,
                          Q, lambdas = NULL,
                          ctmletype,
                          V=5, folds = NULL,
                          alpha=.995, family= "gaussian",
                          gbound=.025, like_type = "RSS",
                          fluctuation="logistic", verbose=FALSE,
                          detailed=FALSE, PEN=FALSE,
                          g1W = NULL, g1WPrev = NULL,
                          stopFactor=10^6) {
      # ctmle_glmnet.R:97-112, verbatim.
      n <- length(Y)
      maptoYstar <- fluctuation=="logistic"
      Qbounds <- range(Y)
      QAW <- (1 - A) * Q[, 1] + A * Q[, 2]
      Q <- cbind(QAW, Q0W = Q[, 1], Q1W = Q[, 2])
      Q <- bound(Q, Qbounds)
      Ystar <- bound(Y, Qbounds)
      ab <- range(Ystar, na.rm = TRUE)
      Ystar[is.na(Ystar)] <- 0
      Ystar <- (Ystar - ab[1])/diff(ab)
      Q <- (Q - ab[1])/diff(ab)
      Qbounds <- c(alpha, 1 - alpha)
      Q <- bound(Q, Qbounds)
      Q <- qlogis(Q)
      family.stage2 <-  "binomial"
      Qinit<-list(Q=Q, Y=Ystar, Qbounds=Qbounds, ab=ab, family=family.stage2)

      # ctmle_glmnet.R:213-219 and 224-230, with the internal functions named.
      candidates <- ctmle:::stage2_glmnet(Y=Qinit$Y, X=data.frame(A,Wg), Q=Qinit$Q,
                                          lambdas = lambdas,ctmletype=ctmletype,
                                          family=Qinit$family,
                                          Qbounds=Qinit$Qbounds,ab=Qinit$ab,
                                          training_set=rep(TRUE,length(Y)),
                                          like_type=like_type,gbound=gbound, verbose=verbose,
                                          stopFactor=stopFactor)

      cv.res <- ctmle:::cv_glmnet(Y=Qinit$Y,X=data.frame(A,Wg),
                                  est.all=candidates$results.all$est,
                                  Q=Qinit$Q,lambdas=lambdas,ctmletype=ctmletype,
                                  family=Qinit$family,
                                  Qbounds=Qinit$Qbounds, ab=Qinit$ab,
                                  like_type=like_type,gbound=gbound,
                                  verbose=verbose, PEN=PEN , V=V, folds = folds)

      # ctmle_glmnet.R:346-347, verbatim.
      candidates$results.all$varDstar_origscale <- candidates$results.all$varDstar *
            ifelse(maptoYstar, (diff(Qinit$ab))^2, 1)

      list(candidates = candidates, cv.res = cv.res, n = n)
}

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

fit_lasso <- function(frame, W, Q, label) {
  folds <- selection_folds(frame)
  lambdas <- c(STUDY_LAMBDAS, 0)
  # The default path of the fit's own construction (functions_glmnet.R:130-134): once on all
  # rows, and once on the training rows of each cross-validation fold.
  lassox <- model.matrix(~ W1 + W2 + W3, W)[, -1]
  path_minimum <- function(rows) {
    study_stage("default glmnet path", min(glmnet(
      x = lassox[rows, , drop = FALSE], y = as.factor(frame$A[rows]), family = "binomial"
    )$lambda))
  }
  path_min <- path_minimum(seq_len(nrow(frame)))
  fold_path_min <- vapply(folds, function(held) path_minimum(-held), numeric(1))
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
  # A ctmleGlmnet error comes back as a condition, so the adapter below can look at it.  When
  # the harness records failures, the stage has already made it a study_fitting_failure.  A
  # fatal error passes through.
  fit <- tryCatch(
    study_stage("ctmleGlmnet", do.call(ctmleGlmnet, arguments)),
    error = function(condition) {
      if (inherits(condition, "study_fatal_error")) stop(condition)
      condition
    }
  )
  if (!inherits(fit, "error")) {
    return(list(
      estimate = as.numeric(fit$est),
      std_error = sqrt(as.numeric(fit$var.psi)),
      best_k = as.integer(fit$best_k),
      n_candidates = length(fit$candidates$candidates$lambdas),
      adapter_used = 0L,
      path_min_lambda = path_min,
      fold_path_min_lambda = fold_path_min
    ))
  }
  # Only the quirk-7 failure is adapted: the rebuilt selection must be the last penalty, or
  # the original ctmleGlmnet error is recorded as this group's failure.
  rebuilt <- study_stage("rebuild_type1", do.call(rebuild_type1, arguments))
  best_k <- rebuilt$cv.res$best_k
  if (!identical(as.integer(best_k), length(lambdas))) stop(fit)
  results <- rebuilt$candidates$results.all
  list(
    estimate = as.numeric(results$est[best_k]),
    std_error = sqrt(as.numeric(results$varDstar_origscale[best_k]) / rebuilt$n),
    best_k = as.integer(best_k),
    n_candidates = length(rebuilt$candidates$candidates$lambdas),
    adapter_used = 1L,
    path_min_lambda = path_min,
    fold_path_min_lambda = fold_path_min
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
  # R's plain D* variance at the selected candidate, never `var.psi`, which carries the
  # first-covariate ICg term (quirk 6).
  variance <- as.numeric(fit$candidates$results.all$varDstar_origscale[best_k]) / n
  list(
    estimate = as.numeric(fit$est),
    std_error = sqrt(variance),
    best_k = best_k,
    n_candidates = NCOL(fit$candidates$candidates$gns),
    adapter_used = 0L,
    path_min_lambda = NA_real_,
    fold_path_min_lambda = rep(NA_real_, V)
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
  W <- frame[c("W1", "W2", "W3")]
  q_fit <- study_stage("initial glm", glm(Y ~ A + W1 + W2 + W3, data = frame, family = binomial()))
  q0 <- predict(q_fit, transform(frame, A = 0), type = "response")
  q1 <- predict(q_fit, transform(frame, A = 1), type = "response")
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
