# Competing-risk extension of the pinned lmtp 1.5.4 fold adapter.
#
# Sourced *after* `lmtp_crossfit_adapter.R`, whose `lmtp_internal` and `fold_list` this file
# uses and does not define.  The guard below says so rather than letting a lone `source()`
# fail on an undefined function several frames deep.
#
# This wrapper restates the shared adapter's orchestration with `D` populated, and adds the
# one-fold training-equals-validation convention the ordinary row needs.  The restatement is
# duplication and is deliberate: `lmtp_crossfit_adapter.R` is hashed into four committed
# manifests, two of which belong to studies this change does not regenerate, so its bytes
# cannot move.  Fold the two together when the end-of-study and survival rows are next
# regenerated for a reason of their own; until then the copy is the only way to leave those
# rows' provenance intact.

if (!exists("lmtp_internal") || !exists("fold_list")) {
  stop("source lmtp_crossfit_adapter.R before this file: it defines lmtp_internal and fold_list")
}

# With one learner, SuperLearner's final SL.glm or SL.mean fit is the direct fit below.  Its inner
# folds estimate a library weight that is identically one and do not change the prediction.
# The adapter removes that redundant cross-validation while retaining lmtp's own outer folds,
# competing-risk recursion, fluctuation, and influence-curve implementation.
#
# That is a claim about SuperLearner, not about this study, so it is checked rather than
# asserted -- see `check_direct_learner_matches_superlearner` at the foot of this file, which
# runs once per R process before any study fit.
lmtp_direct_learner <- function(data, y, learners, outcome_type, id, folds) {
  if (!learners %in% c("SL.glm", "SL.mean")) {
    stop("the direct adapter only supports one SL.glm or SL.mean learner")
  }
  if (folds < 2L) stop("the declared learner-fold count must be at least two")
  features <- setdiff(names(data), c(id, y))
  response <- data[[y]]
  binomial_outcome <- outcome_type == "binomial"
  if (identical(learners, "SL.mean")) {
    return(structure(
      list(
        mean = mean(response),
        features = character(0),
        binomial_outcome = binomial_outcome
      ),
      class = "lmtp_direct_learner"
    ))
  }
  x <- cbind("(Intercept)" = 1, as.matrix(data[, features, drop = FALSE]))
  fit <- if (binomial_outcome) {
    glm.fit(x, response, family = binomial())
  } else {
    lm.fit(x, response)
  }
  if (anyNA(fit$coefficients)) {
    stop(sprintf(
      "the direct SL.glm fit of %s on %s is rank deficient",
      y, paste(features, collapse = ", ")
    ))
  }
  structure(
    list(
      coefficients = fit$coefficients,
      features = features,
      binomial_outcome = binomial_outcome,
      mean = NULL
    ),
    class = "lmtp_direct_learner"
  )
}

predict.lmtp_direct_learner <- function(object, newdata, tol = .Machine$double.eps, ...) {
  prediction <- if (!is.null(object$mean)) {
    rep(object$mean, nrow(newdata))
  } else {
    x <- cbind("(Intercept)" = 1, as.matrix(newdata[, object$features, drop = FALSE]))
    linear <- drop(x %*% object$coefficients)
    if (object$binomial_outcome) plogis(linear) else linear
  }
  if (is.null(tol)) return(prediction)
  pmin(1 - tol, pmax(tol, prediction))
}

# Every value the two routes produce, on a fixed sample, for both learners and both outcome
# types.  A comment claiming the substitution is prediction-identical is a claim nothing
# checks; this is the check, and it runs before the substitution is installed so a failure
# stops the study rather than quietly changing what "lmtp" means in its equivalence row.
check_direct_learner_matches_superlearner <- function(tolerance = 1e-10) {
  # Its own seed, and the process's restored afterwards.  This runs at source time, before
  # any study fit, and a check that left the generator somewhere else than it found it would
  # make the run depend on whether the check had run.  Nothing downstream draws today -- the
  # direct learner does no cross-validation -- but that is a property of the substitution
  # rather than a guarantee, and a study whose determinism rests on it should not have to.
  restore <- if (exists(".Random.seed", envir = globalenv())) {
    saved <- get(".Random.seed", envir = globalenv())
    function() assign(".Random.seed", saved, envir = globalenv())
  } else {
    function() rm(".Random.seed", envir = globalenv())
  }
  on.exit(restore(), add = TRUE)
  set.seed(20260825)
  n <- 200L
  frame <- data.frame(
    x1 = stats::rnorm(n),
    x2 = stats::rbinom(n, 1, 0.4),
    ..i..lmtp_id = seq_len(n)
  )
  targets <- list(
    binomial = stats::rbinom(n, 1, stats::plogis(0.3 * frame$x1 - 0.5 * frame$x2)),
    continuous = stats::plogis(0.2 + 0.4 * frame$x1 - 0.3 * frame$x2)
  )
  for (outcome_type in names(targets)) {
    for (learner in c("SL.glm", "SL.mean")) {
      data <- cbind(frame, y = targets[[outcome_type]])
      direct <- lmtp_direct_learner(
        data, "y", learner, outcome_type, "..i..lmtp_id", 2L
      )
      family <- if (outcome_type == "binomial") stats::binomial() else stats::gaussian()
      ensemble <- SuperLearner::SuperLearner(
        Y = data$y,
        X = data[, c("x1", "x2"), drop = FALSE],
        SL.library = learner,
        family = family,
        cvControl = SuperLearner::SuperLearner.CV.control(V = 2L),
        # SuperLearner resolves its screening algorithm by name in `env`, which defaults to
        # the calling frame.  Called from inside a function, that frame does not hold `All`.
        env = asNamespace("SuperLearner")
      )
      expected <- as.numeric(
        stats::predict(ensemble, newdata = data[, c("x1", "x2"), drop = FALSE])$pred
      )
      observed <- stats::predict(direct, data, tol = NULL)
      worst <- max(abs(observed - expected))
      if (!is.finite(worst) || worst > tolerance) {
        stop(sprintf(
          "the direct %s fit differs from SuperLearner's by %.3g on a %s outcome; the
           registered competing-risk rows describe their comparator as lmtp running that
           learner, so the substitution has to be prediction-identical",
          learner, worst, outcome_type
        ))
      }
    }
  }
  invisible(TRUE)
}

check_direct_learner_matches_superlearner()

lmtp_namespace <- asNamespace("lmtp")
unlockBinding("run_ensemble", lmtp_namespace)
assign("run_ensemble", lmtp_direct_learner, envir = lmtp_namespace)
lockBinding("run_ensemble", lmtp_namespace)

# The shared adapter's screens on a supplied ratio matrix, restated here for the reason the
# orchestration below is.  Dropping them was not an option: the competing runner supplies a
# hand-written lookup table over eight history cells per node, and these three checks are what
# stand between a mistyped cell and an equivalence row that agrees for the wrong reason.
check_supplied_density_ratios <- function(density_ratios, estimated) {
  if (!identical(dim(density_ratios), dim(estimated))) {
    stop(sprintf(
      "supplied density ratios are %s; lmtp's are %s",
      paste(dim(density_ratios), collapse = " x "),
      paste(dim(estimated), collapse = " x ")
    ))
  }
  # A per-node ratio is zero exactly where the unit left the followed path at that node.
  # lmtp's own matrix has that property, so requiring the supplied one to agree with it
  # cell for cell is what catches a ratio built from the wrong arm, the wrong node, or a
  # missing censoring factor -- none of which changes the shape.
  # ``any(xor(...))`` rather than ``identical(...)``: the latter compares dimnames too, so
  # a supplied matrix built with ``cbind(first, second)`` failed on its column labels while
  # every value agreed.
  disagreeing <- sum(xor(density_ratios == 0, estimated == 0))
  if (disagreeing > 0) {
    stop(sprintf(
      "the supplied density ratios follow a different path from lmtp's: %d of %d cells
       disagree on whether the unit is still on the regimen at that node",
      disagreeing, length(estimated)
    ))
  }
  if (!all(is.finite(density_ratios))) {
    stop("the supplied density ratios contain non-finite values")
  }
  # Agreement in *shape* with lmtp's own estimate.  The zero-pattern check above is the
  # sharp structural one; this catches a formula that is right about which units follow and
  # wrong about the weight -- a dropped censoring factor, an inverted arm probability.
  supplied <- apply(density_ratios, 1, prod)
  fitted <- apply(estimated, 1, prod)
  agreement <- suppressWarnings(cor(supplied, fitted))
  if (!is.finite(agreement) || agreement < 0.8) {
    stop(sprintf(
      "the supplied cumulative density ratio correlates %.4f with lmtp's own estimate of
       the same quantity; a correct formula tracks it closely even where the two differ
       on the value", agreement
    ))
  }
  # A gross-error screen only, deliberately loose, and applied to the *first* column rather
  # than to the cumulative product.
  #
  # The shared adapter screens the cumulative mean against one.  That is right for the laws it
  # was written against and wrong here.  A unit that had an event of either cause at the first
  # node has no second-node arm, so its second column is structurally zero, and the cumulative
  # ratio's expectation is the all-cause event-free probability under the plan.  On this law
  # that is 0.3125 under `never` and 0.4375 under either treated plan, so a screen anchored at
  # one rejects a correct matrix.  lmtp's own cumulative mean is no better a reference: its
  # second-node mechanism is a main-effects `glm` on a history whose true conditional is a
  # saturated table, which is the misspecification this study supplies exact ratios to avoid.
  #
  # The first column has no such problem.  Nothing has left through an event yet, so its only
  # zeros are the ones an inverse-probability weight exists to correct, and its expectation is
  # exactly one whatever the event process does downstream.  The second node stays covered by
  # the zero-pattern and correlation checks above.
  #
  # Screened at five standard errors of the column's own mean rather than at a fixed width,
  # because the two have to be separated at every sample size the runner is called at.  A
  # dropped first-node censoring factor displaces the mean by about 0.25 whatever `n` is,
  # while the sampling spread of a correct column shrinks with it: a fixed band wide enough
  # not to reject a correct matrix at the smoke's size is too wide to catch that mutation at
  # the same size, and a band tight enough at the study's size rejects correct ones at the
  # smoke's.  Five standard errors clears the widest deviation seen in 400 draws at every size
  # by more than a factor of two.
  first_node <- density_ratios[, 1]
  spread <- stats::sd(first_node) / sqrt(length(first_node))
  if (!is.finite(mean(first_node)) || abs(mean(first_node) - 1) > 5 * spread) {
    stop(sprintf(
      "the supplied first-node density ratio averages %.4f, which is %.1f standard errors
       from one; that column has no competing event upstream of it, so a correct one
       averages one", mean(first_node), abs(mean(first_node) - 1) / spread
    ))
  }
  invisible(TRUE)
}

lmtp_competing_tmle_with_folds <- function(
  data,
  shifted,
  trt,
  outcome,
  compete,
  baseline = NULL,
  time_vary = NULL,
  cens = NULL,
  outcome_type = "survival",
  fold_assignment,
  learners_outcome = "SL.glm",
  learners_trt = "SL.glm",
  density_ratios = NULL,
  control = lmtp::lmtp_control(
    .trim = 1,
    .learners_outcome_folds = 2,
    .learners_trt_folds = 2,
    .return_full_fits = TRUE
  )
) {
  if (nrow(data) != length(fold_assignment)) stop("fold assignment has the wrong length")
  variables <- c(unlist(trt), outcome, compete, unlist(time_vary), baseline, cens)
  variables <- unique(variables[!is.na(variables)])
  natural <- data[, variables, drop = FALSE]
  intervention <- shifted[, variables, drop = FALSE]

  labels <- sort(unique(as.integer(fold_assignment)))
  if (!identical(labels, seq.int(0L, length(labels) - 1L))) {
    stop("fold assignment must use contiguous zero-based labels")
  }
  # The ordinary row is one fold, and lmtp's own fold list has no such case: `fold_list`
  # refuses a fold with no training rows, which is exactly what one fold is.  Training on
  # every row and evaluating on every row is what `n_folds=1` means on the cleverly side, so
  # the paired comparison states it here rather than approximating it with two folds.
  if (length(labels) == 1L) {
    folds <- list(list(training_set = seq_len(nrow(data)), validation_set = seq_len(nrow(data))))
  } else {
    folds <- fold_list(fold_assignment)
  }

  Task <- lmtp_internal("LmtpTask")
  task <- Task$new(
    data = natural,
    shifted = intervention,
    A = trt,
    Y = outcome,
    L = time_vary,
    W = baseline,
    C = cens,
    D = compete,
    k = Inf,
    id = NULL,
    outcome_type = outcome_type,
    bounds = NULL,
    folds = length(folds),
    weights = NULL
  )
  task$folds <- folds
  progress <- function(...) invisible(NULL)

  # Fitted unconditionally, as the shared adapter does, even when the caller supplies exact
  # ratios: lmtp's own estimate is what the screens below compare against, and a screen that
  # only runs when nobody supplied anything screens nothing.
  density <- lmtp_internal("cf_density_ratios")(task, learners_trt, FALSE, control, progress)
  if (!is.null(density_ratios)) {
    if (nrow(density_ratios) != nrow(data) || ncol(density_ratios) != length(trt)) {
      stop("supplied density ratios have the wrong dimensions")
    }
    if (any(density_ratios < 0)) stop("supplied density ratios are negative")
    check_supplied_density_ratios(density_ratios, density$density_ratios)
    density <- list(density_ratios = density_ratios, fits = NULL)
  }

  regressions <- lmtp_internal("cf_tmle")(
    task, density$density_ratios, learners_outcome, control, progress
  )
  if (!isTRUE(control$.return_full_fits)) {
    stop("the evidence runner requires .return_full_fits=TRUE")
  }

  initial <- rep(NA_real_, nrow(data))
  for (fold in seq_along(task$folds)) {
    natural_fold <- lmtp_internal("get_folded_data")(task$natural, task$folds, fold)
    shifted_fold <- lmtp_internal("get_folded_data")(task$shifted, task$folds, fold)
    y0 <- task$is_outcome_free(natural_fold$valid, 0)
    c0 <- task$observed(natural_fold$valid, 0)
    d0 <- task$is_competing_risk_free(natural_fold$valid, 0)
    valid <- c0 & y0 & d0
    history <- task$vars$history("L", 2)
    missing_columns <- setdiff(history, names(natural_fold$valid))
    if (length(missing_columns)) {
      stop(sprintf(
        "fold %d is missing %s from the first sequential regression's history",
        fold, paste(missing_columns, collapse = ", ")
      ))
    }
    first_treatment <- lmtp_internal("current_trt")(task$vars$A, 1)
    under_shift <- natural_fold$valid[valid, c("..i..lmtp_id", history)]
    under_shift[, first_treatment] <- shifted_fold$valid[valid, first_treatment]
    if (length(regressions$fits[[fold]]) != length(task$vars$A)) {
      stop(sprintf(
        "fold %d returned %d sequential regressions for %d treatment nodes",
        fold, length(regressions$fits[[fold]]), length(task$vars$A)
      ))
    }
    predicted <- predict(regressions$fits[[fold]][[1]], under_shift, 1e-05)
    if (anyNA(predicted) || any(predicted < 0) || any(predicted > 1)) {
      stop(sprintf("fold %d produced an initial plug-in outside [0, 1]", fold))
    }
    held_out <- task$folds[[fold]]$validation_set[valid]
    initial[held_out] <- predicted
  }
  if (anyNA(initial)) stop("the initial plug-in did not cover every row")

  result <- lmtp_internal("theta_dr")(
    task = task,
    sequential_regressions = list(
      natural = regressions$natural,
      shifted = regressions$shifted
    ),
    density_ratios = density$density_ratios,
    fits_m = regressions$fits,
    fits_r = density$fits,
    shift = "supplied competing-risk panel with exact folds",
    is_sdr = FALSE
  )
  result$initial <- initial
  result$fold_assignment <- as.integer(fold_assignment)
  result
}
