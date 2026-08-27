# A one-fold point-treatment adapter for pinned lmtp 1.5.4.

lmtp_point_internal <- function(name) {
  value <- getFromNamespace(name, "lmtp")
  if (is.null(value)) stop(sprintf("lmtp internal %s is unavailable", name))
  value
}

lmtp_point_tmle <- function(
  data,
  shifted,
  density_ratio,
  mtp,
  outcome_type = "binomial",
  bounds = NULL,
  W = "W",
  learners_outcome = "SL.glm",
  learners_trt = "SL.glm"
) {
  if (nrow(data) != nrow(shifted) || length(density_ratio) != nrow(data)) {
    stop("natural, shifted, and density-ratio rows do not align")
  }
  Task <- lmtp_point_internal("LmtpTask")
  task <- Task$new(
    data = data,
    shifted = shifted,
    A = "A",
    Y = "Y",
    L = NULL,
    W = W,
    C = NULL,
    D = NULL,
    k = Inf,
    id = NULL,
    outcome_type = outcome_type,
    bounds = bounds,
    folds = 1,
    weights = NULL
  )
  control <- lmtp_control(
    .trim = 1,
    .learners_outcome_folds = 2,
    .learners_trt_folds = 2,
    .return_full_fits = TRUE
  )
  progress <- function(...) invisible(NULL)
  density <- lmtp_point_internal("cf_density_ratios")(
    task, learners_trt, mtp, control, progress
  )
  supplied <- matrix(as.numeric(density_ratio), ncol = 1)
  if (!identical(dim(supplied), dim(density$density_ratios))) {
    stop("the supplied point-treatment density ratio has the wrong shape")
  }
  if (any(!is.finite(supplied)) || any(supplied < 0)) {
    stop("the supplied point-treatment density ratio is invalid")
  }
  if (!mtp && any(xor(supplied == 0, density$density_ratios == 0))) {
    stop("the supplied deterministic ratio follows a different intervention path from lmtp")
  }
  density$density_ratios <- supplied
  regressions <- lmtp_point_internal("cf_tmle")(
    task, supplied, learners_outcome, control, progress
  )
  fit <- regressions$fits[[1]][[1]]
  initial <- as.numeric(predict(fit, shifted[c(W, "A")]))
  if (anyNA(initial) || any(initial < 0) || any(initial > 1)) {
    stop("lmtp produced an invalid initial point-treatment plug-in")
  }
  result <- lmtp_point_internal("theta_dr")(
    task = task,
    sequential_regressions = list(
      natural = regressions$natural,
      shifted = regressions$shifted
    ),
    density_ratios = supplied,
    fits_m = regressions$fits,
    fits_r = density$fits,
    shift = "supplied point-treatment policy",
    is_sdr = FALSE
  )
  result$initial <- task$rescale(mean(initial))
  result
}
