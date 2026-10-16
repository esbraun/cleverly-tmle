suppressPackageStartupMessages({
  library(sl3)
  library(tmle3)
})
options(digits = 17)

# ---- option fixtures shared: begin ----
# The four scripts in tests/canonical/option_fixtures carry this block byte for byte;
# tests/unit/test_option_fixtures.py compares the copies.  Change it in all four.
#
# It writes the fixture as JSON with base R alone (the pinned images need not hold jsonlite) and
# records where the fixture came from.  A length-one vector is written as a scalar and a longer
# one as an array; a missing or non-finite number is written as null.  Floats carry 17
# significant digits, so the file round-trips exactly.

fixture_arguments <- function(usage) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 1L) stop(usage)
  args[[1]]
}

fixture_string <- function(value) {
  if (is.na(value)) "null" else encodeString(value, quote = "\"")
}

fixture_number <- function(value) {
  if (is.na(value) || !is.finite(value)) {
    "null"
  } else if (value == round(value) && abs(value) < 1e15) {
    sprintf("%.0f", value)
  } else {
    sprintf("%.17g", value)
  }
}

fixture_json <- function(x, indent = 0L) {
  pad <- strrep("  ", indent)
  inner <- strrep("  ", indent + 1L)
  if (is.null(x)) {
    return("null")
  }
  if (is.list(x)) {
    if (length(x) == 0L) {
      return(if (is.null(names(x))) "[]" else "{}")
    }
    parts <- vapply(x, fixture_json, character(1), indent = indent + 1L, USE.NAMES = FALSE)
    if (!is.null(names(x))) {
      keys <- vapply(names(x), fixture_string, character(1), USE.NAMES = FALSE)
      return(paste0("{\n", paste0(inner, keys, ": ", parts, collapse = ",\n"), "\n", pad, "}"))
    }
    return(paste0("[\n", paste0(inner, parts, collapse = ",\n"), "\n", pad, "]"))
  }
  if (is.character(x)) {
    values <- vapply(x, fixture_string, character(1), USE.NAMES = FALSE)
  } else if (is.logical(x)) {
    values <- ifelse(is.na(x), "null", ifelse(x, "true", "false"))
  } else if (is.numeric(x)) {
    values <- vapply(as.numeric(x), fixture_number, character(1), USE.NAMES = FALSE)
  } else {
    stop(sprintf("cannot write a %s to JSON", class(x)[[1]]))
  }
  if (length(values) == 1L) values else paste0("[", paste(values, collapse = ", "), "]")
}

fixture_provenance <- function(script, packages) {
  # The checksum is the script's own, so a JSON file says which script wrote it.  The queue
  # command also passes CLEVERLY_SCRIPT_SHA256, used when tools::sha256sum is absent.
  file_argument <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  path <- if (length(file_argument) > 0L) sub("^--file=", "", file_argument[[1]]) else NA_character_
  digest <- Sys.getenv("CLEVERLY_SCRIPT_SHA256", unset = NA_character_)
  if (!is.na(path) && exists("sha256sum", envir = asNamespace("tools"), inherits = FALSE)) {
    digest <- unname(tools::sha256sum(path))
  }
  list(
    script = script,
    script_sha256 = digest,
    image_id = Sys.getenv("CLEVERLY_IMAGE_ID", unset = NA_character_),
    r_version = R.version.string,
    packages = lapply(
      stats::setNames(nm = packages),
      function(name) as.character(utils::packageVersion(name))
    )
  )
}

fixture_write <- function(payload, path) {
  # Raw bytes, so the file ends in LF whatever the platform's text-mode habits are.
  text <- paste0(fixture_json(payload), "\n")
  connection <- file(path, open = "wb")
  on.exit(close(connection), add = TRUE)
  writeBin(charToRaw(text), connection)
  invisible(path)
}
# ---- option fixtures shared: end ----

# X22 parts (a), (b) and (c), the comparator side.  `tmle_TSM_all()` on one binary-outcome sample,
# fitted with `tmle3_Update` at the settings below.  The script exports what a Python run needs
# to repeat each updater: the outcome, the initial outcome regression at the observed treatment
# and at each arm, the observed-arm propensity, the per-step epsilons and EDs, and the final
# targeted arrays, estimates and thresholds.  It asserts that `bound(., 0.005)`, which
# `generate_submodel_data` applies at every step, never binds.
#
# Sources read at tmle3 0.2.0, commit ed72f8a (`R/tmle3_Update.R`, `R/tmle3_Fit.R`,
# `R/Param_TSM.R`, `R/Targeted_Likelihood.R`, `R/tmle3_Spec_TSM_all.R`):
#   tmle3_Update$new(maxit, cvtmle, one_dimensional, constrain_step, delta_epsilon,
#     convergence_type, fluctuation_type, optim_delta_epsilon, use_best, verbose)
#   updater$step_number, updater$epsilons[[step]]$Y, updater$EDs[[step]]
#   check_convergence: ED_threshold is se_Dstar / min(log(n), 10) for "scaled_var" and 1 / n for
#     "sample_size", with n = length(unique(tmle_task$id)) and se_Dstar = sqrt(var(IC) / n).
#   update: update_best after each step that did not converge and once after the loop, then
#     set_best.  update_best keeps a step only when sqrt(sum(ED^2)) is strictly smaller.
#   fit_submodel under constrain_step: optim(method = "Brent") on [0, delta_epsilon], or
#     delta_epsilon itself when optim_delta_epsilon is FALSE.
#   collapse_covariates: H %*% (ED / norm(ED, "2")), with ED from the estimates before the step.

output <- fixture_arguments("usage: tmle3_updater.R OUTPUT.json")
provenance <- fixture_provenance("tmle3_updater.R", c("tmle3", "sl3"))

SEED <- 20261015L
N <- 300L
BOUND <- 0.005
DELTA <- 0.01
MAXIT_CONSTRAINED <- 200L

RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(SEED)
W1 <- rnorm(N)
W2 <- rnorm(N)
A <- rbinom(N, 1L, plogis(0.4 * W1 - 0.3 * W2))
Y <- rbinom(N, 1L, plogis(-0.3 + 0.8 * A + 0.5 * W1 - 0.4 * W2))
data <- data.frame(W1 = W1, W2 = W2, A = A, Y = Y)
nodes <- list(W = c("W1", "W2"), A = "A", Y = "Y")
learners <- list(A = sl3::Lrnr_glm$new(), Y = sl3::Lrnr_glm$new())

spec <- tmle_TSM_all()
task <- spec$make_tmle_task(data, nodes)
initial <- spec$make_initial_likelihood(task, learners)

numbers <- function(x) as.numeric(unlist(x, use.names = FALSE))

# The counterfactual prediction of the outcome regression for the arm one parameter targets,
# read as `Param_TSM$estimates` reads it.
arm_prediction <- function(param) {
  cf_task <- param$cf_likelihood$enumerate_cf_tasks(task)[[1]]
  numbers(param$cf_likelihood$get_likelihood(cf_task, "Y", "full"))
}

# Parameters on the untargeted likelihood give the initial arrays and the clever covariates,
# which do not change from step to step.
initial_params <- spec$make_params(task, initial)
H0 <- do.call(cbind, lapply(initial_params, function(p) numbers(p$clever_covariates(task, "full")$Y)))
q_observed <- numbers(initial$get_likelihood(task, "Y", "full"))
q_arms <- lapply(initial_params, arm_prediction)
propensity_observed <- numbers(initial$get_likelihood(task, "A", "full"))
initial_estimates <- lapply(initial_params, function(p) p$estimates(task, "full"))
ED0 <- colMeans(do.call(cbind, lapply(initial_estimates, `[[`, "IC")))

inside <- function(q) all(q >= BOUND & q <= 1 - BOUND)
stopifnot(inside(q_observed), inside(unlist(q_arms)))

# The observed-treatment prediction after each step, rebuilt from the epsilons: the value
# `generate_submodel_data` and `apply_submodel` give.  It checks the reading of the collapse above
# and that the bound never binds along the walk.
replay <- function(epsilons, EDs, one_dimensional) {
  q <- q_observed
  previous <- ED0
  binds <- FALSE
  trace <- vector("list", length(epsilons))
  for (k in seq_along(epsilons)) {
    if (!inside(q)) binds <- TRUE
    h <- if (one_dimensional) H0 %*% (previous / sqrt(sum(previous^2))) else H0
    q <- plogis(qlogis(pmin(pmax(q, BOUND), 1 - BOUND)) + as.numeric(h %*% epsilons[[k]]))
    trace[[k]] <- q
    previous <- EDs[[k]]
  }
  if (!inside(q)) binds <- TRUE
  list(trace = trace, binds = binds)
}

run_updater <- function(name, convergence_type, constrained, optimise, use_best) {
  maxit <- if (constrained) MAXIT_CONSTRAINED else 100L
  updater <- if (constrained) {
    tmle3_Update$new(
      maxit = maxit, cvtmle = FALSE, one_dimensional = TRUE, constrain_step = TRUE,
      delta_epsilon = DELTA, convergence_type = convergence_type,
      optim_delta_epsilon = optimise, use_best = use_best
    )
  } else {
    tmle3_Update$new(
      maxit = maxit, cvtmle = FALSE, convergence_type = convergence_type, use_best = use_best
    )
  }
  targeted <- Targeted_Likelihood$new(initial, updater)
  params <- spec$make_params(task, targeted)
  updater$tmle_params <- params
  fit <- fit_tmle3(task, targeted, params, updater)

  steps <- updater$step_number
  epsilons <- lapply(seq_len(steps), function(k) as.numeric(updater$epsilons[[k]]$Y))
  EDs <- lapply(seq_len(steps), function(k) as.numeric(updater$EDs[[k]]))
  norms <- vapply(EDs, function(ed) sqrt(sum(ed^2)), numeric(1))
  # `which.min` returns the first minimum, as the strict `<` of `update_best` does.
  best_step <- if (use_best) which.min(norms) else NULL

  n_units <- length(unique(task$id))
  influence <- do.call(cbind, lapply(fit$estimates, `[[`, "IC"))
  thresholds <- if (convergence_type == "scaled_var") {
    sqrt(apply(influence, 2, var) / n_units) / min(log(n_units), 10)
  } else {
    rep(1 / n_units, ncol(influence))
  }

  reported <- if (use_best) best_step else steps
  walk <- replay(epsilons, EDs, constrained)
  final_observed <- numbers(targeted$get_likelihood(task, "Y", "full"))

  list(
    name = name,
    options = list(
      convergence_type = convergence_type, constrain_step = constrained,
      one_dimensional = constrained, delta_epsilon = if (constrained) DELTA else NULL,
      optim_delta_epsilon = if (constrained) optimise else NULL,
      use_best = use_best, maxit = maxit
    ),
    steps = steps,
    converged = steps < maxit,
    best_step = best_step,
    epsilons = epsilons,
    eds = EDs,
    final = list(
      psi = vapply(fit$estimates, function(e) e$psi, numeric(1)),
      ed = as.numeric(fit$ED),
      thresholds = as.numeric(thresholds),
      q_observed = final_observed,
      q_arm0 = arm_prediction(params[[1]]),
      q_arm1 = arm_prediction(params[[2]])
    ),
    replay = list(
      max_abs_difference = max(abs(walk$trace[[reported]] - final_observed)),
      bound_binds = walk$binds
    )
  )
}

settings <- list(
  list("sample_size", "sample_size", FALSE, NA),
  list("scaled_var", "scaled_var", FALSE, NA),
  list("constrained_optim", "sample_size", TRUE, TRUE),
  list("constrained_fixed", "sample_size", TRUE, FALSE),
  list("constrained_optim_scaled_var", "scaled_var", TRUE, TRUE)
)
updaters <- list()
for (setting in settings) {
  for (use_best in c(FALSE, TRUE)) {
    name <- sprintf("%s_%s", setting[[1]], if (use_best) "best" else "last")
    # An updater that errors is recorded, not fatal: the comparison then names it.
    updaters[[length(updaters) + 1L]] <- tryCatch(
      run_updater(name, setting[[2]], setting[[3]], setting[[4]], use_best),
      error = function(condition) list(name = name, error = conditionMessage(condition))
    )
  }
}

payload <- list(
  fixture = "tmle3_updater",
  schema = 1L,
  provenance = provenance,
  seed = SEED,
  n = N,
  bound = BOUND,
  arm_levels = c(0, 1),
  data = list(Y = as.numeric(Y), A = as.numeric(A), W1 = W1, W2 = W2),
  initial = list(
    propensity_observed = propensity_observed,
    q_observed = q_observed,
    q_arm0 = q_arms[[1]],
    q_arm1 = q_arms[[2]],
    ed = as.numeric(ED0)
  ),
  updaters = updaters
)
fixture_write(payload, output)
