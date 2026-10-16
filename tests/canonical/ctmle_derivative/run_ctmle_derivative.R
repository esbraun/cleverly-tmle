suppressPackageStartupMessages({
  library(ctmle)
  library(glmnet)
})
source("/fixture/study_harness.R")
options(digits = 17)

# R ctmle 0.1.2 at 18de559, ctmletype = 3 (C-TMLE0), for the ctmle-derivative study.
#
# Q is R's own main-terms GLM, except under binary_constant_q:
#
# binary_declared_pair: glmnet at the declared pair of penalties.csv, the pair the package
#   passes as lambdas=. Both sides fit the same estimator.
# binary_constant_q:    the declared pair, with Q = mean(Y) for both arms, the package's
#   DummyRegressor. The update then moves the estimate, so the derivative columns show.
#   R bounds only the initial Q; the package also bounds each targeted arm prediction at
#   0.005 and 0.995 after every step. Under a constant Q the derivative coefficients are large,
#   so the two sides differ by about 1e-5 in SD, 0.003 of the parity witness's reference SD.
# binary_pipeline:      cv.glmnet(nfolds = 10, nlambda = 100), g1W at lambda.min and the
#   partner at 1.05 lambda.min, as the ctmle README does.
#
# glmnet.control(fdev = 0, devmax = 1) keeps the whole path, the setting the package's default
# grid reproduces. Each replicate is seeded from its scenario and replicate, so cv.glmnet's
# folds do not depend on the worker count.

args <- study_arguments("usage: run_ctmle_derivative.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(args$samples), stringsAsFactors = FALSE)
truths <- read.csv(args$truths, stringsAsFactors = FALSE)
truth_key <- paste(truths$scenario, truths$replicate, truths$estimand, sep = "|")
declared <- read.csv("/fixture/ctmle_derivative/penalties.csv", stringsAsFactors = FALSE)
lam <- declared$lambda[declared$role == "mechanism"]
lam_plus <- declared$lambda[declared$role == "partner"]
if (length(lam) != 1L || length(lam_plus) != 1L || lam == lam_plus) {
  stop("penalties.csv must hold one mechanism and one different partner penalty")
}
gbound <- 0.025
alpha <- 0.995
covariates <- c("W1", "W2", "W3", "W4")
scenarios <- c("binary_declared_pair", "binary_pipeline", "binary_constant_q")
invisible(glmnet.control(fdev = 0, devmax = 1))

mechanism_pair <- function(W, A, scenario) {
  if (scenario %in% c("binary_declared_pair", "binary_constant_q")) {
    fit <- study_stage("mechanism glmnet", glmnet(
      W, factor(A), family = "binomial", lambda = sort(c(lam, lam_plus), decreasing = TRUE),
      thresh = 1e-14, maxit = 1e7
    ))
    return(list(
      g1W = as.vector(predict(fit, newx = W, s = lam, type = "response")),
      g1WPrev = as.vector(predict(fit, newx = W, s = lam_plus, type = "response")),
      mechanism = lam,
      partner = lam_plus
    ))
  }
  cv <- study_stage(
    "mechanism cv.glmnet", cv.glmnet(W, factor(A), family = "binomial", nfolds = 10, nlambda = 100)
  )
  chosen <- cv$lambda.min
  list(
    g1W = as.vector(predict(cv, newx = W, s = chosen, type = "response")),
    g1WPrev = as.vector(predict(cv, newx = W, s = chosen * 1.05, type = "response")),
    mechanism = chosen,
    partner = chosen * 1.05
  )
}

fit_one <- function(frame) {
  scenario <- frame$scenario[[1]]
  replicate <- frame$replicate[[1]]
  set.seed(7919L * match(scenario, scenarios) + replicate + 1L)
  truth_row <- match(paste(scenario, replicate, "ate", sep = "|"), truth_key)
  if (is.na(truth_row)) stop(sprintf("missing truth for %s/%s/ate", scenario, replicate))
  truth <- truths$truth[[truth_row]]
  W <- as.matrix(frame[covariates])
  A <- frame$A
  if (scenario == "binary_constant_q") {
    q0 <- q1 <- rep(mean(frame$Y), nrow(frame))
  } else {
    q_fit <- study_stage(
      "initial glm", glm(Y ~ A + W1 + W2 + W3 + W4, data = frame, family = binomial())
    )
    q0 <- predict(q_fit, transform(frame, A = 0), type = "response")
    q1 <- predict(q_fit, transform(frame, A = 1), type = "response")
  }
  pair <- mechanism_pair(W, A, scenario)
  if (any(pair$g1W == pair$g1WPrev)) {
    stop("the mechanism equals its partner on a row, and ctmle would replace the difference")
  }
  fit <- study_stage("ctmleGlmnet", ctmleGlmnet(
    Y = frame$Y, A = A, W = data.frame(W), Q = cbind(q0, q1), ctmletype = 3,
    g1W = pair$g1W, g1WPrev = pair$g1WPrev, alpha = alpha, family = "binomial",
    gbound = gbound
  ))
  estimate <- as.numeric(fit$est)
  if (!is.finite(estimate)) stop("ctmleGlmnet returned a non-finite estimate")
  std_error <- sqrt(as.numeric(fit$var.psi))
  ci <- as.numeric(fit$CI)
  data.frame(
    implementation = "r-ctmle",
    scenario = scenario,
    replicate = replicate,
    n = nrow(frame),
    estimand = "ate",
    truth = truth,
    estimate = estimate,
    inference_estimate = estimate,
    std_error = std_error,
    ci_lower = ci[[1]],
    ci_upper = ci[[2]],
    inference_scale = "identity",
    covered = as.integer(ci[[1]] <= truth && truth <= ci[[2]]),
    initial_estimate = mean(q1 - q0),
    mechanism_penalty = pair$mechanism,
    partner_penalty = pair$partner,
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
    paste("commit", Sys.getenv("CTMLE_COMMIT"))
  ),
  key = c("scenario", "replicate")
)
