suppressPackageStartupMessages({
  library(sl3)
  library(tmle3)
})
options(digits = 17)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(args[[1]]), stringsAsFactors = FALSE)
truths <- read.csv(args[[2]], stringsAsFactors = FALSE)
output <- args[[3]]
scenario <- "bounded_continuous_projection"
labels <- c("msm[(intercept)]", "msm[a]", "msm[W]")
z <- qnorm(0.975)

# Param_MSM documents custom weight functions, but this pinned release compares its input
# with two string sentinels before it checks is.function().  Giving the function a class with
# a narrow equality method lets those sentinel checks fall through to the documented custom
# function branch without modifying the pinned package.
weight <- function(A, V) 1 + 0.5 * A + 5 * V
class(weight) <- c("cleverly_msm_weight", class(weight))
Ops.cleverly_msm_weight <- function(e1, e2) {
  if (.Generic == "==") return(FALSE)
  NextMethod()
}
learners <- list(A = sl3::Lrnr_glm$new(), Y = sl3::Lrnr_glm$new())

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  stage <- function(label, expression) {
    tryCatch(
      force(expression),
      error = function(condition) {
        call <- paste(deparse(conditionCall(condition)), collapse = " ")
        stop(sprintf("%s at %s: %s", label, call, conditionMessage(condition)), call. = FALSE)
      }
    )
  }
  data <- frame[c("W", "A", "Y")]
  # This tmle3 release simplifies a one-column W node to a vector, while Param_MSM indexes
  # that node by the V column name.  A deterministic second baseline column keeps the node a
  # named data.table; the spec itself then adds V to W as its documented construction does.
  data$W_squared <- data$W^2
  data <- data[c("W", "W_squared", "A", "Y")]
  nodes <- list(W = "W_squared", V = "W", A = "A", Y = "Y")
  spec <- tmle3_Spec_MSM$new(msm = "A + V", weight = weight, weight_ub = NULL)
  task <- stage("task", spec$make_tmle_task(data, nodes))
  initial <- stage("initial likelihood", spec$make_initial_likelihood(task, learners))
  updater <- tmle3_Update$new(cvtmle = FALSE, convergence_type = "sample_size")
  targeted <- Targeted_Likelihood$new(initial, updater)
  parameter <- stage("targeted parameter", spec$make_params(task, targeted))
  updater$tmle_params <- parameter
  stage("targeting", fit_tmle3(task, targeted, parameter, updater))

  targeted_values <- stage("targeted estimates", parameter$estimates(task))
  initial_parameter <- stage("initial parameter", suppressWarnings(Param_MSM$new(
    initial,
    "W",
    strata_name = "V",
    msm = "A + V",
    weight = weight,
    weight_ub = NULL,
    treatment_values = c(0, 1)
  )))
  initial_values <- stage("initial estimates", initial_parameter$estimates(task))

  transform_beta <- function(beta) {
    arm0 <- unname(beta[[1]])
    arm1 <- unname(beta[[2]])
    c(arm0, arm1 - arm0, unname(beta[[3]]))
  }
  transform_ic <- function(ic) {
    values <- as.matrix(ic)
    cbind(values[, 1], values[, 2] - values[, 1], values[, 3])
  }
  estimate <- transform_beta(targeted_values$psi)
  initial_estimate <- transform_beta(initial_values$psi)
  influence <- transform_ic(targeted_values$IC)
  standard_error <- apply(influence, 2, sd) / sqrt(nrow(data))
  low <- estimate - z * standard_error
  high <- estimate + z * standard_error

  selected <- truths$replicate == replicate & truths$scenario == scenario
  truth <- truths$truth[selected][match(labels, truths$estimand[selected])]
  if (any(is.na(truth))) stop(sprintf("truth join failed for replicate %s", replicate))
  data.frame(
    implementation = "tmle3",
    scenario = scenario,
    replicate = replicate,
    n = nrow(data),
    estimand = labels,
    truth = truth,
    estimate = estimate,
    inference_estimate = estimate,
    std_error = standard_error,
    ci_lower = low,
    ci_upper = high,
    inference_scale = "identity",
    covered = as.integer(low <= truth & truth <= high),
    initial_estimate = initial_estimate,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

groups <- split(samples, samples$replicate)
expected <- length(groups)
rm(samples)
invisible(gc())

fit_group <- function(index) {
  frame <- groups[[index]]
  tryCatch(
    {
      result <- fit_one(frame)
      if (index %% 10 == 0) cat(sprintf("completed %d/%d samples\n", index, expected))
      result
    },
    error = function(condition) {
      structure(
        list(index = index, replicate = frame$replicate[[1]], message = conditionMessage(condition)),
        class = "study_error"
      )
    }
  )
}

requested <- suppressWarnings(as.integer(Sys.getenv("CLEVERLY_R_CORES", "")))
cores <- if (is.na(requested) || requested < 1L) max(1L, parallel::detectCores()) else requested
memory_kb <- as.numeric(
  sub("[^0-9]*([0-9]+).*", "\\1", grep("MemTotal", readLines("/proc/meminfo"), value = TRUE))
)
footprint_kb <- as.numeric(utils::object.size(groups)) / 1024
affordable <- max(1L, as.integer(floor(memory_kb * 0.5 / max(footprint_kb * 0.5, 1))))
if (affordable < cores) cores <- affordable
cat(sprintf("fitting %d samples on %d cores\n", expected, cores))
results <- parallel::mclapply(seq_along(groups), fit_group, mc.cores = cores, mc.preschedule = FALSE)
malformed <- which(!vapply(results, is.data.frame, logical(1)) &
  !vapply(results, inherits, logical(1), what = "study_error"))
if (length(malformed)) stop(sprintf("workers returned no result for groups %s", paste(malformed, collapse = ", ")))
failed <- vapply(results, inherits, logical(1), what = "study_error")
if (any(failed)) {
  messages <- vapply(
    results[failed],
    function(error) sprintf("replicate %s: %s", error$replicate, error$message),
    character(1)
  )
  stop(paste(messages, collapse = "\n"))
}
result <- do.call(rbind, results)
if (length(unique(result$replicate)) != expected) stop("the R run dropped replications")
write.csv(result, output, row.names = FALSE, na = "")
cat("tmle3 ", as.character(packageVersion("tmle3")), "\n", sep = "")
cat("sl3 ", as.character(packageVersion("sl3")), "\n", sep = "")
cat("R ", R.version.string, "\n", sep = "")
