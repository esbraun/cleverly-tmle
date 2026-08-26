suppressPackageStartupMessages(library(ltmle))
options(digits = 17)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(args[[1]]), stringsAsFactors = FALSE, check.names = FALSE)
truths <- read.csv(args[[2]], stringsAsFactors = FALSE, check.names = FALSE)
output <- args[[3]]

scenario <- "censored_regimen_projection"
dynamic_label <- "treat then continue if l2 positive"
regimens <- c("never", "always", "early", dynamic_label)
duration <- c(never = 0, always = 2, early = 1)
duration[[dynamic_label]] <- 1
projection_weight <- c(never = 0.1, always = 10, early = 0.1)
projection_weight[[dynamic_label]] <- 10
labels <- c("msm_regimen[(intercept)]", "msm_regimen[duration]")
qform <- c(L2 = "Q.kplus1 ~ W1 + W2", Y = "Q.kplus1 ~ W1 + W2 + L2")
z <- qnorm(0.975)

assigned <- function(frame, label) {
  a1 <- if (label == "never") 0 else 1
  a2 <- if (label == "never" || label == "early") {
    rep(0, nrow(frame))
  } else if (label == "always") {
    rep(1, nrow(frame))
  } else {
    as.numeric(!is.na(frame$L2) & frame$L2 > 0)
  }
  cbind(rep(a1, nrow(frame)), a2)
}

mechanism <- function(frame, abar) {
  l2 <- ifelse(is.na(frame$L2), 0, frame$L2)
  p_a1 <- plogis(0.3 * frame$W1 - 0.4 * frame$W2)
  p_c1 <- plogis(2.2 + 0.3 * frame$W1 - 0.3 * abar[, 1])
  p_a2 <- plogis(0.5 * l2 + 0.6 * abar[, 1] - 0.2 * frame$W2)
  p_c2 <- plogis(2.4 + 0.2 * l2)
  cbind(p_a1, p_c1, p_a2, p_c2)
}

fit_regimen <- function(frame, label) {
  data <- frame[c("W1", "W2", "A1", "C1", "L2", "A2", "C2", "Y")]
  abar <- assigned(frame, label)
  g <- mechanism(frame, abar)
  arguments <- list(
    data = data,
    Anodes = c("A1", "A2"),
    Cnodes = c("C1", "C2"),
    Lnodes = "L2",
    Ynodes = "Y",
    survivalOutcome = FALSE,
    Qform = qform,
    gform = g,
    abar = abar,
    gbounds = c(1e-8, 1),
    SL.library = "glm",
    stratify = TRUE,
    variance.method = "ic"
  )
  targeted <- withCallingHandlers(
    do.call(ltmle, arguments),
    warning = function(condition) {
      if (grepl("Cnodes|censoring", conditionMessage(condition), ignore.case = TRUE)) {
        invokeRestart("muffleWarning")
      }
      stop(sprintf("unexpected ltmle warning: %s", conditionMessage(condition)))
    }
  )
  if (any(abs(targeted$cum.g - targeted$cum.g.unbounded) > 1e-12, na.rm = TRUE)) {
    stop(sprintf("%s activated a cumulative g bound", label))
  }
  first_q <- targeted$fit$Q[[1]]
  initial <- if (inherits(first_q, "no.Y.variation")) {
    unname(first_q$Y.value)
  } else {
    coefficients <- first_q[, "Estimate"]
    design <- model.matrix(~ W1 + W2, data = frame)
    unname(mean(plogis(design[, names(coefficients), drop = FALSE] %*% coefficients)))
  }
  list(
    estimate = unname(targeted$estimates[["tmle"]]),
    initial = initial,
    ic = as.numeric(targeted$IC$tmle)
  )
}

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  fits <- setNames(lapply(regimens, function(label) fit_regimen(frame, label)), regimens)
  design <- cbind(1, unname(duration[regimens]))
  weights <- unname(projection_weight[regimens])
  operator <- solve(t(design) %*% (design * weights), t(design * weights))
  estimates <- vapply(fits, `[[`, numeric(1), "estimate")
  initials <- vapply(fits, `[[`, numeric(1), "initial")
  influence <- do.call(cbind, lapply(fits, `[[`, "ic")) %*% t(operator)
  beta <- as.numeric(operator %*% estimates)
  initial_beta <- as.numeric(operator %*% initials)
  standard_error <- apply(influence, 2, sd) / sqrt(nrow(frame))
  low <- beta - z * standard_error
  high <- beta + z * standard_error

  selected <- truths$replicate == replicate & truths$scenario == scenario
  truth <- truths$truth[selected][match(labels, truths$estimand[selected])]
  if (any(is.na(truth))) stop(sprintf("truth join failed for replicate %s", replicate))
  data.frame(
    implementation = "ltmle projected regimen fits",
    scenario = scenario,
    replicate = replicate,
    n = nrow(frame),
    estimand = labels,
    truth = truth,
    estimate = beta,
    inference_estimate = beta,
    std_error = standard_error,
    ci_lower = low,
    ci_upper = high,
    inference_scale = "identity",
    covered = as.integer(low <= truth & truth <= high),
    initial_estimate = initial_beta,
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
cat("ltmle ", as.character(packageVersion("ltmle")), "\n", sep = "")
cat("R ", R.version.string, "\n", sep = "")
