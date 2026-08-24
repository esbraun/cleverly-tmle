suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_crossfit_adapter.R")
options(digits = 17)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
sample_path <- args[[1]]
truths <- read.csv(args[[2]], stringsAsFactors = FALSE, check.names = FALSE)
output <- args[[3]]

scenario <- "censored_end_of_study"
dynamic_label <- "treat then continue if l2 positive"
plans <- c("never", "always", dynamic_label)
z <- qnorm(0.975)

assigned <- function(frame, label) {
  a1 <- if (label == "never") 0 else 1
  a2 <- if (label == "never") {
    rep(0, nrow(frame))
  } else if (label == "always") {
    rep(1, nrow(frame))
  } else {
    as.numeric(!is.na(frame$L2) & frame$L2 > 0)
  }
  cbind(rep(a1, nrow(frame)), a2)
}

fit_plan <- function(frame, label) {
  natural <- frame[c("W1", "W2", "A1", "C1", "L2", "A2", "C2", "Y")]
  shifted <- natural
  arms <- assigned(frame, label)
  shifted$A1 <- arms[, 1]
  shifted$A2 <- arms[, 2]
  fit <- lmtp_tmle_with_folds(
    natural,
    shifted,
    trt = c("A1", "A2"),
    outcome = "Y",
    baseline = c("W1", "W2"),
    time_vary = list(NULL, "L2"),
    cens = c("C1", "C2"),
    outcome_type = "binomial",
    fold_assignment = frame$fold,
    learners_outcome = "SL.glm",
    learners_trt = "SL.glm",
    control = lmtp_control(
      .trim = 1,
      .learners_outcome_folds = 2,
      .learners_trt_folds = 2,
      .return_full_fits = TRUE
    )
  )
  if (!identical(fit$fold_assignment, as.integer(frame$fold))) {
    stop("lmtp did not retain the supplied fold assignment")
  }
  list(
    estimate = fit$estimate@x,
    initial = mean(fit$initial),
    ic = fit$estimate@eif
  )
}

truth_for <- function(replicate, estimand) {
  selected <- truths$replicate == replicate & truths$scenario == scenario & truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows", sum(selected)))
  truths$truth[selected]
}

row_for <- function(replicate, name, estimate, initial, ic, n) {
  truth <- truth_for(replicate, name)
  standard_error <- sd(ic) / sqrt(n)
  low <- estimate - z * standard_error
  high <- estimate + z * standard_error
  data.frame(
    implementation = "lmtp",
    scenario = scenario,
    replicate = replicate,
    n = n,
    estimand = name,
    truth = truth,
    estimate = estimate,
    inference_estimate = estimate,
    std_error = standard_error,
    ci_lower = low,
    ci_upper = high,
    inference_scale = "identity",
    covered = as.integer(low <= truth && truth <= high),
    initial_estimate = initial,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  fits <- setNames(lapply(plans, function(label) fit_plan(frame, label)), plans)
  rows <- lapply(plans, function(label) {
    fit <- fits[[label]]
    row_for(
      replicate,
      sprintf("ey_regimen[%s]", label),
      fit$estimate,
      fit$initial,
      fit$ic,
      nrow(frame)
    )
  })
  for (label in c("always", dynamic_label)) {
    left <- fits[[label]]
    right <- fits[["never"]]
    rows[[length(rows) + 1]] <- row_for(
      replicate,
      sprintf("ate_regimen[%s vs never]", label),
      left$estimate - right$estimate,
      left$initial - right$initial,
      left$ic - right$ic,
      nrow(frame)
    )
  }
  do.call(rbind, rows)
}

expected <- length(unique(truths$replicate))
completed <- 0L

fit_group <- function(frame) {
  tryCatch(
    {
      result <- fit_one(frame)
      result
    },
    error = function(condition) {
      structure(
        list(replicate = frame$replicate[[1]], message = conditionMessage(condition)),
        class = "study_error"
      )
    }
  )
}

requested <- suppressWarnings(as.integer(Sys.getenv("CLEVERLY_R_CORES", "")))
cores <- if (is.na(requested) || requested < 1L) max(1L, parallel::detectCores()) else requested
cat(sprintf("fitting %d samples on %d cores\n", expected, cores))
connection <- gzfile(sample_path, open = "rt")
header <- readLines(connection, n = 1L)
carry <- NULL
results <- list()
repeat {
  lines <- readLines(connection, n = 64000L)
  at_end <- length(lines) == 0L
  current <- if (at_end) {
    NULL
  } else {
    as.data.frame(data.table::fread(text = paste(c(header, lines), collapse = "\n")))
  }
  frame <- if (is.null(carry)) current else if (is.null(current)) carry else rbind(carry, current)
  carry <- NULL
  if (!is.null(frame) && nrow(frame) && !at_end) {
    last_replicate <- frame$replicate[[nrow(frame)]]
    carry <- frame[frame$replicate == last_replicate, ]
    frame <- frame[frame$replicate != last_replicate, ]
  }
  if (!is.null(frame) && nrow(frame)) {
    groups <- split(frame, frame$replicate)
    batch <- parallel::mclapply(groups, fit_group, mc.cores = cores, mc.preschedule = TRUE)
    failed <- vapply(batch, inherits, logical(1), what = "study_error")
    if (any(failed)) {
      messages <- vapply(
        batch[failed],
        function(error) sprintf("replicate %s: %s", error$replicate, error$message),
        character(1)
      )
      stop(paste(messages, collapse = "\n"))
    }
    results <- c(results, batch)
    completed <- completed + length(batch)
    cat(sprintf("completed %d/%d samples\n", completed, expected))
  }
  if (at_end) break
}
close(connection)
result <- do.call(rbind, results)
if (length(unique(result$replicate)) != expected) stop("the R run dropped replications")
write.csv(result, output, row.names = FALSE, na = "")
cat("lmtp ", as.character(packageVersion("lmtp")), "\n", sep = "")
cat("R ", R.version.string, "\n", sep = "")
