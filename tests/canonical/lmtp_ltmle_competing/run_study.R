suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_crossfit_adapter.R")
source("/fixture/lmtp_competing_adapter.R")
options(digits = 17)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
sample_path <- args[[1]]
truths <- read.csv(args[[2]], stringsAsFactors = FALSE, check.names = FALSE)
output <- args[[3]]

scenario <- "censored_competing_risk_curve"
causes <- c("relapse", "death")
z <- qnorm(0.975)

assigned <- function(frame, label, horizon) {
  a1 <- if (label == "never") 0 else 1
  if (horizon == 1) return(matrix(rep(a1, nrow(frame)), ncol = 1))
  a2 <- if (label == "never") {
    rep(0, nrow(frame))
  } else if (label == "always") {
    rep(1, nrow(frame))
  } else {
    as.numeric(!is.na(frame$L2) & frame$L2 == 1)
  }
  cbind(rep(a1, nrow(frame)), a2)
}

exact_ratios <- function(frame, arms, horizon) {
  w <- as.integer(frame$W)
  a1 <- as.integer(arms[, 1])
  p_a1 <- ifelse(w == 0, 0.50, 0.25)
  g1 <- ifelse(a1 == 1, p_a1, 1 - p_a1)
  c1 <- ifelse(w == 0 & a1 == 1, 0.50, 0.75)
  followed1 <- frame$A1 == a1 & frame$C1 == 1
  first <- ifelse(followed1, 1 / (g1 * c1), 0)
  if (horizon == 1) return(matrix(first, ncol = 1))

  l2 <- as.integer(ifelse(is.na(frame$L2), 0, frame$L2))
  a2 <- as.integer(arms[, 2])
  p_a2 <- ifelse(
    w == 0 & a1 == 0 & l2 == 0, 0.50,
    ifelse(
      w == 0 & a1 == 0 & l2 == 1, 0.75,
      ifelse(
        w == 0 & a1 == 1 & l2 == 0, 0.25,
        ifelse(
          w == 0 & a1 == 1 & l2 == 1, 0.50,
          ifelse(
            w == 1 & a1 == 0 & l2 == 0, 0.75,
            ifelse(w == 1 & a1 == 0 & l2 == 1, 0.50,
              ifelse(w == 1 & a1 == 1 & l2 == 0, 0.50, 0.25)
            )
          )
        )
      )
    )
  )
  g2 <- ifelse(a2 == 1, p_a2, 1 - p_a2)
  c2_zero <- ifelse(
    w == 0 & a1 == 0 & l2 == 0, 0.75,
    ifelse(
      w == 0 & a1 == 0 & l2 == 1, 0.75,
      ifelse(
        w == 0 & a1 == 1 & l2 == 0, 0.50,
        ifelse(
          w == 0 & a1 == 1 & l2 == 1, 0.75,
          ifelse(
            w == 1 & a1 == 0 & l2 == 0, 0.75,
            ifelse(w == 1 & a1 == 0 & l2 == 1, 0.50,
              ifelse(w == 1 & a1 == 1 & l2 == 0, 0.75, 0.75)
            )
          )
        )
      )
    )
  )
  c2_one <- ifelse(
    w == 0 & a1 == 0 & l2 == 0, 0.50,
    ifelse(
      w == 0 & a1 == 0 & l2 == 1, 0.75,
      ifelse(
        w == 0 & a1 == 1 & l2 == 0, 0.75,
        ifelse(
          w == 0 & a1 == 1 & l2 == 1, 0.50,
          ifelse(
            w == 1 & a1 == 0 & l2 == 0, 0.75,
            ifelse(w == 1 & a1 == 0 & l2 == 1, 0.75,
              ifelse(w == 1 & a1 == 1 & l2 == 0, 0.50, 0.75)
            )
          )
        )
      )
    )
  )
  c2 <- ifelse(a2 == 1, c2_one, c2_zero)
  followed2 <- !is.na(frame$A2) & frame$A2 == a2 & !is.na(frame$C2) & frame$C2 == 1
  cbind(first, ifelse(followed2, 1 / (g2 * c2), 0))
}

fit_plan <- function(frame, label, cause, horizon) {
  target <- if (cause == "relapse") c("R1", "R2") else c("D1", "D2")
  other <- if (cause == "relapse") c("D1", "D2") else c("R1", "R2")
  if (horizon == 1) {
    natural <- frame[c("W", "A1", "C1", target[[1]])]
    trt <- "A1"
    outcome <- target[[1]]
    compete <- NULL
    time_vary <- list(NULL)
    cens <- "C1"
  } else {
    natural <- frame[c(
      "W", "A1", "C1", target[[1]], other[[1]], "L2", "A2", "C2",
      target[[2]], other[[2]]
    )]
    trt <- c("A1", "A2")
    outcome <- target
    compete <- other
    time_vary <- list(NULL, "L2")
    cens <- c("C1", "C2")
  }
  shifted <- natural
  arms <- assigned(frame, label, horizon)
  shifted$A1 <- arms[, 1]
  if (horizon == 2) shifted$A2 <- arms[, 2]
  fit <- lmtp_competing_tmle_with_folds(
    natural,
    shifted,
    trt = trt,
    outcome = outcome,
    compete = compete,
    baseline = "W",
    time_vary = time_vary,
    cens = cens,
    outcome_type = if (horizon == 1) "binomial" else "survival",
    fold_assignment = frame$fold,
    learners_outcome = "SL.mean",
    learners_trt = "SL.glm",
    density_ratios = exact_ratios(frame, arms, horizon),
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
  if (horizon == 1) {
    list(estimate = fit$estimate@x, initial = mean(fit$initial), ic = fit$estimate@eif)
  } else {
    list(
      estimate = 1 - fit$estimate@x,
      initial = 1 - mean(fit$initial),
      ic = -fit$estimate@eif
    )
  }
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
  rows <- list()
  for (cause in causes) {
    fits <- list(
      "never @ t=1" = fit_plan(frame, "never", cause, 1),
      "never @ t=2" = fit_plan(frame, "never", cause, 2),
      "always @ t=1" = fit_plan(frame, "always", cause, 1),
      "always @ t=2" = fit_plan(frame, "always", cause, 2),
      "continue_if_l2 @ t=2" = fit_plan(frame, "continue_if_l2", cause, 2)
    )
    for (key in names(fits)) {
      fit <- fits[[key]]
      pieces <- strsplit(key, " @ t=", fixed = TRUE)[[1]]
      rows[[length(rows) + 1]] <- row_for(
        replicate,
        sprintf("cif_regimen[%s, %s @ t=%s]", pieces[[1]], cause, pieces[[2]]),
        fit$estimate,
        fit$initial,
        fit$ic,
        nrow(frame)
      )
    }
    comparisons <- list(
      c("always", "1", "always @ t=1", "never @ t=1"),
      c("always", "2", "always @ t=2", "never @ t=2"),
      c("continue_if_l2", "2", "continue_if_l2 @ t=2", "never @ t=2")
    )
    for (spec in comparisons) {
      left <- fits[[spec[[3]]]]
      right <- fits[[spec[[4]]]]
      rows[[length(rows) + 1]] <- row_for(
        replicate,
        sprintf(
          "ate_regimen[%s vs never, %s @ t=%s]", spec[[1]], cause, spec[[2]]
        ),
        left$estimate - right$estimate,
        left$initial - right$initial,
        left$ic - right$ic,
        nrow(frame)
      )
    }
  }
  do.call(rbind, rows)
}

expected <- length(unique(truths$replicate))
completed <- 0L

fit_group <- function(frame) {
  tryCatch(
    fit_one(frame),
    error = function(condition) structure(
      list(replicate = frame$replicate[[1]], message = conditionMessage(condition)),
      class = "study_error"
    )
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
  # Keep enough complete panels in one batch for each worker to reuse its loaded R packages.
  # At the registered n this is 127 replications plus one carried partial panel.
  lines <- readLines(connection, n = 512000L)
  at_end <- length(lines) == 0L
  current <- if (at_end) NULL else as.data.frame(
    data.table::fread(text = paste(c(header, lines), collapse = "\n"))
  )
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
